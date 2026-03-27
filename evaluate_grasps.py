import argparse
import os
import pickle
import numpy as np
import torch
import trimesh
from joblib import Parallel, delayed
from trimesh import Trimesh
import torch.nn.functional as F

from lib.metrics.basic_metric import AverageMeter
from lib.metrics.penetration import penetration
from lib.metrics.intersection import solid_intersection_volume
from lib.metrics.simulator import simulation_sample
from lib.metrics.diversity import diversity
from lib.metrics.diversity import transform_to_canonical, convert_joints

from lib.contact.hand_object import HandObject
from lib.metrics.affordance_accuracy import compute_affordance_metrics_batch

def _get_unit_scales(linear_unit: str):
    """
    返回线性与体积的换算倍率：
      - 若 linear_unit == 'cm'：米->厘米倍率=100，立方米->立方厘米倍率=100^3
      - 若 linear_unit == 'm'：倍率=1
    """
    linear_unit = (linear_unit or "cm").lower()
    if linear_unit == "cm":
        s = 100.0
    elif linear_unit == "m":
        s = 1.0
    else:
        raise ValueError(f"Unsupported linear_unit: {linear_unit}")
    return s, s ** 3

def _unit_scale(unit: str) -> float:
    unit = (unit or "mm").lower()
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 100.0
    if unit == "mm":
        return 1000.0
    raise ValueError(f"Unsupported diversity_unit: {unit}")

class DumpedGraspsLoader(object):
    def __init__(self, dumped_grasps_dir, proc_dir):
        self.dumped_grasps_dir = dumped_grasps_dir
        self.proc_dir = proc_dir
        grasp_fname = sorted(os.listdir(dumped_grasps_dir))
        grasp_fname = [el for el in grasp_fname if el.endswith(".pkl")]
        self.grasp_files = [os.path.join(dumped_grasps_dir, el) for el in grasp_fname]
        
        with open("assets/closed_mano_faces.pkl", "rb") as f:
            faces = pickle.load(f)
        self.hand_wt_faces = np.asarray(faces, dtype=np.int32)
        
        # ✅ 新增：预先检查缺失物体
        self._check_missing_objects()
        
    def _check_missing_objects(self):
        """预先检查哪些物体文件缺失"""
        print("Checking for missing object files...")
        missing_objs = {}
        
        for grasp_file in self.grasp_files:
            with open(grasp_file, "rb") as f:
                grasp_item = pickle.load(f)
            obj_id = grasp_item["obj_id"]
            
            obj_wt_path = os.path.join(self.proc_dir, "watertight", f"{obj_id}.obj")
            obj_vox_path = os.path.join(self.proc_dir, "voxel", f"{obj_id}.binvox")
            obj_vhacd_path = os.path.join(self.proc_dir, "vhacd", f"{obj_id}.obj")
            
            missing_types = []
            if not os.path.exists(obj_wt_path):
                missing_types.append("watertight")
            if not os.path.exists(obj_vox_path):
                missing_types.append("voxel")
            if not os.path.exists(obj_vhacd_path):
                missing_types.append("vhacd")
            
            if missing_types:
                if obj_id not in missing_objs:
                    missing_objs[obj_id] = missing_types
        
        if missing_objs:
            print(f"\n⚠️  Found {len(missing_objs)} objects with missing files:")
            for obj_id, types in sorted(missing_objs.items())[:10]:  # 只显示前10个
                print(f"   {obj_id}: missing {', '.join(types)}")
            if len(missing_objs) > 10:
                print(f"   ... and {len(missing_objs) - 10} more")
            
            print(f"\n💡 Tip: Run preprocessing for missing objects:")
            print(f"   cd OakInk-Grasp-Generation")
            obj_ids_str = ' '.join(sorted(missing_objs.keys())[:20])
            print(f"   python scripts/process_obj_mesh.py --obj_id {obj_ids_str}")
            print(f"\n⚠️  These samples will be SKIPPED during evaluation.\n")
        else:
            print("✅ All object files present!\n")
        
    def __len__(self):
        return len(self.grasp_files)

    def __getitem__(self, idx):
        """
        加载单个样本，如果文件缺失则返回 None
        """
        with open(self.grasp_files[idx], "rb") as f:
            grasp_item = pickle.load(f)

        sample_id = os.path.basename(self.grasp_files[idx]).split(".")[0]
        grasp_item["sample_id"] = sample_id

        obj_id = grasp_item["obj_id"]
        
        # ✅ 检查文件是否存在
        obj_wt_path = os.path.join(self.proc_dir, "watertight", f"{obj_id}.obj")
        obj_vox_path = os.path.join(self.proc_dir, "voxel", f"{obj_id}.binvox")
        obj_vhacd_path = os.path.join(self.proc_dir, "vhacd", f"{obj_id}.obj")
        
        # ✅ 如果任何文件缺失，返回 None（表示跳过该样本）
        if not os.path.exists(obj_wt_path):
            return None
        if not os.path.exists(obj_vox_path):
            return None
        if not os.path.exists(obj_vhacd_path):
            return None
        
        obj_wt = trimesh.load(obj_wt_path, process=False)
        obj_vox = trimesh.load(obj_vox_path)

        grasp_item["obj_wt"] = obj_wt
        grasp_item["obj_vox"] = obj_vox
        grasp_item["obj_vhacd_path"] = obj_vhacd_path
        grasp_item["hand_faces"] = self.hand_wt_faces
        return grasp_item

def _sample_obj_points_with_normals(obj_wt: Trimesh, n_points: int):
    """
    从 watertight mesh 表面采样点，使用对应三角面法向作为点法向。
    返回:
      points: (N,3) float32
      normals:(N,3) float32
    """
    pts, face_idx = trimesh.sample.sample_surface(obj_wt, n_points)
    pts = np.asarray(pts, dtype=np.float32)
    face_idx = np.asarray(face_idx, dtype=np.int64)
    nrm = np.asarray(obj_wt.face_normals[face_idx], dtype=np.float32)
    return pts, nrm


@torch.no_grad()
def _contact_map_via_hand_object(
    hand_object: HandObject,
    hand_verts_np: np.ndarray,   # (778,3)
    obj_wt: Trimesh,
    n_points: int,
    device: torch.device,
):
    """
    用 HandObject 生成 contact_map（训练一致）：
      输入：hand verts + 采样的 obj points/normals
      输出：obj_pts (1,N,3), obj_n (1,N,3), contact_map (1,N,1)
    """
    obj_pts_np, obj_n_np = _sample_obj_points_with_normals(obj_wt, n_points)

    hand_verts = torch.from_numpy(np.asarray(hand_verts_np, dtype=np.float32)).to(device).unsqueeze(0)  # (1,778,3)
    obj_pts = torch.from_numpy(obj_pts_np).to(device).unsqueeze(0)                                      # (1,N,3)
    obj_n = torch.from_numpy(obj_n_np).to(device).unsqueeze(0)                                          # (1,N,3)

    out = hand_object.forward(hand_verts=hand_verts, obj_verts=obj_pts, obj_vn=obj_n)
    contact_map = out["contacts_object"]  # 期望形状 (1,N,1)

    # 保险：确保维度正确
    if contact_map.ndim == 2:
        contact_map = contact_map.unsqueeze(-1)
    return obj_pts, obj_n, contact_map

def evaluate_single_grasp(idx, grasp_loader, sims_dir):
    """
    评估单个抓取样本
    返回：包含评估结果的字典，如果样本被跳过则返回 None
    """
    grasp_item = grasp_loader[idx]
    
    # ✅ 如果样本被跳过（返回 None）
    if grasp_item is None:
        return None
    
    sample_id = grasp_item["sample_id"]
    
    obj_wt: Trimesh = grasp_item["obj_wt"]
    obj_vox: Trimesh = grasp_item["obj_vox"]
    obj_vhacd_path: str = grasp_item["obj_vhacd_path"]
    obj_rotmat = grasp_item["obj_rotmat"]
    hand_verts = grasp_item["hand_verts_r"]
    hand_faces = grasp_item["hand_faces"]

    obj_wt_verts = np.asarray(obj_wt.vertices, dtype=np.float32)
    obj_wt_faces = np.array(obj_wt.faces, dtype=np.int32)
    obj_vox_points = np.asfarray(obj_vox.points, dtype=np.float32)
    obj_element_volume = obj_vox.element_volume

    pentr_dep = penetration(obj_verts=obj_wt_verts, obj_faces=obj_wt_faces, hand_verts=hand_verts)

    pentr_vol = solid_intersection_volume(
        hand_verts=hand_verts,
        hand_faces=hand_faces,
        obj_vox_points=obj_vox_points,
        obj_vox_el_vol=obj_element_volume,
        return_kin=False
    )

    sims_disp = simulation_sample(
        sample_idx=idx,
        sample_info={
            "sample_id": sample_id,
            "hand_verts": hand_verts,
            "hand_faces": hand_faces,
            "obj_verts": obj_wt_verts,
            "obj_faces": obj_wt_faces,
            "obj_vhacd_fname": obj_vhacd_path,
            "obj_rotmat": obj_rotmat
        },
        save_gif_folder=os.path.join(sims_dir, "gif"),
        save_obj_folder=os.path.join(sims_dir, "vhacd"),
        tmp_folder=os.path.join(sims_dir, "tmp"),
        use_gui=False,
        sample_vis_freq=1
    )

    eval_res = {
        "sample_id": sample_id,
        "pentr_dep": pentr_dep,
        "pentr_vol": pentr_vol,
        "sims_disp": sims_disp,
        "hand_verts": hand_verts,
        "obj_id": grasp_item["obj_id"],
        "intent_name": grasp_item.get("intent_name", "unknown"),
    }
    return eval_res


def evaluate_quality_mode(arg):
    """评估核心质量指标（Penetration, Intersection, Simulation, Contact）"""
    # ✅ 修改：统一从 results 目录读取
    dumped_grasps_dir = os.path.join(arg.exp_path, "results")
    simulation_dir = os.path.join(arg.exp_path, "simulation_quality")
    evaluation_dir = os.path.join(arg.exp_path, "evaluations_quality")
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    dumped_grasps_loader = DumpedGraspsLoader(dumped_grasps_dir, arg.proc_dir)

    task_list = []
    print(f"[Quality Mode] Evaluating {len(dumped_grasps_loader)} grasps...")
    for i in range(len(dumped_grasps_loader)):
        task_list.append(delayed(evaluate_single_grasp)(i, dumped_grasps_loader, simulation_dir))

    pentr_dep = AverageMeter("pentr_dep")
    pentr_vol = AverageMeter("pentr_vol")
    sims_disp = AverageMeter("sims_disp")
    
    sims_disp_list = []
    contact_count = 0

    eval_res = Parallel(n_jobs=arg.n_jobs, verbose=10)(task_list)
    
    valid_res = [r for r in eval_res if r is not None]
    skipped_count = len(eval_res) - len(valid_res)
    
    if skipped_count > 0:
        print(f"\n⚠️  Skipped {skipped_count}/{len(eval_res)} samples due to missing files")
        print(f"✅  Evaluated {len(valid_res)} valid samples\n")
    
    with open(os.path.join(evaluation_dir, "eval_res.pkl"), "wb") as f:
        pickle.dump(valid_res, f)

    for sample_res in valid_res:
        pentr_dep.update(sample_res["pentr_dep"])
        pentr_vol.update(sample_res["pentr_vol"])
        sims_disp.update(sample_res["sims_disp"])
        sims_disp_list.append(sample_res["sims_disp"])
        
        if sample_res["pentr_vol"] > 0:
            contact_count += 1

    contact_ratio = contact_count / len(valid_res) if len(valid_res) > 0 else 0.0

    linear_scale, volume_scale = _get_unit_scales(getattr(arg, "linear_unit", "cm"))
    linear_unit = getattr(arg, "linear_unit", "cm").lower()

    pentr_dep_out = pentr_dep.avg * linear_scale
    pentr_vol_out = pentr_vol.avg * volume_scale
    sims_disp_out = sims_disp.avg * linear_scale
    sims_disp_std_out = float(np.std(sims_disp_list)) * linear_scale if sims_disp_list else 0.0

    # ---- Affordance Accuracy (model-free intent evaluation) ----
    aff_results = None
    if arg.aff_enable:
        print("\n[Affordance] Computing affordance accuracy metrics...")
        aff_results = compute_affordance_metrics_batch(
            dumped_grasps_loader=dumped_grasps_loader,
            oakbase_dir=arg.aff_oakbase_dir,
            meta_dir=arg.aff_meta_dir,
            n_obj_points=int(arg.intent_n_obj_points),
            prior_path=getattr(arg, "aff_prior", None),
        )

    with open(os.path.join(evaluation_dir, "Metric.txt"), "w") as f:
        f.write(f"Total samples: {len(eval_res)}\n")
        f.write(f"Valid samples: {len(valid_res)}\n")
        f.write(f"Skipped samples: {skipped_count}\n\n")
        f.write(f"Penetration Distance (mean, {linear_unit}): {pentr_dep_out:.6f}\n")
        f.write(f"Intersection Volume (mean, {linear_unit}^3): {pentr_vol_out:.6f}\n")
        f.write(f"Simulation Displacement (mean, {linear_unit}): {sims_disp_out:.6f}\n")
        f.write(f"Simulation Displacement (std, {linear_unit}): {sims_disp_std_out:.6f}\n")
        f.write(f"Contact Ratio: {contact_ratio:.4f}\n")
        if aff_results is not None:
            f.write(f"\n--- Affordance Accuracy (model-free) ---\n")
            f.write(f"Prior source: {aff_results['prior_source']}\n")
            f.write(f"Evaluated: {aff_results['n_evaluated']}\n")
            f.write(f"Skipped (missing files): {aff_results['n_skipped']}\n")
            f.write(f"Skipped (no parts): {aff_results['n_no_parts']}\n")
            f.write(f"Contact-Valid Rate: {aff_results['contact_valid_rate']:.4f}\n")
            f.write(f"Aff-Acc (all): {aff_results['aff_acc']:.4f}\n")
            f.write(f"Aff-Ratio (all): {aff_results['aff_ratio']:.4f}\n")
            f.write(f"Aff-Acc (valid only): {aff_results['aff_acc_valid_only']:.4f}\n")
            f.write(f"Aff-Ratio (valid only): {aff_results['aff_ratio_valid_only']:.4f}\n")

    print(f"\n[Quality Mode] Evaluation done! Results saved in: {evaluation_dir}")
    print(f"  Total samples: {len(eval_res)} ")
    print(f"  Valid samples: {len(valid_res)}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Pentr Dist ({linear_unit}): {pentr_dep_out:.6f}")
    print(f"  Pentr Vol ({linear_unit}^3): {pentr_vol_out:.6f}")
    print(f"  Sims Disp ({linear_unit}): {sims_disp_out:.6f} ± {sims_disp_std_out:.6f}")
    print(f"  Contact Ratio: {contact_ratio:.4f}")
    if aff_results is not None:
        print(f"\n  --- Affordance Accuracy (model-free) ---")
        print(f"  Evaluated: {aff_results['n_evaluated']}")
        print(f"  No parts: {aff_results['n_no_parts']}")
        print(f"  Contact-Valid Rate: {aff_results['contact_valid_rate']:.4f}")
        print(f"  Aff-Acc (all): {aff_results['aff_acc']:.4f}")
        print(f"  Aff-Ratio (all): {aff_results['aff_ratio']:.4f}")
        print(f"  Aff-Acc (valid only): {aff_results['aff_acc_valid_only']:.4f}")
        print(f"  Aff-Ratio (valid only): {aff_results['aff_ratio_valid_only']:.4f}")


def evaluate_diversity_mode(arg):
    """
    ✅ HALO 风格（对齐）：
      - feature: MANO hand joints (hand_joints_r)
      - optional: convert_joints(mano->biomech) + transform_to_canonical + convert back
      - scale: default mm
      - entropy: ln (已在 diversity() 中对齐)
    """
    dumped_grasps_dir = os.path.join(arg.exp_path, "results")
    evaluation_dir = os.path.join(arg.exp_path, "evaluations_diversity")
    os.makedirs(evaluation_dir, exist_ok=True)

    if not os.path.isdir(dumped_grasps_dir):
        raise FileNotFoundError(f"Results directory not found: {dumped_grasps_dir}")

    cls_num = int(getattr(arg, "diversity_cls_num", 20))
    unit = getattr(arg, "diversity_unit", "mm")
    scale = _unit_scale(unit)

    use_canonical = bool(getattr(arg, "diversity_canonical", True))

    grasp_files = sorted([f for f in os.listdir(dumped_grasps_dir) if f.endswith(".pkl")])
    grasp_paths = [os.path.join(dumped_grasps_dir, f) for f in grasp_files]

    print(f"[Diversity Mode] Loading {len(grasp_paths)} grasps...")
    print(f"[Diversity Mode] Feature: hand_joints_r (HALO-style), canonical={use_canonical}, unit={unit}, cls_num={cls_num}")

    all_features = []
    skipped_count = 0

    for p in grasp_paths:
        try:
            with open(p, "rb") as f:
                item = pickle.load(f)
        except Exception:
            skipped_count += 1
            continue

        joints = item.get("hand_joints_r", None)
        if joints is None:
            skipped_count += 1
            continue

        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim != 2 or joints.shape[1] != 3:
            skipped_count += 1
            continue

        # ✅ 尺度对齐：m -> (cm/mm)
        joints = joints * scale

        if use_canonical:
            # HALO 的 canonical 流程：mano -> biomech -> canonical -> mano
            with torch.no_grad():
                jt = torch.from_numpy(joints).float().unsqueeze(0)  # (1,J,3) torch
                jt = convert_joints(jt, source="mano", target="biomech")
                jt_after, _T = transform_to_canonical(jt)  # -> torch
                jt_after = convert_joints(jt_after, source="biomech", target="mano")
                joints = jt_after[0].cpu().numpy().astype(np.float32)

        feat = joints.reshape(-1)  # (J*3,)
        all_features.append(feat)

    valid_count = len(all_features)
    if valid_count < 2:
        entropy_val, avg_dist = 0.0, 0.0
    else:
        cluster_array = np.asarray(all_features, dtype=np.float32)
        actual_cls_num = min(valid_count, cls_num)
        print(f"[Diversity Mode] Clustering {valid_count} samples into {actual_cls_num} clusters...")
        entropy_val, avg_dist = diversity(cluster_array, cls_num=actual_cls_num)

    with open(os.path.join(evaluation_dir, "Metric.txt"), "w") as f:
        f.write(f"Total samples: {len(grasp_paths)}\n")
        f.write(f"Valid samples: {valid_count}\n")
        f.write(f"Skipped samples: {skipped_count}\n\n")
        f.write(f"cls_num (global): {cls_num}\n")
        f.write(f"feature: hand_joints_r\n")
        f.write(f"canonical: {use_canonical}\n")
        f.write(f"unit: {unit}\n")
        f.write(f"Diversity Entropy (ln): {float(entropy_val):.4f}\n")
        f.write(f"Diversity Avg Dist: {float(avg_dist):.4f}\n")

    print(f"\n[Diversity Mode] Evaluation done! Results saved in: {evaluation_dir}")
    print(f"  Valid samples: {valid_count}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Diversity Entropy (ln): {float(entropy_val):.4f}")
    print(f"  Diversity Avg Dist: {float(avg_dist):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generated grasps')
    parser.add_argument("-g", "--gpu_id", type=str, default="1")
    parser.add_argument("--n_jobs", type=int, default=8, help="parallel jobs")
    parser.add_argument("--exp_path", type=str, required=True, help="experiment output dir")
    parser.add_argument("--proc_dir", type=str, default="data/GRAB_object_process")
    
    parser.add_argument('--mode', type=str, choices=['quality', 'diversity'], default='quality',
                       help='Evaluation mode: quality or diversity')
    
    parser.add_argument("--diversity_cls_num", type=int, default=20,
                        help="[Diversity mode] KMeans cluster number (HALO default: 20)")
    parser.add_argument("--diversity_canonical", type=int, default=1,
                        help="[Diversity mode] Use HALO canonical transform (1/0). Default: 1")
    parser.add_argument("--diversity_unit", type=str, choices=["m", "cm", "mm"], default="cm",
                        help="[Diversity mode] Unit scaling applied to joints before clustering. Default: mm")
    
    parser.add_argument("--linear_unit", type=str, choices=["m", "cm"], default="cm")
    
    parser.add_argument("--intent_n_obj_points", type=int, default=2048)

    parser.add_argument("--aff_enable", action="store_true",
                        help="Enable affordance accuracy evaluation (model-free)")
    parser.add_argument("--aff_oakbase_dir", type=str, default="/home/kingston/wzc/data/OakInk/OakBase",
                        help="Path to OakBase part segmentation root")
    parser.add_argument("--aff_meta_dir", type=str, default="/home/kingston/wzc/data/OakInk/shape/metaV2",
                        help="Path to OakInk metaV2 directory")
    parser.add_argument("--aff_prior", type=str, default="assets/affordance_prior.json",
                        help="Path to GT affordance prior JSON (from build_affordance_prior.py). "
                             "If file exists, uses data-driven expected parts; otherwise falls back to rules.")

    arg = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)

    if arg.mode == 'quality':
        evaluate_quality_mode(arg)
    else:
        evaluate_diversity_mode(arg)