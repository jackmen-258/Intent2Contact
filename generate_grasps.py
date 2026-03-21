import os
import argparse
import pickle
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.utils.config import get_config
from lib.datasets.oishape_dataset import OIShape
from lib.diffusion.latent_diffusion_model import LatentHandDiffusion
from manotorch.manolayer import ManoLayer, MANOOutput
from lib.contact.hand_object import HandObject
from lib.datasets.utils import CENTER_IDX

from lib.datasets.grab_dataset import GRABTest

def load_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    prefix = f"model-diffusion-"
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]

    def _extract_step(fname: str) -> int:
        stem = os.path.splitext(fname)[0]
        parts = stem.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unrecognized checkpoint filename: {fname}")
        return int(parts[-1])

    checkpoint_files.sort(key=_extract_step)
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(checkpoint_dir, latest_checkpoint)

def setup_models(device, diffusion_checkpoint=None, checkpoint_dir=None):
    """
    设置模型
    
    Args:
        device: 设备
        diffusion_checkpoint: checkpoint 文件路径
        checkpoint_dir: checkpoint 目录路径（用于判断实验配置）
    """
    models = {}
    
    if diffusion_checkpoint:
        # ✅ 根据 checkpoint_dir 路径判断配置
        fusion_type = "bi_attn"
        disable_intent = False
        
        if checkpoint_dir is not None:
            if 'concat' in checkpoint_dir.lower():
                fusion_type = "concat"
                print(f"[INFO] Detected 'concat' in checkpoint_dir, using fusion_type='concat'")
            
            if 'without_intent_cond' in checkpoint_dir.lower():
                disable_intent = True
                print(f"[INFO] Detected 'without_intent_cond' in checkpoint_dir, setting disable_intent=True")
        
        print(f"Loading Diffusion model from {diffusion_checkpoint}")
        print(f"  fusion_type: {fusion_type}")
        print(f"  disable_intent: {disable_intent}")
        
        diffusion_model = LatentHandDiffusion(
            params_dim=61, 
            latent_dim=64,
            fusion_type=fusion_type,
            disable_intent=disable_intent
        ).to(device)
        
        checkpoint = torch.load(diffusion_checkpoint, map_location=device)
        diffusion_model.load_state_dict(checkpoint['model'])
        diffusion_model.eval()
        models['diffusion'] = diffusion_model
    
    mano_layer = ManoLayer(center_idx=CENTER_IDX, mano_assets_root="assets/mano_v1_2").to(device)
    models['mano'] = mano_layer

    hand_object = HandObject(device)
    models['hand_object'] = hand_object
    
    return models

def generate_hand_mesh(models, obj_verts, obj_vn, intent_id):
    mano_params = models['diffusion'].sample(obj_verts, obj_vn, intent_id)

    pose = mano_params[:, :48]
    trans = mano_params[:, 48:51]
    shape = mano_params[:, 51:]

    mano_output: MANOOutput = models['mano'](pose, shape)

    hand_verts = mano_output.verts + trans.unsqueeze(1)
    hand_joints = mano_output.joints + trans.unsqueeze(1)

    return hand_verts, hand_joints, mano_params

def _extract_hand_gt_verts(grasp, mano_layer) -> np.ndarray | None:
    """
    尝试从 dataloader batch 中拿到 GT 手网格顶点（778,3）
    """
    # 1) 直接给顶点
    for k in ("hand_verts_gt", "hand_verts", "hand_gt_verts", "gt_hand_verts"):
        if k in grasp:
            v = grasp[k]
            if torch.is_tensor(v):
                v = v[0].detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 2 and v.shape[1] == 3:
                return v
            if v.ndim == 3 and v.shape[-1] == 3:
                return v[0]
            return None

    # 2) 用 MANO 参数重建
    pose = None
    shape = None
    trans = None

    for k in ("hand_pose", "mano_pose", "pose"):
        if k in grasp:
            pose = grasp[k]
            break
    for k in ("hand_shape", "mano_shape", "shape", "betas"):
        if k in grasp:
            shape = grasp[k]
            break
    for k in ("hand_trans", "mano_trans", "trans", "translation"):
        if k in grasp:
            trans = grasp[k]
            break

    if pose is None:
        return None

    if not torch.is_tensor(pose):
        pose = torch.as_tensor(pose)
    pose = pose.to(next(mano_layer.parameters()).device).float()
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)

    if shape is None:
        shape = torch.zeros((pose.shape[0], 10), device=pose.device, dtype=torch.float32)
    else:
        if not torch.is_tensor(shape):
            shape = torch.as_tensor(shape)
        shape = shape.to(pose.device).float()
        if shape.ndim == 1:
            shape = shape.unsqueeze(0)

    if trans is None:
        trans = torch.zeros((pose.shape[0], 3), device=pose.device, dtype=torch.float32)
    else:
        if not torch.is_tensor(trans):
            trans = torch.as_tensor(trans)
        trans = trans.to(pose.device).float()
        if trans.ndim == 1:
            trans = trans.unsqueeze(0)

    with torch.no_grad():
        mano_out: MANOOutput = mano_layer(pose, shape)
        verts = mano_out.verts + trans.unsqueeze(1)

    return verts[0].detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate grasps for evaluation')
    parser.add_argument('--oishape_cfg', default='config/all_cate.yaml')
    parser.add_argument('--dataset', type=str, default='oishape', 
                       choices=['oishape', 'grab'],
                       help='Dataset type for evaluation')
    parser.add_argument('--grab_cfg', type=str, default='config/grab_test.yaml',
                       help='GRAB dataset config (only used when --dataset=grab)')
    parser.add_argument('--save_root', default='experiments/exp_001', type=str, 
                       help='Experiment root directory')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints')
    
    parser.add_argument('--save_mesh', action='store_true', 
                       help='Also save .ply meshes for visualization')
    
    args = parser.parse_args()
    
    results_dir = os.path.join(args.save_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    args.device = device

    # ✅ 根据 --dataset 加载不同数据集
    if args.dataset == 'oishape':
        cfg = get_config(config_file=args.oishape_cfg)
        test_ds = OIShape(cfg.DATASET.TEST)
    else:  # grab
        cfg = get_config(config_file=args.grab_cfg)
        test_ds = GRABTest(cfg.DATASET.TEST)
    
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    diffusion_checkpoint = load_latest_checkpoint(args.checkpoint_dir)
    models = setup_models(device, diffusion_checkpoint, args.checkpoint_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Results will be saved in: {results_dir}")
    print(f"Total test samples: {len(test_ds)}")
    print("-" * 60)

    for idx, grasp in enumerate(tqdm(test_dl, desc="Generating grasps")):
        obj_verts = grasp["obj_verts"].to(device)
        obj_vn = grasp["obj_vn"].to(device)
        intent_id = grasp["intent_id"].to(device)
        obj_id = grasp["obj_id"][0]
        intent_name = grasp["intent_name"][0]
        sample_idx = grasp["sample_idx"].item()  # ✅ 提取样本编号

        hand_verts, hand_joints, mano_params = generate_hand_mesh(models, obj_verts, obj_vn, intent_id)
        hand_verts_np = hand_verts[0].cpu().numpy()
        hand_joints_np = hand_joints[0].cpu().numpy()

        obj_rotmat = grasp["obj_rotmat"].cpu().numpy()
        if obj_rotmat.ndim == 3:
            obj_rotmat = obj_rotmat[0]

        grasp_result = {
            "obj_id": obj_id,
            "intent_name": intent_name,
            "hand_verts_r": hand_verts_np,
            "hand_joints_r": hand_joints_np,
            "obj_rotmat": obj_rotmat,
        }

        # ✅ 文件名包含样本编号
        save_path = os.path.join(results_dir, f"{obj_id}_{intent_name}_{sample_idx:04d}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(grasp_result, f)

        if args.save_mesh:
            mesh_dir = os.path.join(results_dir, f"{obj_id}_{intent_name}_{idx:04d}_vis")
            os.makedirs(mesh_dir, exist_ok=True)

            hand_faces = models['mano'].th_faces.detach().cpu().numpy()
            hand_pred_mesh = trimesh.Trimesh(vertices=hand_verts_np, faces=hand_faces, process=False)
            hand_pred_mesh.export(os.path.join(mesh_dir, "hand_pred.ply"))

            # ✅ GRAB 没有 GT，跳过 GT 保存
            if args.dataset == 'oishape':
                hand_gt_verts = _extract_hand_gt_verts(grasp, models['mano'])
                if hand_gt_verts is not None:
                    hand_gt_mesh = trimesh.Trimesh(vertices=hand_gt_verts, faces=hand_faces, process=False)
                    hand_gt_mesh.export(os.path.join(mesh_dir, "hand_gt.ply"))

            obj_mesh = test_ds.obj_warehouse[obj_id]
            obj_mesh.export(os.path.join(mesh_dir, "object.ply"))

    print(f"\n✅ Done! Generated {len(test_ds)} grasps")
    print(f"\nNow you can evaluate with:")
    print(f"  python evaluate_grasps.py --exp_path {args.save_root} --mode quality")
    print(f"  python evaluate_grasps.py --exp_path {args.save_root} --mode diversity")