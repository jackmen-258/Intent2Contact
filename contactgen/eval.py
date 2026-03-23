import os
import random
import argparse
import pickle
import numpy as np
import trimesh
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from manopth.manolayer import ManoLayer

from contactgen.utils.cfg_parser import Config
from contactgen.model import ContactGenModel
from contactgen.hand_sdf.hand_model import ArtiHand
from contactgen.contact.contact_optimizer import optimize_pose
from contactgen.datasets.eval_dataset import TestSet
from contactgen.datasets.oishape_dataset import OIShape, ALL_INTENT
from contactgen.datasets.grab_eval_dataset import GRABEvalDataset  # ✅ 新增


def export_to_pkl(hand_verts, hand_joints, obj_id, obj_rotmat, intent_name, save_path):
    """导出为 Intent2Contact 兼容的 pkl 格式"""
    sample = {
        "obj_id": obj_id,
        "obj_rotmat": obj_rotmat,
        "hand_verts_r": hand_verts.astype(np.float32),
        "hand_joints_r": hand_joints.astype(np.float32),
        "intent_name": intent_name,
    }
    with open(save_path, "wb") as f:
        pickle.dump(sample, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Eval')
    parser.add_argument('--checkpoint', default='exp_on_oishape/checkpoint.pt', type=str)
    parser.add_argument('--dataset', default='oishape', type=str, 
                       choices=['grab', 'oishape', 'grab_eval'])
    parser.add_argument('--n_samples', default=1, type=int, 
                       help='number of samples per generation (batch size)')
    parser.add_argument('--num_samples_per_obj', default=5, type=int,
                       help='number of generations per object')
    parser.add_argument('--save_root', default='exp_on_oishape/eval_on_oishape', type=str)
    parser.add_argument('--seed', default=2026, type=int)
    
    # ✅ 新增物体筛选参数
    parser.add_argument('--obj_ids', type=str, default=None,
                       help='Comma-separated list of object IDs to test (e.g., "SM1,SM2,SM3")')
    
    # GRAB 数据集路径参数
    parser.add_argument('--grab_obj_dir', default='grab_data/obj_meshes', type=str,
                       help='GRAB object directory')
    
    # contact solver options
    parser.add_argument('--w_contact', default=1e-1, type=float)
    parser.add_argument('--w_pene', default=3.0, type=float)
    parser.add_argument('--w_uv', default=1e-2, type=float)

    args = parser.parse_args()
    
    # ✅ 解析指定的物体ID列表
    target_obj_ids = None
    if args.obj_ids:
        target_obj_ids = set([obj_id.strip() for obj_id in args.obj_ids.split(',')])
        print(f"🎯 Testing on specified objects: {target_obj_ids}")
    
    # 创建输出目录
    results_dir = os.path.join(args.save_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    cfg_path = "contactgen/configs/default.yaml"
    cfg = Config(default_cfg_path=cfg_path)
    device = torch.device('cuda')
    model = ContactGenModel(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    
    # ✅ 加载数据集
    if args.dataset == 'grab':
        dataset = TestSet()  # 原有的 GRAB 测试集
    elif args.dataset == 'grab_eval':
        dataset = GRABEvalDataset(
            obj_root=args.grab_obj_dir,
            n_samples=2048,
            num_samples_per_obj=args.num_samples_per_obj
        )
    else:  # oishape
        dataset = OIShape(split='test')

    action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}
    
    # 加载 Hand SDF 模型
    config_file = "contactgen/hand_sdf/config.yaml"
    config = OmegaConf.load(config_file)
    hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
    checkpoint = torch.load("contactgen/hand_sdf/hand_model.pt")
    hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
    hand_model.eval()
    hand_model.to(device)

    # 加载 MANO layer
    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=26, 
                          side='right', flat_hand_mean=False)
    mano_layer.to(device)

    # ✅ 加载指尖索引（MANO 右手）
    tip_indices = torch.tensor([745, 320, 444, 556, 673], dtype=torch.long, device=device)

    with open("assets/closed_mano_faces.pkl", 'rb') as f:
        hand_face = pickle.load(f)

    print(f"Processing {len(dataset)} samples from {args.dataset} dataset...")
    
    sample_count = 0
    for idx in tqdm(range(len(dataset))):
        input = dataset[idx]
        
        # ✅ 提取输入数据（兼容三种数据集格式）
        if args.dataset == 'grab':
            obj_name = input['obj_name']
            obj_verts_np = input['obj_verts']
            obj_vn_np = input['obj_vn']
            obj_id = obj_name
            intent_name = "use"
            obj_rotmat = np.eye(3, dtype=np.float32)
        elif args.dataset == 'grab_eval':
            obj_id = input['obj_id']
            obj_verts_np = input['obj_verts']
            obj_vn_np = input['obj_vn']
            sample_idx = input['sample_idx']
            intent_name = "use"
            obj_rotmat = np.eye(3, dtype=np.float32)
        else:  # oishape
            obj_verts_np = input['obj_verts']
            obj_vn_np = input['obj_vn']
            obj_id = dataset.grasp_list[idx]['obj_id']
            action_id = dataset.grasp_list[idx]['action_id']
            intent_name = action_id_to_intent.get(action_id, "unknown")
            obj_rotmat = np.eye(3, dtype=np.float32)
        
        # ✅ 如果指定了物体ID列表,则跳过不在列表中的物体
        if target_obj_ids and obj_id not in target_obj_ids:
            continue
        
        # 准备批次输入
        obj_verts = torch.from_numpy(obj_verts_np).unsqueeze(0).float().to(device).repeat(args.n_samples, 1, 1)
        obj_vn = torch.from_numpy(obj_vn_np).unsqueeze(0).float().to(device).repeat(args.n_samples, 1, 1)
        
        # 生成抓握
        with torch.no_grad():
            sample_results = model.sample(obj_verts, obj_vn)
        
        contacts_object, partition_object, uv_object = sample_results
        contacts_object = contacts_object.squeeze()
        partition_object = partition_object.argmax(dim=-1)
        
        # 优化手部姿态
        global_pose, mano_pose, mano_shape, mano_trans = optimize_pose(
            hand_model, mano_layer, obj_verts, contacts_object, partition_object, uv_object,
            w_contact=args.w_contact, w_pene=args.w_pene, w_uv=args.w_uv
        )
        
        # 获取手部顶点和关节
        hand_verts, hand_frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), 
            th_betas=mano_shape, 
            th_trans=mano_trans
        )
        
        # 构建 21 个关节：16 个骨骼关节 + 5 个指尖
        hand_joints_16 = hand_frames[:, :, :3, 3]
        hand_tips = hand_verts[:, tip_indices, :]
        hand_joints_21 = torch.cat([hand_joints_16, hand_tips], dim=1)
        
        hand_verts = hand_verts.detach().cpu().numpy()
        hand_joints = hand_joints_21.detach().cpu().numpy()
        
        # 校验形状
        assert hand_verts.shape[1:] == (778, 3), f"Unexpected verts shape: {hand_verts.shape}"
        assert hand_joints.shape[1:] == (21, 3), f"Unexpected joints shape: {hand_joints.shape}"
        
        # 保存每个生成的样本
        for i in range(len(hand_verts)):
            # ✅ 保存 pkl（Intent2Contact 格式）
            pkl_filename = f"{obj_id}_{intent_name}_{sample_count:06d}.pkl"
            pkl_path = os.path.join(results_dir, pkl_filename)
            export_to_pkl(
                hand_verts=hand_verts[i],
                hand_joints=hand_joints[i],
                obj_id=obj_id,
                obj_rotmat=obj_rotmat,
                intent_name=intent_name,
                save_path=pkl_path
            )
            
            # 可选：保存 mesh
            save_dir = os.path.join(args.save_root, obj_id)
            os.makedirs(save_dir, exist_ok=True)
            hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_face, process=False)
            hand_mesh.export(os.path.join(save_dir, f'grasp_{sample_count:06d}.obj'))
            
            sample_count += 1
    
    print(f"Generation complete! {sample_count} samples saved to {results_dir}")