import os
import pickle
import glob
import numpy as np
import trimesh
from torch.utils.data import Dataset

from .utils import CENTER_IDX


class GRABTest(Dataset):
    """
    GRAB 测试集（仅物体，无手部 GT）
    用于跨域评估：在 GRAB 物体上生成抓握
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置对象，需包含:
                - DATA_ROOT: GRAB 物体根目录（如 'data/GRAB/object_models'）
                - NUM_POINTS: 采样点数（默认 2048）
                - INTENT_LIST: 可用的 intent 列表（可选，默认只用 'use'）
                - NUM_SAMPLES_PER_OBJ: 每个物体生成的样本数（默认 50）
        """
        self.data_root = cfg.get('DATA_ROOT', 'data/GRAB/object_models')
        self.num_points = cfg.get('NUM_POINTS', 2048)
        self.intent_list = cfg.get('INTENT_LIST', ['use'])
        self.num_samples_per_obj = cfg.get('NUM_SAMPLES_PER_OBJ', 50)  # ✅ 新增
        
        # 扫描所有物体
        obj_files = glob.glob(os.path.join(self.data_root, '*.ply'))
        obj_files += glob.glob(os.path.join(self.data_root, '*.obj'))
        
        if not obj_files:
            raise FileNotFoundError(f"No objects found in {self.data_root}")
        
        # 构建样本列表（每个物体 × 每个 intent × 重复次数）
        self.samples = []
        self.obj_warehouse = {}
        
        for obj_path in obj_files:
            obj_id = os.path.splitext(os.path.basename(obj_path))[0]
            
            # 加载并缓存物体
            obj_mesh = trimesh.load(obj_path, process=False, force='mesh', skip_materials=True)
            self.obj_warehouse[obj_id] = obj_mesh
            
            # ✅ 为每个 intent 和每次重复创建样本
            for intent_name in self.intent_list:
                for sample_idx in range(self.num_samples_per_obj):
                    self.samples.append({
                        'obj_id': obj_id,
                        'intent_name': intent_name,
                        'sample_idx': sample_idx,  # 样本编号（用于保存文件名）
                    })
        
        print(f"[GRABTest] Loaded {len(self.obj_warehouse)} objects, "
              f"{len(self.samples)} samples "
              f"({len(self.intent_list)} intents × {self.num_samples_per_obj} samples/obj)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj_id = sample['obj_id']
        intent_name = sample['intent_name']
        sample_idx = sample['sample_idx']  # ✅ 提取样本编号
        
        # 加载物体网格
        obj_mesh = self.obj_warehouse[obj_id]
        
        # ✅ 每次采样不同的点（因为 trimesh.sample 是随机的）
        # 如果需要确定性采样，可以设置种子: np.random.seed(idx)
        sample_result = trimesh.sample.sample_surface(obj_mesh, self.num_points)
        obj_verts = np.array(sample_result[0], dtype=np.float32)
        obj_vn = np.array(obj_mesh.face_normals[sample_result[1]], dtype=np.float32)
        
        # Intent ID
        intent_id = self.intent_list.index(intent_name)
        
        return {
            'obj_verts': obj_verts,
            'obj_vn': obj_vn,
            'obj_id': obj_id,
            'intent_name': intent_name,
            'intent_id': intent_id,
            'sample_idx': sample_idx,  # ✅ 返回样本编号
            'obj_rotmat': np.eye(3, dtype=np.float32),
        }