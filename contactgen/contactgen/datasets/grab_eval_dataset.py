import os
import numpy as np
import trimesh
from torch.utils import data


class GRABEvalDataset(data.Dataset):
    """
    GRAB 数据集评估类（用于 ContactGen）
    仅包含物体信息，用于生成抓握
    """
    def __init__(self, 
                 obj_root='data/GRAB/object_models',
                 n_samples=2048,
                 num_samples_per_obj=50):
        """
        Args:
            obj_root: GRAB 物体网格目录
            n_samples: 采样点数
            num_samples_per_obj: 每个物体生成的样本数
        """
        self.obj_root = obj_root
        self.n_samples = n_samples
        self.num_samples_per_obj = num_samples_per_obj
        
        # 扫描所有物体
        object_files = []
        for ext in ['.ply', '.obj']:
            object_files.extend([f for f in os.listdir(obj_root) if f.endswith(ext)])
        
        if not object_files:
            raise FileNotFoundError(f"No objects found in {obj_root}")
        
        self.object_files = sorted(object_files)
        
        # 构建样本列表（每个物体重复 num_samples_per_obj 次）
        self.samples = []
        for obj_file in self.object_files:
            obj_id = os.path.splitext(obj_file)[0]
            for sample_idx in range(num_samples_per_obj):
                self.samples.append({
                    'obj_file': obj_file,
                    'obj_id': obj_id,
                    'sample_idx': sample_idx,
                })
        
        print(f"[GRABEvalDataset] Loaded {len(self.object_files)} objects, "
              f"{len(self.samples)} total samples ({num_samples_per_obj} samples/obj)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample_info = self.samples[item]
        obj_file = sample_info['obj_file']
        obj_id = sample_info['obj_id']
        sample_idx = sample_info['sample_idx']
        
        # 加载物体网格
        obj_mesh_path = os.path.join(self.obj_root, obj_file)
        obj_mesh = trimesh.load(obj_mesh_path, process=False, force='mesh')
        
        # 采样表面点
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)

        return {
            "obj_id": obj_id,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn,
            "sample_idx": sample_idx,
        }