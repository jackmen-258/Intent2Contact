import pickle
import torch
from torch.nn import functional as F
import pytorch3d.ops
from pytorch3d.structures import Meshes

from .contact.diffcontact import calculate_contact_capsule
from .contact.opt_utils import compute_uv


class HandObject:
    # face_path：存储手部网格面数据的文件路径
    # hand_part_label_path：存储手部分标签数据的文件路径
    def __init__(self, device, face_path="assets/closed_mano_faces.pkl", hand_part_label_path="assets/hand_part_label.pkl"):
        with open(face_path, 'rb') as f:
            # hand_faces 增加一个维度以适配批量处理
            self.hand_faces = torch.Tensor(pickle.load(f)).unsqueeze(0).to(device)
        with open(hand_part_label_path, 'rb') as f:
            self.hand_part_label = torch.Tensor(pickle.load(f)).long().to(device)

    # hand_verts：手部顶点坐标
    # hand_frames：手部骨架变换矩阵
    # obj_verts：物体顶点坐标
    # obj_vn：物体顶点法线
    def forward(self, hand_verts, hand_frames, obj_verts, obj_vn):
        # 使用 pytorch3d 库中的 Meshes 类创建手部网格
        hand_mesh = Meshes(verts=hand_verts, faces=self.hand_faces)
        # 调用 calculate_contact_capsule 函数生成 contact map
        obj_contact_target, _ = calculate_contact_capsule(hand_mesh.verts_padded(),
                                                          hand_mesh.verts_normals_padded(),
                                                          obj_verts, obj_vn,
                                                          caps_top=0.0005, caps_bot=-0.0015,
                                                          caps_rad=0.003,
                                                          caps_on_hand=False)
        obj_cmap = obj_contact_target
        
        # 使用 knn_points 函数找到每个物体顶点最近的手部顶点索引，并根据这些索引从手部分标签中生成 part map
        _, nearest_idx, _ = pytorch3d.ops.knn_points(obj_verts, hand_verts, K=1, return_nn=True)
        nearest_idx = nearest_idx.squeeze(dim=-1)
        obj_partition = self.hand_part_label[nearest_idx]

        # 调用 compute_uv 函数计算UV坐标，生成direction map
        obj_uv = compute_uv(hand_frames, obj_verts, obj_partition)
        # 将 part map 转换为 one-hot 编码 
        obj_partition = F.one_hot(obj_partition, num_classes=16)

        # 将数据封装为字典并返回
        data_out = {
            "verts_object": obj_verts,
            "feat_object": obj_vn, 
            "contacts_object": obj_cmap, 
            "partition_object": obj_partition,
            "uv_object": obj_uv,
        }
        return data_out
