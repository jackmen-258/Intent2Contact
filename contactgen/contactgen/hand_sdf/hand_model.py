from typing import Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.linalg import transform_points

from .implicit_decoder import ImplicitDecoder


class ArtiHand(nn.Module):

    def __init__(self,
                 model_params: Dict[str, Any],
                 pose_size: int = 4,
                 shape_size: int = 10,
                 pose_deform_projection_bias: bool = False,
                 num_joints: int = 16):
        super(ArtiHand, self).__init__()
        
        # num_joints 对应论文中 piecewise hand model 的 B parts
        self.num_joints = num_joints
        self.pose_size = pose_size

        self.latent_size = shape_size
        self.shape_size_list = self.latent_size
        self.shape_size_list = [self.shape_size_list] * self.num_joints
      

        # 创建一个模型列表，每个关节都有一个 ImplicitDecoder 模型实例
        # 使用 model_params 来初始化每个解码器
        models = []
        for i in range(self.num_joints):
            model_params['latent_size'] = self.pose_size + self.shape_size_list[i]
            models += [ImplicitDecoder(**model_params)]

        self.model = nn.ModuleList(models)
        # 创建一个线性层，用于从手部关节的位置信投影到姿态空间（轴角表示）
        # pose_deform_projection_bias：一个布尔值，决定在投影层中是否使用偏置项
        self.projection = nn.Linear(in_features=self.num_joints * 3,
                                    out_features=self.num_joints * self.pose_size,
                                    bias=pose_deform_projection_bias)

    # queries：输入的查询数据
    # soft_blend：用于控制结果中的软混合权重
    def forward(self, queries, soft_blend=100.0):
        # dim 表示查询点的坐标？
        batch_size, num_joints, num_queries, dim = queries.shape

        # 使用 torch.split 将 queries 沿 dim=1 分割成 num_joints 个部分
        res_list = [self.model[i](q.reshape(batch_size * num_queries, -1), 1.0)
                for i, q in enumerate(torch.split(queries, 1, dim=1))]
        # dim=-1 为预测的 sdf 值？
        res = torch.cat(res_list, dim=-1)
        del res_list
        res_parts = res.reshape(batch_size, num_queries, -1)

        if soft_blend is not None:
            # 软混合：对查询点到各个关节的sdf进行加权混合
            weights = F.softmin(soft_blend * res_parts, dim=-1)
            res = (res_parts * weights).sum(-1, keepdim=True)
        else:
            # 硬选择：直接取查询点到各个关节的sdf中的最小值
            res = res_parts.min(dim=-1)[0]

        # res：处理后的查询结果，形状为 (batch_size, num_queries, 1)
        return res, res_parts

    def add_shape_feature(self,
                          queries: torch.FloatTensor,
                          shape_indices: Optional[torch.LongTensor] = None,
                          latent_shape_code: Optional[torch.FloatTensor] = None):
        batch_size, num_joints, num_queries, dim = queries.shape
        assert (shape_indices is None) + (latent_shape_code is None) == 1
        if shape_indices is not None:
            latents = self.latents(shape_indices)
        else:
            latents = latent_shape_code

        latents = latents.unsqueeze(1).expand(-1, num_queries, -1)
        latents = latents.unsqueeze(1).expand(-1, num_joints, -1, -1)
        queries = torch.cat([latents, queries], dim=-1)
        return queries

    def add_pose_feature(self, queries, root, trans):
        batch_size = trans.shape[0]
        num_queries = queries.shape[2]
        num_joints = trans.shape[1]

        root = root.unsqueeze(1).unsqueeze(1).expand(batch_size, num_joints, 1, 3)
        root = root.reshape(-1, 1, 3)
        root = transform_points(trans.reshape(-1, 4, 4), root)
        root = root.reshape(batch_size, num_joints, 1, 3)
        root = root.reshape(batch_size, -1)
        reduced_root = self.projection(root)
        reduced_root = reduced_root.reshape(batch_size, num_joints, 1,
                                                    -1)
        trans_feature = reduced_root.expand(batch_size, num_joints, num_queries, -1)
        queries = torch.cat([trans_feature, queries], dim=-1)
        return queries

    def transform_queries(self, queries, trans):
        batch_size = trans.shape[0]
        num_queries = queries.shape[-2]
        num_joints = trans.shape[1]

        if queries.dim() == 3:
            queries = queries.unsqueeze(1).expand(batch_size, num_joints,
                                                  num_queries, 3)
        queries = queries.reshape(-1, num_queries, 3)
        trans = trans.reshape(-1, 4, 4)
        queries = transform_points(trans, queries)
        queries = queries.reshape(batch_size, num_joints, num_queries, 3)

        return queries