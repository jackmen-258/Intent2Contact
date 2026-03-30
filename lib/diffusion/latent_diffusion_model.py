import os
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d.ops
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from manotorch.manolayer import ManoLayer, MANOOutput

from tqdm.auto import tqdm
from einops import reduce
from lib.networks.unet import LatentUnet
from lib.diffusion.utils import (
    default,
    extract,
    linear_beta_schedule,
    cosine_beta_schedule
)
from lib.networks.pointnet2 import Pointnet2
from lib.utils.text_embed import SimpleIntentEmbedding
from lib.datasets.utils import CENTER_IDX


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(-1).expand(*inds.shape, t.size(-1))
    return t.gather(dim, dummy.long())


def point2point_signed(x, y, x_normals=None, y_normals=None):
    """
    Signed distance between two point clouds.

    Args:
        x: (B, P1, 3), e.g. hand vertices
        y: (B, P2, 3), e.g. object point cloud
        x_normals: optional (B, P1, 3)
        y_normals: optional (B, P2, 3)

    Returns:
        y2x_signed: (B, P2), e.g. object-to-hand signed distance
        x2y_signed: (B, P1), e.g. hand-to-object distance
        yidx_near: (B, P2), nearest x index for each y point
    """
    _, xidx_near, x_near = pytorch3d.ops.knn_points(x, y, K=1, return_nn=True)
    _, yidx_near, y_near = pytorch3d.ops.knn_points(y, x, K=1, return_nn=True)

    x_near = x_near[:, :, 0, :]
    y_near = y_near[:, :, 0, :]
    xidx_near = xidx_near.squeeze(-1)
    yidx_near = yidx_near.squeeze(-1)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        x_nn = batched_index_select(x_normals, 1, yidx_near)
        y2x_signed = y2x.norm(dim=-1) * torch.sign((x_nn * y2x).sum(dim=-1))
    else:
        y2x_signed = y2x.norm(dim=-1)

    if y_normals is not None:
        y_nn = batched_index_select(y_normals, 1, xidx_near)
        x2y_signed = x2y.norm(dim=-1) * torch.sign((y_nn * x2y).sum(dim=-1))
    else:
        x2y_signed = x2y.norm(dim=-1)

    return y2x_signed, x2y_signed, yidx_near

def geodesic_loss(pred_aa, gt_aa):
    B = pred_aa.shape[0]
    K = pred_aa.shape[1] // 3
    
    pred_R = axis_angle_to_matrix(pred_aa.view(B, K, 3))
    gt_R = axis_angle_to_matrix(gt_aa.view(B, K, 3))

    R_diff = torch.matmul(pred_R.transpose(-2, -1), gt_R)
    trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
    cos = torch.clamp((trace - 1.0) / 2.0, -1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    
    return ang.mean()


def _rotation_matrix_to_rot6d(rot_mats: torch.Tensor) -> torch.Tensor:
    return rot_mats[..., :2].reshape(rot_mats.shape[0], -1)


def _rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    reshaped_input = rot_6d.view(-1, 3, 2)
    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def _build_unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edge_set = set()
    for tri in faces:
        tri = [int(v) for v in tri]
        edge_set.add(tuple(sorted((tri[0], tri[1]))))
        edge_set.add(tuple(sorted((tri[1], tri[2]))))
        edge_set.add(tuple(sorted((tri[2], tri[0]))))
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    return edges

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()

        self.fc2 = nn.Linear(dim_out, dim_out)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        h = self.fc1(self.act(self.norm(x)))
        h = self.fc2(h)
        return self.skip(x) + h


class GrabNetResBlock(nn.Module):
    def __init__(self, fin, fout, n_neurons=256):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.fc1 = nn.Linear(fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, fout)
        self.bn2 = nn.BatchNorm1d(fout)
        if fin != fout:
            self.fc3 = nn.Linear(fin, fout)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        xin = x if self.fin == self.fout else self.act(self.fc3(x))
        xout = self.fc1(x)
        xout = self.bn1(xout)
        xout = self.act(xout)
        xout = self.fc2(xout)
        xout = self.bn2(xout)
        xout = xin + xout
        if final_nl:
            return self.act(xout)
        return xout


class GrabNetStyleRefineNet(nn.Module):
    def __init__(
        self,
        center_idx: int,
        mano_assets_root: str = "assets/mano_v1_2",
        v_weights_path: str = "assets/rhand_weight.npy",
        closed_faces_path: str = "assets/closed_mano_faces.pkl",
        n_iters: int = 3,
        h_size: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.h_size = h_size
        self.in_size = 778 + 16 * 6 + 3

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=center_idx,
            mano_assets_root=mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
        )

        v_weights = np.load(v_weights_path).astype(np.float32)
        if os.path.isfile(closed_faces_path):
            with open(closed_faces_path, "rb") as f:
                faces = np.asarray(pickle.load(f), dtype=np.int64)
        else:
            faces = self.mano_layer.th_faces.detach().cpu().numpy().astype(np.int64)
        vpe = _build_unique_edges_from_faces(faces)

        self.register_buffer("v_weights", torch.from_numpy(v_weights))
        self.register_buffer("v_weights2", torch.pow(torch.from_numpy(v_weights), 1.0 / 2.5))
        self.register_buffer("vpe", torch.from_numpy(vpe).long())

        self.bn1 = nn.BatchNorm1d(778)
        self.rb1 = GrabNetResBlock(self.in_size, h_size)
        self.rb2 = GrabNetResBlock(self.in_size + h_size, h_size)
        self.rb3 = GrabNetResBlock(self.in_size + h_size, h_size)
        self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.out_p.weight)
        nn.init.zeros_(self.out_p.bias)
        nn.init.zeros_(self.out_t.weight)
        nn.init.zeros_(self.out_t.bias)

    def params_to_state(self, mano_params: torch.Tensor):
        pose = mano_params[:, :48]
        trans = mano_params[:, 48:51]
        shape = mano_params[:, 51:]

        rot_mats = axis_angle_to_matrix(pose.view(-1, 16, 3))
        pose_6d = _rotation_matrix_to_rot6d(rot_mats)
        return pose_6d, trans, shape

    def state_to_params(self, pose_6d: torch.Tensor, trans: torch.Tensor, shape: torch.Tensor):
        batch_size = pose_6d.shape[0]
        rot_mats = _rotation_6d_to_matrix(pose_6d).view(batch_size, 16, 3, 3)
        pose = matrix_to_axis_angle(rot_mats).reshape(batch_size, 48)
        return torch.cat([pose, trans, shape], dim=-1)

    def params_to_verts(self, mano_params: torch.Tensor):
        pose = mano_params[:, :48]
        trans = mano_params[:, 48:51]
        shape = mano_params[:, 51:]
        mano_output: MANOOutput = self.mano_layer(pose, shape)
        return mano_output.verts + trans.unsqueeze(1)

    def _hand_faces(self, batch_size: int, device: torch.device):
        return self.mano_layer.th_faces.to(device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1, -1)

    def compute_h2o_dist(self, hand_verts: torch.Tensor, obj_verts: torch.Tensor):
        hand_faces = self._hand_faces(hand_verts.shape[0], hand_verts.device)
        hand_mesh = Meshes(verts=hand_verts, faces=hand_faces)
        hand_normals = hand_mesh.verts_normals_padded()
        _, h2o, _ = point2point_signed(hand_verts, obj_verts, x_normals=hand_normals)
        return h2o.abs()

    def edges_for(self, verts: torch.Tensor):
        return verts[:, self.vpe[:, 0]] - verts[:, self.vpe[:, 1]]

    def forward(self, coarse_params: torch.Tensor, obj_verts: torch.Tensor):
        pose_6d, trans, shape = self.params_to_state(coarse_params)
        coarse_verts = self.params_to_verts(coarse_params)
        h2o_dist = self.compute_h2o_dist(coarse_verts, obj_verts)
        coarse_h2o_dist = h2o_dist

        init_pose = pose_6d
        init_trans = trans

        for i in range(self.n_iters):
            if i != 0:
                current_params = self.state_to_params(init_pose, init_trans, shape)
                current_verts = self.params_to_verts(current_params)
                h2o_dist = self.compute_h2o_dist(current_verts, obj_verts)

            h2o_dist_bn = self.bn1(h2o_dist)
            x0 = torch.cat([h2o_dist_bn, init_pose, init_trans], dim=1)
            x = self.rb1(x0)
            x = self.dropout(x)
            x = self.rb2(torch.cat([x, x0], dim=1))
            x = self.dropout(x)
            x = self.rb3(torch.cat([x, x0], dim=1))
            x = self.dropout(x)

            init_pose = init_pose + self.out_p(x)
            init_trans = init_trans + self.out_t(x)

        refined_params = self.state_to_params(init_pose, init_trans, shape)
        refined_verts = self.params_to_verts(refined_params)
        refined_h2o_dist = self.compute_h2o_dist(refined_verts, obj_verts)

        return {
            "refined_params": refined_params,
            "refined_verts": refined_verts,
            "refined_h2o_dist": refined_h2o_dist,
            "coarse_verts": coarse_verts,
            "coarse_h2o_dist": coarse_h2o_dist,
        }


class ManoEncoder(nn.Module):
    def __init__(self, in_dim, latent_D):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, 512)
        self.res_block1 = ResBlock(512, 512)
        self.res_block2 = ResBlock(512, 512)
        self.res_block3 = ResBlock(512, 256)
        self.res_block4 = ResBlock(256, 256)
        
        self.mu_head = nn.Linear(256, latent_D)
        self.logvar_head = nn.Linear(256, latent_D)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)
        
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class ManoDecoder(nn.Module):
    def __init__(self, latent_dim, params_dim):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, 256)
        self.res_block1 = ResBlock(256, 256)
        self.res_block2 = ResBlock(256, 512)
        self.res_block3 = ResBlock(512, 512)
        self.res_block4 = ResBlock(512, 512)

        self.pose_head  = nn.Linear(512, 48)   # axis-angle
        self.trans_head = nn.Linear(512, 3)    # translation
        self.shape_head = nn.Linear(512, 10)   # MANO shape

    def forward(self, z):
        h = self.input_proj(z)
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.res_block4(h)

        pose  = self.pose_head(h)
        trans = self.trans_head(h)
        shape = torch.clamp(self.shape_head(h), -3.0, 3.0)
        out = torch.cat([pose, trans, shape], dim=-1)

        return out

class ConcatFusion(nn.Module):
    def __init__(self, obj_dim, intent_dim, fusion_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obj_dim + intent_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, obj_feats, intent_embed):
        obj_global = obj_feats.mean(dim=1)
        fused = torch.cat([obj_global, intent_embed], dim=-1)
        return self.proj(fused)

class BiAttentionFusion(nn.Module):
    """
    双向交叉注意力融合：
    - Intent-to-Object: 意图关注物体的哪些区域
    - Object-to-Intent: 物体点如何被意图调制
    """
    def __init__(self, obj_dim, intent_dim, hidden_dim=128, num_heads=4, dropout=0.1, n_intent_tokens=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_intent_tokens = n_intent_tokens

        # 投影层
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.intent_proj = nn.Linear(intent_dim, hidden_dim)

        # 将单个intent embedding展开为多个token，提供多语义维度
        self.intent_expand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * n_intent_tokens),
            nn.GELU(),
        )
        
        # ===== 方向1: Intent-to-Object Attention =====
        # Query: Intent, Key/Value: Object
        # 语义：意图想要关注物体的哪些区域？
        self.intent_to_obj_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.intent_to_obj_norm = nn.LayerNorm(hidden_dim)
        
        # ===== 方向2: Object-to-Intent Attention =====
        # Query: Object, Key/Value: Intent
        # 语义：每个物体点应该如何被意图调制？
        self.obj_to_intent_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.obj_to_intent_norm = nn.LayerNorm(hidden_dim)
        
        # ===== 融合两个方向的输出 =====
        # intent_to_obj 输出: [B, hidden_dim] (意图视角的物体表示)
        # obj_to_intent 输出: [B, hidden_dim] (池化后的意图调制物体表示)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, obj_feats, intent_embed):
        """
        Args:
            obj_feats: [B, N, obj_dim] 物体点特征
            intent_embed: [B, intent_dim] 意图嵌入
        Returns:
            condition: [B, hidden_dim] 融合后的条件向量
        """
        B, N, _ = obj_feats.shape
        
        # 投影到统一维度
        obj_h = self.obj_proj(obj_feats)              # [B, N, hidden_dim]
        intent_h = self.intent_proj(intent_embed)     # [B, hidden_dim]
        intent_h_expanded = intent_h.unsqueeze(1)     # [B, 1, hidden_dim]
        
        # ===== 方向1: Intent-to-Object =====
        # Query: Intent, Key/Value: Object
        # 语义：意图关注物体的哪些功能区域，同时保留注意力权重用于指导O2I聚合
        i2o_out, i2o_attn = self.intent_to_obj_attn(
            query=intent_h_expanded,   # [B, 1, hidden_dim]
            key=obj_h,                 # [B, N, hidden_dim]
            value=obj_h,               # [B, N, hidden_dim]
            need_weights=True,
            average_attn_weights=True
        )  # i2o_out: [B, 1, hidden_dim], i2o_attn: [B, 1, N]

        i2o_out = self.intent_to_obj_norm(i2o_out.squeeze(1) + intent_h)  # [B, hidden_dim] + residual

        # ===== 方向2: Object-to-Intent =====
        # Query: Object, Key/Value: Intent (expanded to multiple tokens)
        # 语义：每个物体点从意图的多个语义维度中选择性聚合信息
        intent_tokens = self.intent_expand(intent_h)  # [B, hidden_dim * K]
        intent_tokens = intent_tokens.view(B, self.n_intent_tokens, self.hidden_dim)  # [B, K, hidden_dim]

        o2i_out, _ = self.obj_to_intent_attn(
            query=obj_h,               # [B, N, hidden_dim]
            key=intent_tokens,         # [B, K, hidden_dim]
            value=intent_tokens        # [B, K, hidden_dim]
        )  # [B, N, hidden_dim]

        o2i_out = self.obj_to_intent_norm(o2i_out + obj_h)  # [B, N, hidden_dim] + residual

        # 用I2O的注意力权重加权聚合O2I输出：聚焦于intent关注的功能区域
        # i2o_attn: [B, 1, N] → [B, N, 1]
        attn_weights = i2o_attn.squeeze(1).unsqueeze(-1)  # [B, N, 1]
        o2i_pooled = (o2i_out * attn_weights).sum(dim=1)  # [B, hidden_dim]
        
        # ===== 融合两个方向 =====
        fused = torch.cat([i2o_out, o2i_pooled], dim=-1)  # [B, hidden_dim * 2]
        condition = self.fusion_proj(fused)  # [B, hidden_dim]
        
        return condition
    
    def get_intent_to_obj_weights(self, obj_feats, intent_embed, device=None):
        """
        返回 per-point attention weights [B, N]（归一化到 [0,1]）
        """
        self.eval()
        with torch.no_grad():
            B, N, _ = obj_feats.shape
            intent_h = self.intent_proj(intent_embed).unsqueeze(1)  # [B,1,H]
            obj_h = self.obj_proj(obj_feats)                       # [B,N,H]

            _, attn_weights = self.intent_to_obj_attn(
                query=intent_h,
                key=obj_h,
                value=obj_h,
                need_weights=True,
                average_attn_weights=False
            )
            # attn_weights: [B, num_heads, tgt_len=1, src_len=N]

            # ✅ 聚合 heads -> [B, 1, N]，再 squeeze 掉 tgt_len 维 -> [B, N]
            attn = attn_weights.mean(dim=1).squeeze(1)  # [B, N]

            # ✅ 归一化到 [0,1]（沿 N 维做）
            attn_min = attn.min(dim=1, keepdim=True)[0]
            attn_max = attn.max(dim=1, keepdim=True)[0]
            attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)

        return attn  # [B, N]

class LatentHandDiffusion(nn.Module):
    def __init__(
        self,
        params_dim = 61,
        latent_dim = 64,
        obj_dim = 64,
        intent_dim = 64,
        fusion_dim = 128,
        time_emb_dim = 128, 
        timesteps = 1000,                                                           
        loss_type = 'l2',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        fusion_type: str = "bi_attn",
        disable_intent: bool = False,
        vae_geom_h2o_weight: float = 5.0,
        vae_geom_o2h_weight: float = 5.0,
        refine_n_iters: int = 3,
        refine_h_size: int = 512,
        refine_v_weights_path: str = "assets/rhand_weight.npy",
        refine_closed_faces_path: str = "assets/closed_mano_faces.pkl",
    ):
        super().__init__()

        self.params_dim = params_dim
        self.latent_dim = latent_dim

        self.obj_dim = obj_dim
        self.intent_dim = intent_dim

        self.encoder = ManoEncoder(params_dim, latent_dim)
        self.decoder = ManoDecoder(latent_dim, params_dim)

        self.obj_pointnet = Pointnet2(in_dim=6, hidden_dim=obj_dim, out_dim=obj_dim)
        self.intent_embed = SimpleIntentEmbedding(num_intents=4, intent_dim=intent_dim)

        # for ablation study(w/o intent in condition)
        self.disable_intent = disable_intent

        # Cross-attention fusion layer
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                obj_dim = obj_dim, 
                intent_dim = intent_dim, 
                fusion_dim = fusion_dim
            )
        elif fusion_type == "bi_attn":
            self.fusion = BiAttentionFusion(
                obj_dim = obj_dim, 
                intent_dim = intent_dim, 
                hidden_dim = fusion_dim,
                num_heads = 4
            )
        else:
            raise ValueError(f'unknown fusion type {fusion_type}')

        self.denoise_fn = LatentUnet(
            latent_dim = latent_dim,
            time_emb_dim = time_emb_dim, 
            cond_dim = fusion_dim,
            base_dim = 128,
            timesteps = timesteps
        )

        self.mano_layer = ManoLayer(
            center_idx=CENTER_IDX,
            mano_assets_root="assets/mano_v1_2"
        )
        self.refine_net = GrabNetStyleRefineNet(
            center_idx=CENTER_IDX,
            mano_assets_root="assets/mano_v1_2",
            v_weights_path=refine_v_weights_path,
            closed_faces_path=refine_closed_faces_path,
            n_iters=refine_n_iters,
            h_size=refine_h_size,
        )

        self.objective = objective
        self.loss_type = loss_type
        self.vae_geom_h2o_weight = vae_geom_h2o_weight
        self.vae_geom_o2h_weight = vae_geom_o2h_weight

        # Beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # snr reweighting
        snr = alphas_cumprod / (1 - alphas_cumprod)
        gamma = 5.0

        if self.objective == 'pred_x0':
            snr_weights = torch.clamp(snr, min=0.1, max=gamma)
        elif self.objective == 'pred_noise':
            snr_weights = torch.clamp(snr, max=gamma) / snr
        elif self.objective == 'pred_v':
            snr_weights = torch.ones_like(snr)

        snr_weights = snr_weights / snr_weights.mean()
        self.register_buffer('snr_weights', snr_weights)

        # latent normalization stats
        self.use_latent_norm = True

        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_std', torch.ones(latent_dim))
        self.latent_stats_computed = False

        # logSNR sampling
        self.use_logsnr_sampling = False

        if self.use_logsnr_sampling:
            logsnr = torch.log(snr)
            self.register_buffer('logsnr', logsnr)
            self.register_buffer('snr', snr)

    def encode_condition(self, obj_verts, obj_vn, intent_id):
        obj_input = torch.cat([obj_verts, obj_vn], dim=-1)

        obj_feats = self.obj_pointnet(obj_input)  # [B, N, obj_dim]

        if self.disable_intent:
            intent_embed = torch.zeros(obj_feats.shape[0], self.intent_dim, device=obj_feats.device, dtype=obj_feats.dtype)
        else:
            intent_embed = self.intent_embed(intent_id)  # [B, intent_dim]

        cond = self.fusion(obj_feats, intent_embed)
        return cond

    def compute_latent_stats(self, dataloader):
        print("[INFO] Computing latent variable statistics (full dataset)...")

        self.eval()
        all_latents = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent stats"):
                hand_pose = batch["hand_pose"].to(next(self.parameters()).device)
                hand_trans = batch["hand_tsl"].to(next(self.parameters()).device)
                hand_shape = batch["hand_shape"].to(next(self.parameters()).device)
                mano_params = torch.cat([hand_pose, hand_trans, hand_shape], dim=-1)

                mu, logvar = self.encode_latent(mano_params)
                all_latents.append(mu)

        all_latents = torch.cat(all_latents, dim=0)

        self.latent_mean.copy_(all_latents.mean(dim=0))
        self.latent_std.copy_(all_latents.std(dim=0) + 1e-8)

        self.latent_stats_computed = True
        
        print(f"[INFO] Latent stats computed from {all_latents.shape[0]} samples")
        print(f"[INFO] Mean range: [{self.latent_mean.min():.4f}, {self.latent_mean.max():.4f}]")
        print(f"[INFO] Std range: [{self.latent_std.min():.4f}, {self.latent_std.max():.4f}]")
        
        self.train()

    def normalize_latent(self, z):
        if not self.use_latent_norm or not self.latent_stats_computed:
            return z
        return (z - self.latent_mean) / self.latent_std
        
    def denormalize_latent(self, z_norm):
        if not self.use_latent_norm or not self.latent_stats_computed:
            return z_norm
        return z_norm * self.latent_std + self.latent_mean

    def encode_latent(self, mano_params):
        mu, logvar = self.encoder(mano_params)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_latent(self, z):
        params = self.decoder(z)
        return params

    @property
    def use_vae_geometry_regularization(self):
        return any(
            weight > 0.0
            for weight in (
                self.vae_geom_h2o_weight,
                self.vae_geom_o2h_weight,
            )
        )

    def sample_latent(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim, device=device)

    def parse_params(self, mano_params):
        pose = mano_params[:, :48]      # [B, 48]
        trans = mano_params[:, 48:51]   # [B, 3]
        shape = mano_params[:, 51:]     # [B, 10]

        return pose, trans, shape

    def params_to_verts(self, mano_params):
        pose, trans, shape = self.parse_params(mano_params)

        mano_output:MANOOutput = self.mano_layer(pose, shape)
        verts = mano_output.verts + trans.unsqueeze(1)  # [B, 778, 3]
        return verts

    def verts_to_normals(self, hand_verts):
        hand_faces = self.mano_layer.th_faces.to(device=hand_verts.device, dtype=torch.long)
        hand_faces = hand_faces.unsqueeze(0).expand(hand_verts.shape[0], -1, -1)
        hand_mesh = Meshes(verts=hand_verts, faces=hand_faces)
        return hand_mesh.verts_normals_padded()

    def compute_vae_geometry_loss(self, pred_verts, obj_verts, hand_verts_gt):
        zero = pred_verts.new_tensor(0.0)

        if (obj_verts is None) or (hand_verts_gt is None) or (not self.use_vae_geometry_regularization):
            return {
                'vae_geom_loss': zero,
                'vae_h2o_loss': zero,
                'vae_o2h_loss': zero,
            }

        pred_normals = self.verts_to_normals(pred_verts)

        with torch.no_grad():
            gt_normals = self.verts_to_normals(hand_verts_gt)
            o2h_signed_gt, h2o_gt, _ = point2point_signed(
                hand_verts_gt, obj_verts, x_normals=gt_normals
            )

        o2h_signed_pred, h2o_pred, _ = point2point_signed(
            pred_verts, obj_verts, x_normals=pred_normals
        )
        vae_h2o_loss = F.l1_loss(h2o_pred, h2o_gt)
        vae_o2h_loss = F.l1_loss(o2h_signed_pred, o2h_signed_gt)

        vae_geom_loss = (
            self.vae_geom_h2o_weight * vae_h2o_loss +
            self.vae_geom_o2h_weight * vae_o2h_loss
        )

        return {
            'vae_geom_loss': vae_geom_loss,
            'vae_h2o_loss': vae_h2o_loss,
            'vae_o2h_loss': vae_o2h_loss,
        }

    def compute_vae_loss(self, mano_params, obj_verts=None, hand_verts_gt=None):
        mu, logvar = self.encode_latent(mano_params)
        z = self.reparameterize(mu, logvar)
        pred_mano_params = self.decode_latent(z)

        with torch.no_grad():
            mu2 = (mu.pow(2)).mean().item()
            sigma2 = (logvar.exp()).mean().item()

        recon_loss_dict = self.compute_reconstruction_loss(pred_mano_params, mano_params)
        pose_loss = recon_loss_dict['pose_loss']
        trans_loss = recon_loss_dict['trans_loss']
        shape_loss = recon_loss_dict['shape_loss']
        recon_loss = recon_loss_dict['recon_loss']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        pred_verts = self.params_to_verts(pred_mano_params)  # [B, 778, 3]
        gt_verts = hand_verts_gt if hand_verts_gt is not None else self.params_to_verts(mano_params)
        cd_loss, _ = chamfer_distance(pred_verts, gt_verts)
        geom_loss_dict = self.compute_vae_geometry_loss(pred_verts, obj_verts, gt_verts)

        loss_dict = {
            'recon_loss': recon_loss,
            'pose_loss': pose_loss,
            'trans_loss': trans_loss,
            'shape_loss': shape_loss,
            'cd_loss': cd_loss,
            'kl_loss': kl_loss,
            'mu2': mu2,
            'sigma2': sigma2,
            **geom_loss_dict,
        }
        return loss_dict

    def compute_reconstruction_loss(self, pred_mano_params, gt_mano_params):
        pred_pose, pred_trans, pred_shape = self.parse_params(pred_mano_params)
        gt_pose, gt_trans, gt_shape = self.parse_params(gt_mano_params)

        pose_loss = geodesic_loss(pred_pose, gt_pose)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        shape_loss = F.mse_loss(pred_shape, gt_shape)
        recon_loss = pose_loss + 50.0 * trans_loss + 10.0 * shape_loss

        return {
            'recon_loss': recon_loss,
            'pose_loss': pose_loss,
            'trans_loss': trans_loss,
            'shape_loss': shape_loss,
        }

    def compute_refine_loss(self, coarse_mano_params, gt_mano_params, obj_verts, hand_verts_gt=None):
        gt_verts = hand_verts_gt if hand_verts_gt is not None else self.params_to_verts(gt_mano_params)

        with torch.no_grad():
            gt_h2o_dist = self.refine_net.compute_h2o_dist(gt_verts, obj_verts)

        refine_out = self.refine_net(coarse_mano_params, obj_verts)
        refined_params = refine_out["refined_params"]
        refined_verts = refine_out["refined_verts"]
        refined_h2o_dist = refine_out["refined_h2o_dist"]

        v_weights2 = self.refine_net.v_weights2.to(device=refined_verts.device, dtype=refined_verts.dtype)

        loss_dist_h = 35.0 * torch.mean(
            torch.einsum("ij,j->ij", torch.abs(refined_h2o_dist - gt_h2o_dist), v_weights2)
        )
        loss_mesh_rec = 35.0 * torch.mean(
            torch.einsum("ijk,j->ijk", torch.abs(refined_verts - gt_verts), v_weights2)
        )
        loss_edge = 30.0 * F.l1_loss(
            self.refine_net.edges_for(refined_verts),
            self.refine_net.edges_for(gt_verts),
        )

        loss_total = loss_dist_h + loss_mesh_rec + loss_edge

        return loss_total, {
            "refine_total": loss_total,
            "refine_dist_h": loss_dist_h,
            "refine_mesh_rec": loss_mesh_rec,
            "refine_edge": loss_edge,
        }

    @torch.no_grad()
    def refine_from_mano_params(self, coarse_mano_params, obj_verts):
        refine_out = self.refine_net(coarse_mano_params, obj_verts)
        return refine_out["refined_params"], refine_out

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_x0(self, x_t, t, model_output):
        if self.objective == 'pred_noise':
            return self.predict_start_from_noise(x_t, t=t, noise=model_output)
        if self.objective == 'pred_x0':
            return model_output
        if self.objective == 'pred_v':
            return self.predict_start_from_v(x_t, t=t, v=model_output)
        raise ValueError(f'unknown objective {self.objective}')

    @torch.no_grad()
    def predict(self, x, t, cond):
        return self.denoise_fn(x, t, cond)

    def p_mean_variance(self, x, t, cond):
        model_output = self.predict(x, t, cond)
        x_start = self.predict_x0(x, t, model_output)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start
        ) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def p_sample_loop(self, obj_verts, obj_vn, intent_id):
        device = self.betas.device
        b = obj_verts.shape[0]

        cond = self.encode_condition(obj_verts, obj_vn, intent_id)
        z = self.sample_latent(b, device=device)
            
        timesteps = list(range(self.num_timesteps))[::-1]
        for i in tqdm(timesteps, desc='sampling', disable=True):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t, cond)
            
        z = self.denormalize_latent(z)
        mano_params = self.decode_latent(z)

        return mano_params

    @torch.no_grad()
    def ddim_sample_loop(self, obj_verts, obj_vn, intent_id, sampling_steps=50, eta=0.0):
        device = self.betas.device
        b = obj_verts.shape[0]

        sampling_steps = int(sampling_steps)
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
        sampling_steps = min(sampling_steps, self.num_timesteps)

        cond = self.encode_condition(obj_verts, obj_vn, intent_id)
        z = self.sample_latent(b, device=device)

        times = torch.linspace(
            -1,
            self.num_timesteps - 1,
            steps=sampling_steps + 1,
            device=device,
        )
        times = torch.flip(times.to(torch.long), dims=(0,))
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))

        for time, time_next in tqdm(time_pairs, desc='sampling', disable=True):
            t = torch.full((b,), time, device=device, dtype=torch.long)
            model_output = self.predict(z, t, cond)
            x_start = self.predict_x0(z, t, model_output)
            pred_noise = self.predict_noise_from_start(z, t, x_start)

            if time_next < 0:
                z = x_start
                continue

            t_next = torch.full((b,), time_next, device=device, dtype=torch.long)
            alpha = extract(self.alphas_cumprod, t, z.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, z.shape)

            sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha)) * torch.sqrt(1.0 - alpha / alpha_next)
            c = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp(min=0.0))
            noise = torch.randn_like(z)

            z = torch.sqrt(alpha_next) * x_start + c * pred_noise + sigma * noise

        z = self.denormalize_latent(z)
        mano_params = self.decode_latent(z)

        return mano_params
    
    @torch.no_grad()
    def p_sample(self, x, t, cond):
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, cond)

        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def p_losses(self, mano_params, obj_verts, obj_vn, intent_id, t, noise=None):
        with torch.no_grad():
            mu, _ = self.encode_latent(mano_params)
        z_start = mu  # use deterministic encoding (mode of posterior) for diffusion target

        # normalize latent for diffusion training
        z_start = self.normalize_latent(z_start)

        noise = default(noise, lambda: torch.randn_like(z_start))
        z_noisy = self.q_sample(z_start, t, noise)

        condition = self.encode_condition(obj_verts, obj_vn, intent_id)
        
        model_output = self.denoise_fn(z_noisy, t, condition)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = z_start
        elif self.objective == 'pred_v':
            target = self.predict_v(z_start, t, noise)

        diff_loss = self.loss_fn(model_output, target, reduction='none')
        diff_loss = reduce(diff_loss, 'b ... -> b', 'mean')
        diff_loss = diff_loss * extract(self.snr_weights, t, diff_loss.shape)
        diff_loss = diff_loss.mean()

        return diff_loss

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, mano_params, obj_verts, obj_vn, intent_id, t=None, noise=None):
        device = mano_params.device
        b = mano_params.shape[0]
        
        if t is None:
            t = self.sample_timesteps_logsnr(b, device)

        loss = self.p_losses(
            mano_params,
            obj_verts,
            obj_vn,
            intent_id,
            t,
            noise=noise,
        )

        return loss
    
    @torch.no_grad()
    def sample(self, obj_verts, obj_vn, intent_id, sampler='ddpm', ddim_steps=50):
        sampler = sampler.lower()
        if sampler == 'ddpm':
            return self.p_sample_loop(obj_verts, obj_vn, intent_id)
        if sampler == 'ddim':
            return self.ddim_sample_loop(obj_verts, obj_vn, intent_id, sampling_steps=ddim_steps)
        raise ValueError(f"Unknown sampler: {sampler}")

    @torch.no_grad()
    def sample_and_refine(self, obj_verts, obj_vn, intent_id, sampler='ddpm', ddim_steps=50):
        coarse_mano_params = self.sample(obj_verts, obj_vn, intent_id, sampler=sampler, ddim_steps=ddim_steps)
        refined_mano_params, refine_out = self.refine_from_mano_params(coarse_mano_params, obj_verts)
        return refined_mano_params, {
            "coarse_mano_params": coarse_mano_params,
            **refine_out,
        }

    def sample_timesteps_logsnr(self, batch_size, device):
        if not self.use_logsnr_sampling:
            return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        min_logsnr, max_logsnr = self.logsnr.min(), self.logsnr.max()
        sampled_logsnr = torch.rand(batch_size, device=device) * \
                        (max_logsnr - min_logsnr) + min_logsnr
        
        logsnr_diffs = torch.abs(self.logsnr[None, :] - sampled_logsnr[:, None])
        t = logsnr_diffs.argmin(dim=1)
        
        return t
