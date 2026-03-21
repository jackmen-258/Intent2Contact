import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

from pytorch3d.loss import chamfer_distance
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.structures import Meshes

from tqdm.auto import tqdm
from einops import reduce
from lib.networks.unet import LatentUnet
from lib.diffusion.utils import (
    default,
    extract,
    linear_beta_schedule,
    cosine_beta_schedule
)

import numpy as np
from lib.networks.pointnet2 import Pointnet2
from lib.utils.text_embed import SimpleIntentEmbedding
from lib.datasets.utils import CENTER_IDX

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
        shape = self.shape_head(h)
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
    def __init__(self, obj_dim, intent_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.intent_proj = nn.Linear(intent_dim, hidden_dim)
        
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
        # 意图作为 Query，关注物体的哪些区域
        # 输出：意图"看到"的物体表示
        i2o_out, i2o_attn_weights = self.intent_to_obj_attn(
            query=intent_h_expanded,   # [B, 1, hidden_dim]
            key=obj_h,                 # [B, N, hidden_dim]
            value=obj_h                # [B, N, hidden_dim]
        )  # i2o_out: [B, 1, hidden_dim], i2o_attn_weights: [B, 1, N]
        
        i2o_out = self.intent_to_obj_norm(i2o_out.squeeze(1) + intent_h)  # [B, hidden_dim] + residual
        
        # ===== 方向2: Object-to-Intent =====
        # 物体点作为 Query，被意图调制
        # 输出：每个物体点经过意图调制后的表示
        o2i_out, o2i_attn_weights = self.obj_to_intent_attn(
            query=obj_h,               # [B, N, hidden_dim]
            key=intent_h_expanded,     # [B, 1, hidden_dim]
            value=intent_h_expanded    # [B, 1, hidden_dim]
        )  # o2i_out: [B, N, hidden_dim], o2i_attn_weights: [B, N, 1]
        
        o2i_out = self.obj_to_intent_norm(o2i_out + obj_h)  # [B, N, hidden_dim] + residual
        o2i_pooled = o2i_out.mean(dim=1)  # [B, hidden_dim] 池化
        
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
        disable_intent: bool = False
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

        self.objective = objective
        self.loss_type = loss_type

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

    def compute_latent_stats(self, dataloader, max_samples=20000):
        print("[INFO] Computing latent variable statistics...")
        
        self.eval()
        all_latents = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent stats"):
                if sample_count >= max_samples:
                    break
                    
                hand_pose = batch["hand_pose"].to(next(self.parameters()).device)
                hand_trans = batch["hand_tsl"].to(next(self.parameters()).device) 
                hand_shape = batch["hand_shape"].to(next(self.parameters()).device)
                mano_params = torch.cat([hand_pose, hand_trans, hand_shape], dim=-1)
                
                mu, logvar = self.encode_latent(mano_params)
                z = self.reparameterize(mu, logvar)
                all_latents.append(z)
                
                sample_count += z.shape[0]
                
                if sample_count >= max_samples:
                    break
        
        all_latents = torch.cat(all_latents, dim=0)[:max_samples]

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

    def compute_vae_loss(self, mano_params):
        mu, logvar = self.encode_latent(mano_params)
        z = self.reparameterize(mu, logvar)
        pred_mano_params = self.decode_latent(z)

        with torch.no_grad():
            mu2 = (mu.pow(2)).mean().item()
            sigma2 = (logvar.exp()).mean().item()

        pred_pose, pred_trans, pred_shape = self.parse_params(pred_mano_params)
        gt_pose, gt_trans, gt_shape = self.parse_params(mano_params)
        
        pose_loss = geodesic_loss(pred_pose, gt_pose)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        shape_loss = F.mse_loss(pred_shape, gt_shape)

        recon_loss = pose_loss + 50.0 * trans_loss + 10.0 * shape_loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        pred_verts = self.params_to_verts(pred_mano_params)  # [B, 778, 3]
        gt_verts = self.params_to_verts(mano_params)      # [B, 778, 3]
        cd_loss, _ = chamfer_distance(pred_verts, gt_verts)

        loss_dict = {
            'recon_loss': recon_loss,
            'pose_loss': pose_loss,
            'trans_loss': trans_loss,
            'shape_loss': shape_loss,
            'cd_loss': cd_loss,
            'kl_loss': kl_loss,
            'mu2': mu2,
            'sigma2': sigma2,
        }
        return loss_dict

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
        
    @torch.no_grad()
    def predict(self, x, t, cond):
        return self.denoise_fn(x, t, cond)

    def p_mean_variance(self, x, t, cond):
        model_output = self.predict(x, t, cond)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        elif self.objective == 'pred_v':
            x_start = self.predict_start_from_v(x, t=t, v=model_output)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
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

    @torch.no_grad()
    def p_sample_loop(self, obj_verts, obj_vn, intent_id):
        device = self.betas.device
        b = obj_verts.shape[0]

        cond = self.encode_condition(obj_verts, obj_vn, intent_id)
        z = self.sample_latent(b, device=device)
            
        timesteps = list(range(self.num_timesteps))[::-1]
        for i in tqdm(timesteps, desc='sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z = self.p_sample(z, t, cond)
            
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
        mu, logvar = self.encode_latent(mano_params)
        z_start = self.reparameterize(mu, logvar)

        # normalize latent for diffusion training
        z_start = self.normalize_latent(z_start)

        noise = default(noise, lambda: torch.randn_like(z_start))
        z_noisy = self.q_sample(z_start, t, noise)

        condition = self.encode_condition(obj_verts, obj_vn, intent_id)
        
        model_output = self.denoise_fn(z_noisy, t, condition)

        if self.objective == 'pred_noise':
            target = noise
            pred_z = self.predict_start_from_noise(z_noisy, t, model_output)
        elif self.objective == 'pred_x0':
            target = z_start
            pred_z = model_output
        elif self.objective == 'pred_v':
            v = self.predict_v(z_start, t, noise)
            target = v
            pred_z = self.predict_start_from_v(z_noisy, t, model_output)

        diff_loss = self.loss_fn(model_output, target, reduction='none')
        diff_loss = reduce(diff_loss, 'b ... -> b', 'mean')
        diff_loss = diff_loss * extract(self.snr_weights, t, diff_loss.shape)
        diff_loss = diff_loss.mean()

        # decode after denormalizing latent
        pred_z = self.denormalize_latent(pred_z)
        pred_params = self.decode_latent(pred_z)
        return diff_loss, pred_params

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

        loss, params = self.p_losses(mano_params, obj_verts, obj_vn, intent_id, t)

        return loss, params
    
    @torch.no_grad()
    def sample(self, obj_verts, obj_vn, intent_id):
        return self.p_sample_loop(obj_verts, obj_vn, intent_id)

    def sample_timesteps_logsnr(self, batch_size, device):
        if not self.use_logsnr_sampling:
            return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        min_logsnr, max_logsnr = self.logsnr.min(), self.logsnr.max()
        sampled_logsnr = torch.rand(batch_size, device=device) * \
                        (max_logsnr - min_logsnr) + min_logsnr
        
        logsnr_diffs = torch.abs(self.logsnr[None, :] - sampled_logsnr[:, None])
        t = logsnr_diffs.argmin(dim=1)
        
        return t