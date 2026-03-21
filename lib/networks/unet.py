import torch
from torch import nn

import math
from einops import rearrange

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        half_dim = dim // 2

        w = torch.randn(half_dim) * 0.02
        if is_random:
            self.register_buffer('weights', w)
        else:
            self.weights = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class TimeEmbedding_v2(nn.Module):
    def __init__(self, dim, random_fourier_features=True, timesteps=1000):
        super().__init__()
        time_dim = dim * 2
        self.timesteps = timesteps

        if random_fourier_features:
            self.emb = RandomOrLearnedSinusoidalPosEmb(dim, is_random=True)
        else:
            self.emb = RandomOrLearnedSinusoidalPosEmb(dim, is_random=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, dim)
        )

    def forward(self, t):
        t_feat = self.emb(t)
        res = self.mlp(t_feat)

        return res

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, feat_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, feat_dim * 2)
        )
        
    def forward(self, h, cond):
        h = self.norm(h)
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * (scale + 1) + shift


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, cond_dim, dropout=0.1):
        super().__init__()
        
        self.adaln1 = AdaptiveLayerNorm(in_dim, cond_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim * 2)
        )

        self.adaln2 = AdaptiveLayerNorm(out_dim, cond_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, time_emb, cond):
        h = self.mlp1(self.adaln1(x, cond))

        time_out = self.time_mlp(time_emb)
        scale, shift = time_out.chunk(2, dim=-1)
        h = h * (scale + 1) + shift   

        h = h + self.mlp2(self.adaln2(h, cond))
        
        return h + self.skip(x)
    
class LatentUnet(nn.Module):
    def __init__(self, latent_dim, time_emb_dim=128, cond_dim=64, base_dim=128, timesteps=1000):
        super().__init__()

        self.time_mlp = TimeEmbedding_v2(time_emb_dim, random_fourier_features=True, timesteps=timesteps)

        self.input_proj = nn.Linear(latent_dim, base_dim)

        # Downsampling
        self.down1 = ResBlock(base_dim, base_dim, time_emb_dim, cond_dim)
        self.down2 = ResBlock(base_dim, base_dim * 2, time_emb_dim, cond_dim)
        self.down3 = ResBlock(base_dim * 2, base_dim * 4, time_emb_dim, cond_dim)

        # Mid Block
        self.mid1 = ResBlock(base_dim * 4, base_dim * 4, time_emb_dim, cond_dim)
        self.mid2 = ResBlock(base_dim * 4, base_dim * 4, time_emb_dim, cond_dim)
        self.mid3 = ResBlock(base_dim * 4, base_dim * 4, time_emb_dim, cond_dim)

        # finetune after concat
        cat_dim_u3 = base_dim * 4 + base_dim * 4
        cat_dim_u2 = base_dim * 2 + base_dim * 2
        cat_dim_u1 = base_dim + base_dim

        self.up3_fuse = nn.Sequential(
            nn.LayerNorm(cat_dim_u3),
            nn.Linear(cat_dim_u3, cat_dim_u3),
            nn.GELU(),
            nn.Linear(cat_dim_u3, cat_dim_u3)
        )
        self.up2_fuse = nn.Sequential(
            nn.LayerNorm(cat_dim_u2),
            nn.Linear(cat_dim_u2, cat_dim_u2),
            nn.GELU(),
            nn.Linear(cat_dim_u2, cat_dim_u2)
        )
        self.up1_fuse = nn.Sequential(
            nn.LayerNorm(cat_dim_u1),
            nn.Linear(cat_dim_u1, cat_dim_u1),
            nn.GELU(),
            nn.Linear(cat_dim_u1, cat_dim_u1)
        )

        # Upsampling
        self.up3 = ResBlock(cat_dim_u3, base_dim * 2, time_emb_dim, cond_dim)
        self.up2 = ResBlock(cat_dim_u2, base_dim, time_emb_dim, cond_dim)
        self.up1 = ResBlock(cat_dim_u1, base_dim, time_emb_dim, cond_dim)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, latent_dim)
        )
       
    def forward(self, z, t, cond):
        """
        z: [B, latent_dim]
        t: [B] 
        cond: [B, cond_dim]
        return: [B, latent_dim]
        """

        time_emb = self.time_mlp(t)
        h = self.input_proj(z)

        # down path
        d1 = self.down1(h, time_emb, cond)
        d2 = self.down2(d1, time_emb, cond)
        d3 = self.down3(d2, time_emb, cond)

        # mid
        m = self.mid1(d3, time_emb, cond)
        m = self.mid2(m, time_emb, cond)
        m = self.mid3(m, time_emb, cond)

        # up（cat -> fuse -> ResBlock）
        u3_in = torch.cat([m, d3], dim=-1)
        u3_in = self.up3_fuse(u3_in)
        u3 = self.up3(u3_in, time_emb, cond)

        u2_in = torch.cat([u3, d2], dim=-1)
        u2_in = self.up2_fuse(u2_in)
        u2 = self.up2(u2_in, time_emb, cond)

        u1_in = torch.cat([u2, d1], dim=-1)
        u1_in = self.up1_fuse(u1_in)
        u1 = self.up1(u1_in, time_emb, cond)

        output = self.output_proj(u1)

        return output
