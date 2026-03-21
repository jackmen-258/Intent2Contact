import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import Pointnet
from .pointnet2 import Pointnet2

class IntentClassifier_V2(nn.Module):
    def __init__(self, num_intents=4, feats_dim=64, contact_embed_dim=64):
        super().__init__()

        self.obj_pointnet = Pointnet2(in_dim=6, hidden_dim=feats_dim, out_dim=feats_dim)

        self.contact_proj = nn.Sequential(
            nn.Linear(4, contact_embed_dim),
            nn.LayerNorm(contact_embed_dim),
            nn.GELU(),
            nn.Linear(contact_embed_dim, contact_embed_dim),
            nn.LayerNorm(contact_embed_dim),
            nn.GELU()
        )

        self.contact_encoder = Pointnet(in_dim=feats_dim + contact_embed_dim, hidden_dim=64, out_dim=64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_intents)
        )
        
    def forward(self, obj_verts, obj_vn, contact_map):
        # [B, N, feats_dim]
        obj_feature = self.obj_pointnet(torch.cat([obj_verts, obj_vn], dim=-1))

        contact_map = contact_map.float()  # [B, N, 1]
        contact_input = torch.cat([contact_map, obj_verts], dim=-1)  # [B, N, 4]
        contact_feat = self.contact_proj(contact_input)  # [B, N, contact_embed_dim]

        fusion = torch.cat([obj_feature, contact_feat], dim=-1)  # [B, N, feats_dim + contact_embed_dim]
        _, contact_feature = self.contact_encoder(fusion)        # [B, 64]

        intent_logits = self.classifier(contact_feature)
        return intent_logits