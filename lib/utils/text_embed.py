import torch
import torch.nn as nn
import os

class SimpleIntentEmbedding(nn.Module):
    """简单的可学习意图嵌入"""
    def __init__(self, num_intents=4, intent_dim=64):
        super().__init__()
        self.num_intents = num_intents
        
        # 可学习的嵌入层
        self.embedding = nn.Embedding(num_intents, intent_dim)
        
        # 可选：添加LayerNorm保持与原设计一致
        self.layer_norm = nn.LayerNorm(intent_dim)
        
        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, intent_id):
        """
        intent_id: [B] tensor，取值范围 [0, num_intents-1]
        返回: [B, intent_dim] tensor
        """
        embeddings = self.embedding(intent_id)
        embeddings = self.layer_norm(embeddings)
        return embeddings
