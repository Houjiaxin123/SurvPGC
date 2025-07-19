
import torch
import numpy as np 
from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer

def exists(val):
    return val is not None


class MLPWSI(nn.Module):
    def __init__(
        self, 
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        device="cpu"
        ):
        super(MLPWSI, self).__init__()

        #---> init self 
        self.num_classes = num_classes

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
            ReLU(),
        )


        self.feed_forward = FeedForward(self.wsi_projection_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim)

        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
        )
        self.device = device
        
    def forward(self, **kwargs):

        wsi = kwargs['data_WSI']

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        #---> aggregate 
        embedding = torch.mean(wsi_embed, dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        return logits