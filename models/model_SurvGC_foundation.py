import torch
import numpy as np
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb

import math
import pandas as pd


def exists(val):
    return val is not None


class SurvGC_F(nn.Module):
    def __init__(
            self,
            clinic_embedding_dim=512,
            gene_embedding_dim=768,
            dropout=0.1,
            num_classes=4,
            clinic_projection_dim=256,
            gene_projection_dim=256,
    ):
        super(SurvGC_F, self).__init__()

        # ---> general props
        self.num_gene = 4
        self.num_clinic = 6
        self.dropout = dropout

        # ---> omics props
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_projection_dim = gene_projection_dim

        self.gene_projection_net = nn.Sequential(
            nn.Linear(self.gene_embedding_dim, self.gene_projection_dim),
        )

        # ---> clinic props
        self.clinic_embedding_dim = clinic_embedding_dim
        self.clinic_projection_dim = clinic_projection_dim

        self.clinic_projection_net = nn.Sequential(
            nn.Linear(self.clinic_embedding_dim, self.clinic_projection_dim),
        )

        # ---> cross attention props
        self.identity = nn.Identity()  # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.clinic_projection_dim,
            dim_head=self.clinic_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_gene
        )

        # ---> logits props
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.clinic_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.clinic_projection_dim // 2)

        # when both top and bottom blocks
        self.to_logits = nn.Sequential(
            nn.Linear(self.clinic_projection_dim, int(self.clinic_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.clinic_projection_dim / 4), self.num_classes)
        )

    def forward(self, **kwargs):

        x_omic = kwargs['x_omic']
        x_clinic = kwargs['x_clinic']
        mask = None
        return_attn = kwargs["return_attn"]

        # ---> project omic to smaller dimension
        omic_embed = self.gene_projection_net(x_omic)

        # ---> project clinic to smaller dimension (same as pathway dimension)
        clinic_embed = self.clinic_projection_net(x_clinic)

        tokens = torch.cat([clinic_embed, omic_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_clinic, cross_attn_clingene, cross_attn_geneclin = self.cross_attender(x=tokens,
                                                                                                      mask=mask if mask is not None else None,
                                                                                                      return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

            # ---> feedforward and layer norm
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        # ---> aggregate
        # modality specific mean
        clin_postSA_embed = mm_embed[:, :self.num_clinic, :]
        clin_postSA_embed = torch.mean(clin_postSA_embed, dim=1)

        gene_postSA_embed = mm_embed[:, self.num_clinic:, :]
        gene_postSA_embed = torch.mean(gene_postSA_embed, dim=1)

        tensor_gc = torch.cat([clin_postSA_embed, gene_postSA_embed], dim=1)

        # when both top and bottom block
        embedding = torch.cat([clin_postSA_embed, gene_postSA_embed], dim=1)  # ---> both branches
        embedding = self.identity(embedding)

        # embedding = torch.mean(mm_embed, dim=1)
        # ---> get logits
        logits = self.to_logits(embedding)


        if kwargs["return_attn"] == True:
            return logits, attn_clinic, cross_attn_clingene, cross_attn_geneclin
        else:
            return logits

    def captum(self, omic, wsi, clinic):

        # ---> unpack inputs
        mask = None
        return_attn = False

        # ---> get pathway embeddings
        omic_embed = self.gene_projection_net(omic)

        clinic_embed = self.clinic_projection_net(clinic)

        tokens = torch.cat([clinic_embed, omic_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_clinic, cross_attn_clingene, cross_attn_geneclin = self.cross_attender(x=tokens,
                                                                                                        mask=mask if mask is not None else None,
                                                                                                        return_attention=True)
        else:
            mm_embed = self.cross_attender1(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        # ---> feedforward and layer norm
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        # ---> aggregate
        # modality specific mean
        clin_postSA_embed = mm_embed[:, :self.num_clinic, :]
        clin_postSA_embed = torch.mean(clin_postSA_embed, dim=1)

        gene_postSA_embed = mm_embed[:, self.num_clinic:, :]
        gene_postSA_embed = torch.mean(gene_postSA_embed, dim=1)

        embedding = torch.cat([clin_postSA_embed, gene_postSA_embed], dim=1)
        embedding = self.identity(embedding)

        # ---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_clinic, cross_attn_clingene, cross_attn_geneclin
        else:
            return risk