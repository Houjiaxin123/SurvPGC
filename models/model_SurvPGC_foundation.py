
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


class SurvPGC_F(nn.Module):
    def __init__(
        self,
        wsi_embedding_dim=1024,
        clinic_embedding_dim=512,
        gene_embedding_dim=768,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        clinic_projection_dim=256,
        gene_projection_dim=256,
        ):
        super(SurvPGC_F, self).__init__()

        #---> general props
        self.num_gene = 4
        self.num_clinic = 6
        self.dropout = dropout

        #---> omics props
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_projection_dim = gene_projection_dim

        self.gene_projection_net = nn.Sequential(
            nn.Linear(self.gene_embedding_dim, self.gene_projection_dim),
        )

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> clinic props
        self.clinic_embedding_dim = clinic_embedding_dim
        self.clinic_projection_dim = clinic_projection_dim

        self.clinic_projection_net = nn.Sequential(
            nn.Linear(self.clinic_embedding_dim, self.clinic_projection_dim),
        )

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender1 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_gene
        )
        self.cross_attender2 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_clinic
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim*2, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )

    
    def forward(self, **kwargs):

        wsi = kwargs['x_path']
        x_omic = kwargs['x_omic']
        x_clinic = kwargs['x_clinic']
        mask = None
        return_attn = kwargs["return_attn"]
        
        #---> project omic to smaller dimension
        omic_embed = self.gene_projection_net(x_omic)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        # ---> project clinic to smaller dimension (same as pathway dimension)
        clinic_embed = self.clinic_projection_net(x_clinic)

        tokens1 = torch.cat([omic_embed, wsi_embed], dim=1)
        tokens1 = self.identity(tokens1)
        tokens2 = torch.cat([clinic_embed, wsi_embed], dim=1)
        tokens2 = self.identity(tokens2)
        
        if return_attn:
            mm_embed1, attn_pathways, cross_attn_genepath, cross_attn_pathgene = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=True)
            mm_embed2, attn_clinic, cross_attn_clipath, cross_attn_pathcli = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed1 = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=False)
            mm_embed2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=False)


            #---> feedforward and layer norm
        mm_embed1 = self.feed_forward(mm_embed1)
        mm_embed1 = self.layer_norm(mm_embed1)
        mm_embed2 = self.feed_forward(mm_embed2)
        mm_embed2 = self.layer_norm(mm_embed2)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed1 = mm_embed1[:, :self.num_gene, :]
        paths_postSA_embed1 = torch.mean(paths_postSA_embed1, dim=1)

        wsi_postSA_embed1 = mm_embed1[:, self.num_gene:, :]
        wsi_postSA_embed1 = torch.mean(wsi_postSA_embed1, dim=1)

        paths_postSA_embed2 = mm_embed2[:, :self.num_clinic, :]
        paths_postSA_embed2 = torch.mean(paths_postSA_embed2, dim=1)

        wsi_postSA_embed2 = mm_embed2[:, self.num_clinic:, :]
        wsi_postSA_embed2 = torch.mean(wsi_postSA_embed2, dim=1)

        tensor_gp = torch.cat([paths_postSA_embed1, wsi_postSA_embed1], dim=1)
        tensor_cp = torch.cat([paths_postSA_embed2, wsi_postSA_embed2], dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed1, wsi_postSA_embed1, paths_postSA_embed2, wsi_postSA_embed2], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only
        embedding = self.identity(embedding)

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.to_logits(embedding)

        if kwargs["bag_loss"] == 'nll_diff_surv':
            if kwargs["return_attn"] == True:
                return tensor_cp, tensor_gp, logits, attn_pathways, cross_attn_genepath, cross_attn_pathgene, attn_clinic, cross_attn_clipath, cross_attn_pathcli
            else:
                return tensor_cp, tensor_gp, logits
        else:
            if kwargs["return_attn"] == True:
                return logits, attn_pathways, cross_attn_genepath, cross_attn_pathgene, attn_clinic, cross_attn_clipath, cross_attn_pathcli
            else:
                return logits

        
    def captum(self, omic, wsi, clinic):
        
        #---> unpack inputs
        mask = None
        return_attn = False

        #---> get pathway embeddings 
        omic_embed = self.gene_projection_net(omic)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        clinic_embed = self.clinic_projection_net(clinic)

        tokens1 = torch.cat([omic_embed, wsi_embed], dim=1)
        tokens1 = self.identity(tokens1)
        tokens2 = torch.cat([clinic_embed, wsi_embed], dim=1)
        tokens2 = self.identity(tokens2)

        if return_attn:
            mm_embed1, attn_pathways, cross_attn_pathhisto, cross_attn_histopath = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=True)
            mm_embed2, attn_clinic, cross_attn_clihisto, cross_attn_histocli = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed1 = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=False)
            mm_embed2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed1 = self.feed_forward(mm_embed1)
        mm_embed1 = self.layer_norm(mm_embed1)
        mm_embed2 = self.feed_forward(mm_embed2)
        mm_embed2 = self.layer_norm(mm_embed2)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed1 = mm_embed1[:, :self.num_gene, :]
        paths_postSA_embed1 = torch.mean(paths_postSA_embed1, dim=1)

        wsi_postSA_embed1 = mm_embed1[:, self.num_gene:, :]
        wsi_postSA_embed1 = torch.mean(wsi_postSA_embed1, dim=1)

        paths_postSA_embed2 = mm_embed2[:, :self.num_clinic, :]
        paths_postSA_embed2 = torch.mean(paths_postSA_embed2, dim=1)

        wsi_postSA_embed2 = mm_embed2[:, self.num_clinic:, :]
        wsi_postSA_embed2 = torch.mean(wsi_postSA_embed2, dim=1)


        embedding = torch.cat([paths_postSA_embed1, wsi_postSA_embed1, paths_postSA_embed2, wsi_postSA_embed2], dim=1)
        embedding = self.identity(embedding)

        #---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_pathways, cross_attn_pathhisto, cross_attn_histopath, attn_clinic, cross_attn_clihisto, cross_attn_histocli
        else:
            return risk