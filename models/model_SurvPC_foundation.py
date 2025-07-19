import torch
from torch import nn

from models.layers.cross_attention import FeedForward, MMAttentionLayer


def exists(val):
    return val is not None


class SurvPC_F(nn.Module):
    def __init__(
            self,
            wsi_embedding_dim=1024,
            clinic_embedding_dim=512,
            dropout=0.1,
            num_classes=4,
            wsi_projection_dim=256,
            clinic_projection_dim=256,
    ):
        super(SurvPC_F, self).__init__()

        # ---> general props
        self.num_clinic = 6
        self.dropout = dropout

        # ---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
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
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_clinic
        )

        # ---> logits props
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes)
        )

    def forward(self, **kwargs):

        wsi = kwargs['x_path']
        x_clinic = kwargs['x_clinic']
        mask = None
        return_attn = kwargs["return_attn"]

        # ---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        # ---> project clinic to smaller dimension (same as pathway dimension)
        clinic_embed = self.clinic_projection_net(x_clinic)

        tokens = torch.cat([clinic_embed, wsi_embed], dim=1)
        # tokens = tokens.unsqueeze(0)

        if return_attn:
            mm_embed, attn_clinic, cross_attn_clihisto, cross_attn_histocli = self.cross_attender(x=tokens,
                                                                                                        mask=mask if mask is not None else None,
                                                                                                        return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

            # ---> feedforward and layer norm
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        # ---> aggregate
        # modality specific mean
        paths_postSA_embed = mm_embed[:, :self.num_clinic, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_clinic:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # ---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only
        embedding = self.identity(embedding)

        # embedding = torch.mean(mm_embed, dim=1)
        # ---> get logits
        logits = self.to_logits(embedding)

        return logits


    def captum(self, wsi, clinic):

        # ---> unpack inputs
        mask = None
        return_attn = False

        # ---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        clinic_embed = self.clinic_projection_net(clinic)

        tokens = torch.cat([clinic_embed, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_clinic, cross_attn_clihisto, cross_attn_histocli = self.cross_attender(x=tokens,
                                                                                                        mask=mask if mask is not None else None,
                                                                                                        return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        # ---> feedforward and layer norm
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        # ---> aggregate
        # modality specific mean
        paths_postSA_embed = mm_embed[:, :self.num_clinic, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_clinic:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
        embedding = self.identity(embedding)

        # ---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_clinic, cross_attn_clihisto, cross_attn_histocli
        else:
            return risk