import torch.nn as nn
from torch.nn import ReLU, ELU
import torch

"""

Implement a MLP to handle single-modal data 

"""

class MLPSingle(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4,
        projection_dim = 512,
        dropout = 0.1, 
        ):
        super(MLPSingle, self).__init__()
        
        # self
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//4), ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim*6//4, n_classes)
            )

    def forward(self, **kwargs):
        # self.cuda()

        #---> unpack
        # data_omics = kwargs["x_omic"].float().squeeze()
        data_omics = kwargs["x_clinic"].float().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics)#[B, n]
        #data = torch.mean(data, dim=0).unsqueeze(dim=0)
        data = data.view(1,-1)

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]
        return logits
    
    def captum(self, omics):

        # self.cuda()

        #---> unpack
        data_omics = omics.float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]

        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk



