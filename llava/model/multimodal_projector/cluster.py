
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random
def init_centers(x,num_centers=256):
    i = random.sample(range(576), num_centers)
    return torch.index_select(x,1,torch.tensor(i,device=x.device))

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""
#
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         x=x.type(torch.float32)
#         ret = super().forward(x)
#         return ret.type(orig_type)

class Clustering(nn.Module):
    def __init__(self, channels, channels_out, num_centers, num_clusters):
        super().__init__()
        self.sim_alpha = nn.Parameter(torch.ones(1,1,1))
        self.sim_beta = nn.Parameter(torch.zeros(1,1,1))
        self.c = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.f = nn.Linear(channels, channels)
        self.ln_1 = nn.LayerNorm(channels)
        self.centers_proposal = nn.AdaptiveAvgPool1d(num_centers)
        self.clust_itr = num_clusters
        self.softmax = nn.Softmax(dim=-2)
        self.mlp = nn.Sequential(OrderedDict([
            ("0", nn.Linear(channels, channels_out)),
            ("1", QuickGELU()),
            ("2", nn.Linear(channels_out, channels_out))
        ])
        )
    def pairwise_cos_sim(self, x1, x2):
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)

        sim = torch.matmul(x1, x2.transpose(-2, -1))
        return sim
    
    def forward(self, x):
        ###
        # Input x: image embedding sequence [B,L,D]
        # Output x: cluster centers of image embedding [B,N,D]
        # ###
        #print(self.v.weight)
        #x = self.ln_1(x)
        value = self.v(x).permute(0,2,1)#[B,D,L]
        #print("v",torch.isnan(value).any())
        feature = self.f(x)#[B,L,D]
        #print("f",torch.isnan(feature).any())
        centers = self.centers_proposal(x.permute(0,2,1)) #[B,D,N]
        #print(centers)
        #print("c",torch.isnan(centers).any())
        centers_feature = self.centers_proposal(feature.permute(0,2,1)).permute(0,2,1)#[B,D,N]
        #print("cf",torch.isnan(centers_feature).any())
        centers = centers.permute(0,2,1)#[B,N,D]
        for _ in range(self.clust_itr):
            centers = self.c(centers)#[B,N,D]
            #(centers)
            #print(value)
            #print(centers@value)
            sim = self.softmax(F.normalize(centers,dim=-1)@F.normalize(value,dim=-1)) #[B,N,L]
            #print("sim1", torch.isnan(sim).any())
            centers = sim@feature#[B,N,D]
            #print("c2", torch.isnan(centers).any())
        #print("c3",torch.isnan(centers).any())
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * self.pairwise_cos_sim(centers, x)) 
        #similarity = self.softmax(self.pairwise_cos_sim(centers, x)) #[B,N,L]
        _, max_idx = similarity.max(dim=1, keepdim=True) # cloest center for each embedding [B,1,L]
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.)
        similarity = similarity * mask
        #print("sim:",torch.isnan(similarity).any())
        out = ((feature.unsqueeze(dim=1) * similarity.unsqueeze(dim=-1)).sum(dim=2) + centers_feature) / (
                    mask.sum(dim=-1, keepdim=True) + 1.0)
        out = x+(out.unsqueeze(dim=2) * similarity.unsqueeze(dim=-1)).sum(dim=1)
        #print("o:",torch.isnan(out).any())
        #print(out.shape)
        out = self.mlp(out)

        return out