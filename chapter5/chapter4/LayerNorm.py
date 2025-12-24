import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #added to prevent division by 0 during normalisation
        self.scale = nn.Parameter(torch.ones(emb_dim)) #variance
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #mean
    def forward(self, x):                              #scale and shift allow the model to reintroduce variability back in if needed. they are learnable parameters so can be altered if the gradient explodes
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift