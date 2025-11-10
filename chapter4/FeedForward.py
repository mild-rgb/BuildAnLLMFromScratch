import torch.nn as nn
from GELU import GELU
from config import GPT_CONFIG_124M

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    ) ##nn.Linear carries out a linear transform on the input matrix (w = Ax + B) where A and B are learnable parametres. in this case, it transforms x to 4x the input dimension, does gelu on it, then transforms back down

    def forward(self, x):
        return self.layers(x)