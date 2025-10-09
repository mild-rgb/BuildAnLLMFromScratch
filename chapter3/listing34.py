import torch.nn as nn
import torch
from listing33 import *

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)] #_ is a throwaway variable and has no use. it just ensures that casual attention is created in the list
        )
    def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)