from typing import Tuple

import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    """Channel Attention."""
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False):
        """Init Channel Attention.
        
        Args:
            dim: Dimention.
            num_heads: Number of heads.
            qkv_bias: Query-Key-Value bias.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
