from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.modules.bricks.mlp import Mlp
from src.models.modules.bricks.droppath import DropPath
from src.models.modules.blocks.conv_pos_enc import ConvPosEnc
from src.models.modules.blocks.channel_attention import ChannelAttention


class ChannelBlock(nn.Module):
    """Channel Block of DaViT."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ffn: bool = True,
                 cpe_act: bool = False):
        """Init ChannelBlock.
        
        Args:
            dim: Dimension.
            num_heads: Number of heads.
            mlp_ratio: Multilayer perceptron ratio for hidden dim.
            qkv_bias: Query-Key-Value bias.
            drop_path: Drop path.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            ffn: If True, will use Mlp.
            cpe_act: If True, ConvPosEnc will use activation. 

        """
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act),
                                  ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x: Tensor, size: Tuple[int]):
        """Forward method."""
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size
