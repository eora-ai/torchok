"""TorchOK Swin Transformer V2

Adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
Copyright (c) 2022 Microsoft
Licensed under The MIT License [see LICENSE for details]
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
from torch import nn as nn
from typing import Tuple


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Divides the input tensor into 4 parts then concatenate this parts by channels
    to get tensor with shape (B, H/2*W/2, 4*C), then use Linear layer to reduce channels number by 2.
    Output tensor shape would be (B, H/2*W/2, 2*C). (It like Convolution with stride=2 and output_channels=2*C, only
    for VIT architectures).
    """
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        """Init PatchMerging.
        
        Args:
            input_resolution: Resolution of input feature.
            dim: Number of input channels.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Froward method for patch merging.

        Args:
            x: Input tensor with shape B, H*W, C.

        Returns:
            x: Output tensor with shape (B, H/2*W/2, 2*C).

        Raises:
            ValueError: If self.input_resolution don't match with input tensor.
            ValueError: If one of H or W in self.input_resolution is odd number.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        if  L != H * W:
            raise ValueError('PatchMerging forward method, input feature has wrong size.')
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"PatchMerging forward method, x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x
