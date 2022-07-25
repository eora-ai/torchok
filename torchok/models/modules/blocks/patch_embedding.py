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
from typing import Optional

from torchok.models.modules.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """Image to Patch Embedding. Mostly used for VIT like architectures.
    
    Split image into patch by using Convolution layer with output_channels = embed_dim, kernel_size = patch_size
    and stride = patch_size.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        """Init PatchEmbed.
        
        Args:
            img_size: Input image size.
            patch_size: Patch token size.
            in_chans: Number of input image channels.
            embed_dim: Number of linear projection output channels.
            norm_layer: Normalization layer.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for patch embedding.
        
        Args:
            x: Input tensor.

        Returns:
            x: Patch embedding tensor with shape (B, Ph*Pw, C).

        Raises:
            ValueError: If input tensor shapes don't match with self.img_size. 
        """
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
