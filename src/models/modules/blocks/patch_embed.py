from typing import Tuple

import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 96,
            overlapped: bool = False):
        """Init PatchEmbed.
        
        Args:
            patch_size: Patch size.
            in_chans: Input channels.
            embed_dim: Embedding dimension.
            overlapped: Overlapping.
        """
        super().__init__()
        self.patch_size = patch_size

        if patch_size == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=kernel,
                stride=patch_size,
                padding=pad)
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x: Tensor, size: Tuple[int]) -> Tuple[Tensor, Tuple[int]]:
        """Forward method."""
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B,
                          H,
                          W,
                          C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size[1]))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize


