from typing import Tuple, Optional

import torch.nn as nn
from torch import Tensor


class ConvPosEnc(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_size: int = 3,
                 use_act: bool = False,
                 normtype: Optional[str] = None):
        """Init ConvPosEnc.
        
        Args:
            dim: Dimension.
            kernel_size: Kernel size.
            use_act: If True, will use GELU activation.
            normtype: Type of normalization.
        """
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              kernel_size,
                              1,
                              kernel_size // 2,
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if use_act else None

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        """Forward method."""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
    
        if self.activation is not None:
            x = x + self.activation(feat)
        return x
