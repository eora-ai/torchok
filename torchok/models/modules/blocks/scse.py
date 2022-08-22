"""TorchOK Concurrent Spatial and Channel Squeeze.

Adapted from:
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/modules.py
Licensed under MIT license [see LICENSE for details]
"""

from torch import nn as nn
from torch import Tensor


class SCSEModule(nn.Module):
    """Concurrent Spatial and Channel Squeeze."""

    def __init__(self, channels: int, reduction: int = 16):
        """Init SCSEModule

        Args:
            channels: Number input channels.
            reduction: Reduction coefficient.
        """
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: Input tensor.
        """
        output = x * self.cSE(x) + x * self.sSE(x)
        return output
