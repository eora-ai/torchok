from typing import List, Union, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from torchok.constructor import NECKS
from torchok.models.necks.base_neck import BaseNeck


@NECKS.register_class
class HRNetSegmentationNeck(BaseNeck):
    """HRNet neck for segmentation task. """
    def __init__(self, in_channels: Union[List[int], Tuple[int, ...]]):
        """Init HRNetSegmentationNeck.

        Args:
            in_channels: Input channels.
        """
        out_channels = sum(in_channels)
        super().__init__(in_channels, out_channels)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward method."""
        input_image, x0, x1, x2, x3 = features

        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        feats = torch.cat([x0, x1, x2, x3], 1)
        return [input_image, feats]
