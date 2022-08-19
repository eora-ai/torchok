from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchok.constructor import NECKS
from torchok.models.base import BaseModel
from torchok.models.modules.bricks.convbnact import ConvBnAct

ConvBnRelu = partial(ConvBnAct, act_layer=nn.ReLU)


@NECKS.register_class
class HRNetSegmentationNeck(BaseModel):
    """HRNet neck for segmentation task. """

    def __init__(self, in_channels: Union[List[int], Tuple[int, ...]]):
        """Init HRNetSegmentationNeck.

        Args:
            in_channels: Input channels.
        """
        out_channels = sum(in_channels)
        super().__init__(in_channels, out_channels)

        self.convbnact = ConvBnRelu(out_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward method."""
        input_image, x0, x1, x2, x3 = features

        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        feats = torch.cat([x0, x1, x2, x3], 1)
        feats = self.convbnact(feats)
        return [input_image, feats]
