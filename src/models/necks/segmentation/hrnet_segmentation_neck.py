from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
from src.constructor import NECKS
from src.models.base_model import BaseModel
from src.models.modules.bricks.convbnact import ConvBnAct


@NECKS.register_class
class HRNetSegmentationNeck(BaseModel):
    """HRNet neck for segmentation task. """
    def __init__(self, in_channels):
        """Init HRNetSegmentationNeck.

        Args:
            in_channels: Input channels.
        """
        super().__init__()
        self.in_channels = sum(in_channels)
        self.last_layer = ConvBnAct(self.in_channels,
                                    self.in_channels,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)

    def forward(self, x: List[Tensor]) -> Tensor:
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        return x

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.in_channels

