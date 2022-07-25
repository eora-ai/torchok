"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
import torch.nn as nn
from torch import Tensor

from torchok.constructor import HEADS
from torchok.models.base import BaseModel
from torchok.models.modules.bricks.convbnact import ConvBnAct


@HEADS.register_class
class HRNetSegmentationHead(BaseModel):
    """HRNet head for segmentation tasks."""
    def __init__(self, in_channels: int, num_classes: int):
        """Init HRNetSegmentationHead.

        Args:
            in_channels: Size of each input sample.
            num_classes: Number of classes.
        """
        super().__init__(in_channels, out_channels=num_classes)
        self.num_classes = num_classes
        self.convbnact = ConvBnAct(in_channels,
                                   in_channels,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1)
        self.final_conv_layer = nn.Conv2d(in_channels=in_channels,
                                          out_channels=num_classes,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method"""
        x = self.convbnact(x)
        x = self.final_conv_layer(x)
        return x
