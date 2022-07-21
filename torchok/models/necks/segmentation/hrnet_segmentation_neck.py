from typing import List, Union

import torch
from torch import Tensor

import torch.nn.functional as F
from torchok.constructor import NECKS
from torchok.models.base import BaseModel
from torchok.models.modules.bricks.convbnact import ConvBnAct


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

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward method."""
        input_image, *features = features
        interpolated_feat = []
        for feature in features:
            interpolated = F.interpolate(feature, size=input_image.shape[2:], mode='bilinear', align_corners=True)
            interpolated_feat.append(interpolated)

        feats = torch.cat(interpolated_feat, 1)
        return feats

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.in_channels
