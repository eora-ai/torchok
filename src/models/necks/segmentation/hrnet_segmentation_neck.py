from typing import List

import torch
from torch import Tensor

import torch.nn.functional as F
from src.constructor import NECKS
from src.models.base import BaseModel


@NECKS.register_class
class HRNetSegmentationNeck(BaseModel):
    """HRNet neck for segmentation task. """
    def __init__(self, in_channels):
        """Init HRNetSegmentationNeck.

        Args:
            in_channels: Input channels.
        """
        out_channels = sum(in_channels)
        super().__init__(in_channels, out_channels)

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward method."""
        input_image, *features = features
        interpolated_feat = []
        for feature in features:
            interpolated = F.interpolate(feature, size=input_image.shape[2:], mode='bilinear', align_corners=True)
            interpolated_feat.append(interpolated)

        feats = torch.cat(interpolated_feat, 1)
        return feats
