from typing import List, Union, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from torchok.constructor import NECKS
from torchok.models.necks.base_neck import BaseNeck


@NECKS.register_class
class HRNetSegmentationNeck(BaseNeck):
    """HRNet neck for segmentation task. """
    def __init__(self, in_channels: Union[List[int], Tuple[int, ...]], start_block: int = 2):
        """Init HRNetSegmentationNeck.

        Args:
            in_channels: Input channels.
        """
        out_channels = sum(in_channels[start_block:])
        super().__init__(start_block, in_channels, out_channels)

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward method."""
        input_image = features[0]
        used_features = features[self._start_block:]
        interpolated_feat = []
        for used_feature in used_features:
            # In original repo it has align_corners=True parameter in F.interpolate(), but with align_corners=True
            # parameter it doesn't convert to ONNX
            interpolated = F.interpolate(used_feature, size=input_image.shape[2:], mode='bilinear')
            interpolated_feat.append(interpolated)

        feats = torch.cat(interpolated_feat, 1)
        return feats
