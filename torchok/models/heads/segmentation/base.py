from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchok.constructor import HEADS
from torchok.models.base import BaseModel


@HEADS.register_class
class SegmentationHead(BaseModel):
    """Base Segmentation head."""

    def __init__(self, in_channels: int, num_classes: int, do_interpolate: bool = True):
        """Init SegmentationHead.

        Args:
            in_channels: Size of each input sample.
            num_classes: A number of classes for output (output shape - ``(batch, classes, h, w)``).
            do_interpolate: If ``True`` will interpolate features after forward pass.
        """
        super().__init__(in_channels, num_classes)

        self.num_classes = num_classes
        self.do_interpolate = do_interpolate

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.init_weights()

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward method."""
        input_image, features = x

        segm_logits = self.classifier(features)
        if self.do_interpolate:
            segm_logits = F.interpolate(segm_logits, size=input_image.shape[2:], mode='bilinear')
        if self.num_classes == 1:
            segm_logits = segm_logits[:, 0]

        return segm_logits
