"""TorchOK Unet.

Adapted from:
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders/unet
Licensed under MIT license [see LICENSE for details]
"""
from typing import Tuple, Optional

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from src.constructor import HEADS
from src.models.heads.base import AbstractHead


@HEADS.register_class
class UnetHead(AbstractHead):
    """Head for Unet architecture."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 do_interpolate: bool = True,
                 img_size: Optional[Tuple[int]] = None):
        """Init UnetHead.

        Args:
            in_features: Size of each input sample.
            num_classes: A number of classes for output (output shape - ``(batch, classes, h, w)``).
            do_interpolate: If ``True`` will interpolate features after forward pass.
            img_size: Requiered output size of image.

        Raises:
            ValueError: If "do_interpolate" is True, when img_size is None.
        """
        super().__init__(in_channels, num_classes)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.do_interpolate = do_interpolate
        self.img_size = img_size

        if self.do_interpolate and self.img_size is None:
            raise ValueError('If "do_interpolate" is True, then img_size must be not None!')

        self.head_features = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        segm_logits = self.head_features(x)
        if self.do_interpolate:
            segm_logits = F.interpolate(segm_logits, size=self.img_size,
                                        mode='bilinear', align_corners=False)
        if self.num_classes == 1:
            segm_logits = segm_logits[:, 0]
        
        return segm_logits
