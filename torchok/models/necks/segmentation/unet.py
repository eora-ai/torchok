"""TorchOK Unet.

Adapted from:
https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders/unet
Licensed under MIT license [see LICENSE for details]
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchok.constructor import NECKS
from torchok.models.base import BaseModel
from torchok.models.modules.blocks.scse import SCSEModule
from torchok.models.modules.bricks.convbnact import ConvBnAct


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_attention: bool = False, use_batchnorm: bool = True):
        """Init DecoderBlock.

        Args:
            in_channels: Input channels.
            skip_channels: Skip channels.
            out_channels: Output channels.
            use_attention: If ``True`` will use ``SEModule``.
            use_batchnorm: If ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``SEModule`` layers is used.
        """
        super().__init__()
        in_channels = in_channels + skip_channels

        self.attention1 = SCSEModule(channels=in_channels) if use_attention else nn.Identity()
        self.conv1 = ConvBnAct(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = ConvBnAct(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = SCSEModule(channels=out_channels) if use_attention else nn.Identity()

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """Forward method.

        Args:
            x: Tensor for upsample.
            skip: Skip-connection features.
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            if skip.size(2) != x.size(2):
                skip = F.interpolate(skip, size=(x.size(2), x.size(3)), mode="nearest")

            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    """Optional building block for Unet like architecture."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        """Init CenterBlock.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            use_batchnorm: If True then `ConvBnAct` will use batch normalization.
        """
        conv1 = ConvBnAct(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = ConvBnAct(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


@NECKS.register_class
class UnetNeck(BaseModel):
    """Unet is a fully convolutional neural network for image semantic segmentation.
    Paper: https://arxiv.org/pdf/1505.04597.
    """

    def __init__(self, in_channels: List[int], decoder_channels: List[int] = (512, 256, 128, 64, 64),
                 use_batchnorm: bool = True, use_attention: bool = False, center: bool = True):
        """Init Unet. The number of stages used in decoder inferred from the ``decoder_channels``,
        larger depth - more features are generated. e.g. for depth=3 encoder will generate list of features with
        following spatial shapes [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor
        will have spatial resolution (H/(2^depth), W/(2^depth)]

        Args:
            in_channels: List of Numbers of ``Conv2D`` layer filters from backbone.
            decoder_channels: List of numbers of ``Conv2D`` layer filters in decoder blocks.
            use_batchnorm: If ``True``, ``BatchNormalisation`` applied between every ``Conv2D`` and activation layers.
            use_attention: If ``True`` will use ``SCSEModule``.
            center: If ``True`` will use ''CenterBlock''.

        Raises:
            ValueError: If the number of blocks is not equal to the length of the `decoder_channels`
                or `encoder_channels - 1`.
        """
        super().__init__(in_channels=in_channels, out_channels=decoder_channels[-1])

        self.n_blocks = len(decoder_channels)
        encoder_channels = in_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = CenterBlock(head_channels, head_channels, use_batchnorm) if center else nn.Identity()

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, use_attention=use_attention)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.init_weights()

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward method."""
        head, *skips, input_image = features[::-1]  # reverse channels to start from head of encoder

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return [input_image, x]
