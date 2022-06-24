"""TorchOK Unet.

Adapted from:
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders/unet
Licensed under MIT license [see LICENSE for details]
"""
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from src.constructor import NECKS
from src.models.base import BaseModel
from src.models.modules.blocks.se import SEModule
from src.models.modules.bricks.convbnact import ConvBnAct


@NECKS.register_class
class UnetNeck(BaseModel):
    """Unet is a fully convolution neural network for image semantic segmentation.
    Paper: https://arxiv.org/pdf/1505.04597.
    """

    def __init__(self,
                 encoder_channels: Tuple[int],
                 decoder_channels: Tuple[int] = (256, 128, 64, 32, 16),
                 n_blocks: int = 4,
                 use_batchnorm: bool = True,
                 use_attention: bool = False,
                 center: bool = True):
        """Init Unet

        Args:
            encoder_channels: Numbers of ``Conv2D`` layer filters.
            decoder_channels: Numbers of ``Conv2D`` layer filters in decoder blocks.
            n_blocks: number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            use_batchnorm: If ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``SEModule`` layers
                is used.
            use_attention: If ``True`` will use ``SEModule``.
            center: If ``True`` will use ''CenterBlock''.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        self.n_blocks = n_blocks

        encoder_channels = encoder_channels[1 - n_blocks:]
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm) if center else None

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, use_attention=use_attention)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = decoder_channels[-2]

        self.__init_weights()

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward method."""
        features = features[1 - self.n_blocks:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head, *skips = features

        x = self.center(head) if self.center is not None else head

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

    def __init_weights(self):
        """Weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.out_channels


class DecoderBlock(nn.Module):
    """Unet's decoder block."""
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            use_attention: bool = False,
            use_batchnorm: bool = True):
        """Init DecoderBlock.

        Args:
            in_channels: Input channels.
            skip_channels: Skip channels.
            out_channels: Output channels.
            use_attention: If ``True`` will use ``SEModule``.
            use_batchnorm: If ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``SEModule`` layers
                is used.
        """
        super().__init__()

        self.conv1 = ConvBnAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm)

        self.attention1 = SEModule(channels=in_channels + skip_channels) if use_attention else None

        self.conv2 = ConvBnAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm)

        self.attention2 = SEModule(channels=out_channels) if use_attention else None

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

            if self.attention1 is not None:
                x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.attention2 is not None:
            x = self.attention2(x)

        return x


class CenterBlock(nn.Sequential):
    """Optional building block for Unet like architecture."""
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: int = True):
        """Init CenterBlock.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            use_batchnorm: If True then `ConvBnAct` will use batch normalization.
        """
        conv1 = ConvBnAct(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm)
        conv2 = ConvBnAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)
