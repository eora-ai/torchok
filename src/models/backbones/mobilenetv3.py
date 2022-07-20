"""TorchOK MobileNetV3.

Adapted from https://github.com/xiaolai-sqlai/mobilenetv3
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModel
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.modules.blocks.inverted_residual import InvertedResidualBlock
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.constructor import BACKBONES


class MobileNetV3_Large(BaseModel):
    """MobileNetV3 Large model."""
    def __init__(self, in_channels: int = 3):
        """Init MobileNetV3_Large.

        Args:
            in_channels: Input channels.
        """
        out_channels = 960
        super().__init__(in_channels, out_channels)

        self.convbnact_stem = ConvBnAct(in_channels, 16, kernel_size=3, padding=1, stride=2, act_layer=nn.Hardswish)

        self.bneck = nn.Sequential(
            InvertedResidualBlock(16, 16, 3, 1, expand_channels=16, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(16, 24, 3, 2, expand_channels=64, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 24, 3, 1, expand_channels=72, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 40, 5, 2, expand_channels=72, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=24,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=120, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=32,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=120, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=32,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 80, 3, 2, expand_channels=240, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=200, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=184, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=184, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 112, 3, 1, expand_channels=480, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=120,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(112, 112, 3, 1, expand_channels=672, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=168,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(112, 160, 5, 1, expand_channels=672, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=168,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(160, 160, 5, 2, expand_channels=960, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=240,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(160, 160, 5, 1, expand_channels=960, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=240,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
        )

        self.convbnact = ConvBnAct(160, out_channels, 1, 0, 1, act_layer=nn.Hardswish)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        out = self.convbnact_stem(x)
        out = self.bneck(out)
        out = self.convbnact(out)
        return out


class MobileNetV3_Small(BaseModel):
    """MobileNetV3 Small model"""
    def __init__(self, in_chans: int = 3):
        """Init MobileNetV3 Small.

        Args:
            in_chans: Input channels.
        """
        out_channels = 576
        super(MobileNetV3_Small, self).__init__(in_chans, out_channels)

        self.convbnact_stem = ConvBnAct(in_chans, 16, kernel_size=3, padding=1, stride=2, act_layer=nn.Hardswish)

        self.bneck = nn.Sequential(
            InvertedResidualBlock(16, 16, 3, 2, expand_channels=16, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=8,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(16, 24, 3, 2, expand_channels=72, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 24, 3, 1, expand_channels=88, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 40, 5, 2, expand_channels=96, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=24,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=240, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=64,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=240, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=64,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(40, 48, 5, 1, expand_channels=120, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=32,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(48, 48, 5, 1, expand_channels=144, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=40,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(48, 96, 5, 2, expand_channels=288, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=72,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(96, 96, 5, 1, expand_channels=576, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=144,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
            InvertedResidualBlock(96, 96, 5, 1, expand_channels=576, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=144,
                                                 use_pooling=True,
                                                 gate=nn.Hardsigmoid,
                                                 bias=True)),
        )

        self.convbnact = ConvBnAct(96, out_channels, 1, 0, 1, act_layer=nn.Hardswish)

    def forward(self, x: Tensor) -> Tensor:
        out = self.convbnact_stem(x)
        out = self.bneck(out)
        out = self.convbnact(out)
        return out


def create_mobilenetv3(variant: str, pretrained: bool, pretrained_url, **model_args):
    """Create MobileNetV3 base model."""
    if variant == 'mobilenet_v3_small':
        model_cls = MobileNetV3_Small
    elif variant == 'mobilenet_v3_large':
        model_cls = MobileNetV3_Large

    return build_model_with_cfg(model_cls, pretrained, pretrained_url, **model_args)


@BACKBONES.register_class
def mobilenet_v3_large(pretrained: bool = False,
                       pretrained_url: str = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/mobilenetv3_large_torchok.pth', **model_args):
    """It's constructing a mobilenet_v3_large model."""
    return create_mobilenetv3('mobilenet_v3_large', pretrained, pretrained_url, **model_args)


@BACKBONES.register_class
def mobilenet_v3_small(pretrained: bool = False,
                       pretrained_url: str = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/mobilenetv3_small_torchok.pth', **model_args):
    """It's constructing a mobilenet_v3_small model."""
    return create_mobilenetv3('mobilenet_v3_small', pretrained, pretrained_url, **model_args)
