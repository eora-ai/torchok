"""TorchOK MobileNetV3.

Adapted from https://github.com/xiaolai-sqlai/mobilenetv3
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import List, Union

import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModel
from src.models.modules.bricks.activations import HSigmoid
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.modules.blocks.se import SEModule
from src.models.modules.blocks.inverted_residual import InvertedResidualBlock
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.constructor import BACKBONES


default_cfgs = {
    'mobilenet_v3_large': dict(url=''),
    'mobilenet_v3_small': dict(url='')
}


class Block(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 in_channels: int,
                 expand_channels: int,
                 out_channels: int,
                 act_layer: nn.Module = nn.ReLU,
                 semodule: SEModule = None,
                 stride: int = 1):
        """Init Block.

        Args:
            kernel_size: Kernel size.
            in_channels: Input channels.
            expand_channels: Expand channels.
            out_channels: Output channels.
            act_layer: Activation layer.
            semodule: SEModule.
            stride: Stride.
        """
        super().__init__()
        self.stride = stride
        self.se = semodule
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []

        if in_channels != expand_channels:
            layers.append(ConvBnAct(in_channels, expand_channels, 1, 0, 1, bias=False, act_layer=act_layer))
        
        layers.append(ConvBnAct(expand_channels, expand_channels, kernel_size, kernel_size//2, stride,
                                    bias=False, groups=expand_channels, act_layer=act_layer))
                        
        layers.append(ConvBnAct(expand_channels, out_channels, 1, 0, 1, bias=False, act_layer=None))

        self.inverted_residual =  nn.Sequential(*layers)

        #shortcut
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(ConvBnAct(in_channels, out_channels, 1, 0, 1, bias=False, act_layer=None))

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        out = self.inverted_residual(x)

        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(BaseModel):
    """MobileNetV3 Large model."""
    def __init__(self, in_chans: int = 3):
        """Init MobileNetV3_Large.

        Args:
            in_chans: Input channels.
        """
        super().__init__()
        self.out_channels = 960
        self.convbnact_stem = ConvBnAct(in_chans, 16, kernel_size=3, padding=1, stride=2, act_layer=nn.Hardswish)

        self.bneck = nn.Sequential(
            InvertedResidualBlock(16, 16, 3, 1, expand_channels=16, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(16, 24, 3, 2, expand_channels=64, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 24, 3, 1, expand_channels=72, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 40, 5, 2, expand_channels=72, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=24, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=120, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=32, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=120, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=32, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(40, 80, 3, 2, expand_channels=240, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=200, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=184, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 80, 3, 1, expand_channels=184, act_layer=nn.Hardswish, use_se=False),
            InvertedResidualBlock(80, 112, 3, 1, expand_channels=480,act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=120, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(112, 112, 3, 1, expand_channels=672, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=168, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(112, 160, 5, 1, expand_channels=672, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=168, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(160, 160, 5, 2, expand_channels=960, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=240, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(160, 160, 5, 1, expand_channels=960, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=240, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
        )

        self.convbnact = ConvBnAct(160, self.out_channels, 1, 0, 1, act_layer=nn.Hardswish)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        out = self.convbnact_stem(x)
        out = self.bneck(out)
        out = self.convbnact(out)
        return out

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        return self.out_channels


class MobileNetV3_Small(BaseModel):
    """MobileNetV3 Small model"""
    def __init__(self, in_chans: int = 3):
        """Init MobileNetV3 Small.

        Args:
            in_chans: Input channels.
        """
        super(MobileNetV3_Small, self).__init__()
        self.out_channels = 576
        self.convbnact_stem = ConvBnAct(in_chans, 16, kernel_size=3, padding=1, stride=2, act_layer=nn.Hardswish)

        self.bneck = nn.Sequential(
            InvertedResidualBlock(16, 16, 3, 2, expand_channels=16, act_layer=nn.ReLU, use_se=True,
                                  se_kwargs=dict(reduction_channels=8, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(16, 24, 3, 2, expand_channels=72, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 24, 3, 1, expand_channels=88, act_layer=nn.ReLU, use_se=False),
            InvertedResidualBlock(24, 40, 5, 2, expand_channels=96, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=24, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),


            InvertedResidualBlock(40, 40, 5, 1, expand_channels=240, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=64, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(40, 40, 5, 1, expand_channels=240, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=64, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(40, 48, 5, 1, expand_channels=120, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=32, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(48, 48, 5, 1, expand_channels=144, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=40, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(48, 96, 5, 2, expand_channels=288, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=72, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(96, 96, 5, 1, expand_channels=576, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=144, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
            InvertedResidualBlock(96, 96, 5, 1, expand_channels=576, act_layer=nn.Hardswish, use_se=True,
                                  se_kwargs=dict(reduction_channels=144, use_pooling=True, gate=nn.Hardsigmoid, bias=True)),
        )

        self.convbnact = ConvBnAct(96, 576, 1, 0, 1, act_layer=nn.Hardswish)

    def forward(self, x: Tensor) -> Tensor:
        out = self.convbnact_stem(x)
        out = self.bneck(out)
        out = self.convbnact(out)
        return out

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        return self.out_channels


def create_mobilenetv3(variant: str, pretrained: bool = False, **kwargs):
    """Create MobileNetV3 base model."""
    if variant == 'mobilenet_v3_small':
        model_cls = MobileNetV3_Small
    elif variant == 'mobilenet_v3_large':
        model_cls = MobileNetV3_Large

    return build_model_with_cfg(model_cls, default_cfg=default_cfgs[variant],
                                pretrained=pretrained, **kwargs)


@BACKBONES.register_class
def mobilenet_v3_large(pretrained: bool = False, **kwargs):
    """It's constructing a mobilenet_v3_large model."""
    return create_mobilenetv3('mobilenet_v3_large', pretrained, **kwargs)


@BACKBONES.register_class
def mobilenet_v3_small(pretrained: bool = False, **kwargs):
    """It's constructing a mobilenet_v3_small model."""
    return create_mobilenetv3('mobilenet_v3_small', pretrained, **kwargs)
