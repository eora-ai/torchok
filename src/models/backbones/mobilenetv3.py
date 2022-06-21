"""TorchOK MobileNetV3.

Adapted from https://github.com/xiaolai-sqlai/mobilenetv3
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import List, Union

import torch.nn as nn
from torch.nn import init
from torch import Tensor

from src.models.base import BaseModel
from src.models.modules.bricks.activations import HSigmoid, HSwish
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.modules.blocks.se import SEModule
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from src.constructor import BACKBONES


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        **kwargs
    }


default_cfgs = {
    'mobilenet_v3_large': _cfg(url=''),
    'mobilenet_v3_small': _cfg(url='')
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

        self.convbnact1 = ConvBnAct(in_channels, expand_channels, 1, 0, 1, bias=False, act_layer=act_layer)
        self.convbnact2 = ConvBnAct(expand_channels, expand_channels, kernel_size, kernel_size//2, stride,
                                    bias=False, groups=expand_channels, act_layer=act_layer)
        self.convbnact3 = ConvBnAct(expand_channels, out_channels, 1, 0, 1, bias=False, act_layer=None)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(ConvBnAct(in_channels, out_channels, 1, 0, 1, bias=False, act_layer=None))

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        out = self.convbnact1(x)
        out = self.convbnact2(out)
        out = self.convbnact3(out)

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
        self.convbnact_stem = ConvBnAct(in_chans, 16, kernel_size=3, padding=1, stride=2, act_layer=HSwish)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, None, 1),
            Block(3, 16, 64, 24, nn.ReLU, None, 2),
            Block(3, 24, 72, 24, nn.ReLU, None, 1),
            Block(5, 24, 72, 40, nn.ReLU,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 2),
            Block(5, 40, 120, 40, nn.ReLU,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 40, 120, 40, nn.ReLU,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(3, 40, 240, 80, HSwish, None, 2),
            Block(3, 80, 200, 80, HSwish, None, 1),
            Block(3, 80, 184, 80, HSwish, None, 1),
            Block(3, 80, 184, 80, HSwish, None, 1),
            Block(3, 80, 480, 112, HSwish,
                  SEModule(112, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(3, 112, 672, 112, HSwish,
                  SEModule(112, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 112, 672, 160, HSwish,
                  SEModule(160, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 160, 672, 160, HSwish,
                  SEModule(160, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 2),
            Block(5, 160, 960, 160, HSwish,
                  SEModule(160, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
        )

        self.convbnact = ConvBnAct(160, self.out_channels, 1, 0, 1, act_layer=HSwish)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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
        self.convbnact_stem = ConvBnAct(in_chans, 16, kernel_size=3, padding=1, stride=2, act_layer=HSwish)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU,
                  SEModule(16, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 2),
            Block(3, 16, 72, 24, nn.ReLU, None, 2),
            Block(3, 24, 88, 24, nn.ReLU, None, 1),
            Block(5, 24, 96, 40, HSwish,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 2),
            Block(5, 40, 240, 40, HSwish,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 40, 240, 40, HSwish,
                  SEModule(40, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 40, 120, 48, HSwish,
                  SEModule(48, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 48, 144, 48, HSwish,
                  SEModule(48, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 48, 288, 96, HSwish,
                  SEModule(96, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 2),
            Block(5, 96, 576, 96, HSwish,
                  SEModule(96, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
            Block(5, 96, 576, 96, HSwish,
                  SEModule(96, reduction=4, use_norm=True, use_pooling=True, gate=HSigmoid), 1),
        )

        self.convbnact = ConvBnAct(96, 576, 1, 0, 1, act_layer=HSwish)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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
