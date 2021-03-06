"""TorchOK ResNet.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Union, List, Dict

import torch
import torch.nn as nn

from src.constructor import BACKBONES
from src.models.modules.blocks.se import SEModule
from src.models.modules.blocks.basicblock import BasicBlock
from src.models.modules.blocks.bottleneck import Bottleneck
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.backbones import BaseBackbone
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


def _cfg(url='', **kwargs):
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
    # ResNet
    'resnet18': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/resnet18-torchok.pth'),
    'resnet34': _cfg(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/resnet34-torchok.pth'),
    'resnet50': _cfg(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/resnet50-torchok.pth',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet101': _cfg(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/resnet101-torchok.pth',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet152': _cfg(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/resnet152-torchok.pth',
        interpolation='bicubic',
        crop_pct=0.95),

    # Squeeze-Excitation ResNets
    'seresnet18': _cfg(interpolation='bicubic'),
    'seresnet34': _cfg(interpolation='bicubic'),
    'seresnet50': _cfg(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/seresnet50-torchok.pth',
        interpolation='bicubic'),
    'seresnet101': _cfg(interpolation='bicubic'),
    'seresnet152': _cfg(interpolation='bicubic')
}


class ResNet(BaseBackbone):
    """ResNet model."""

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: List[int],
                 in_channels: int = 3,
                 block_args: Dict = None):
        """Init ResNet.

        Args:
            block: Block type class.
            layers: Number of layers.
            in_channels: Input channels.
            block_args: Arguments for block_args.
        """
        super().__init__(in_channels=in_channels)
        self.block_args = block_args or dict()
        self.inplanes = 64
        self.channels = [64, 128, 256, 512]
        self._out_channels = 512 * block.expansion
        self._out_feature_channels = [in_channels] + self.channels * block.expansion,

        self.convbnact = ConvBnAct(in_channels, self.channels[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(block, self.channels[0], layers[0], **self.block_args)
        self.layer2 = self.__make_layer(block, self.channels[1], layers[1], stride=2, **self.block_args)
        self.layer3 = self.__make_layer(block, self.channels[2], layers[2], stride=2, **self.block_args)
        self.layer4 = self.__make_layer(block, self.channels[3], layers[3], stride=2, **self.block_args)

    def __make_layer(self,
                     block: Union[BasicBlock, Bottleneck],
                     in_channel: int,
                     blocks: int,
                     stride: int = 1,
                     **block_args) -> nn.Sequential:
        """Create Resnet module which contain downsample and one of BasicBlock or Bottleneck modules.

        Args:
            block: Block type class.
            in_channel: Input channels.
            blocks: Number of blocks.
            stride: Stride for downsample module.
            block_args: Arguments of block.(for example: attn_layer)
        """
        downsample = None

        if stride != 1 or self.inplanes != in_channel * block.expansion:
            downsample = ConvBnAct(self.inplanes,
                                   in_channel * block.expansion,
                                   kernel_size=1,
                                   padding=0,
                                   stride=stride,
                                   act_layer=None)

        layers = []
        layers.append(block(self.inplanes, in_channel, stride, downsample=downsample, **block_args))

        self.inplanes = in_channel * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, in_channel, **block_args))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward method."""
        x = self.convbnact(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward method."""
        features = [x]

        x = self.convbnact(x)
        x = self.maxpool(x)
        features.append(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


def create_resnet(variant, pretrained=False, **kwargs):
    """Create ResNet base model."""
    return build_model_with_cfg(ResNet, default_cfg=default_cfgs[variant],
                                pretrained=pretrained, **kwargs)


@BACKBONES.register_class
def resnet18(pretrained=False, **kwargs):
    """It's constructing a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return create_resnet('resnet18', pretrained, **model_args)


@BACKBONES.register_class
def resnet34(pretrained=False, **kwargs):
    """It's constructing a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet34', pretrained, **model_args)


@BACKBONES.register_class
def resnet50(pretrained=False, **kwargs):
    """It's constructing a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet50', pretrained, **model_args)


@BACKBONES.register_class
def resnet101(pretrained=False, **kwargs):
    """It's constructing a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return create_resnet('resnet101', pretrained, **model_args)


@BACKBONES.register_class
def resnet152(pretrained=False, **kwargs):
    """It's constructing a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return create_resnet('resnet152', pretrained, **model_args)


@BACKBONES.register_class
def seresnet18(pretrained=False, **kwargs):
    """It's constructing a SEResNet-18 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=block_args, **kwargs)
    return create_resnet('seresnet18', pretrained, **model_args)


@BACKBONES.register_class
def seresnet34(pretrained=False, **kwargs):
    """It's constructing a SEResNet-34 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet34', pretrained, **model_args)


@BACKBONES.register_class
def seresnet50(pretrained=False, **kwargs):
    """It's constructing a SEResNet-50 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet50', pretrained, **model_args)


@BACKBONES.register_class
def seresnet101(pretrained=False, **kwargs):
    """It's constructing a SEResNet-101 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet101', pretrained, **model_args)


@BACKBONES.register_class
def seresnet152(pretrained=False, **kwargs):
    """It's constructing a SEResNet-152 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet152', pretrained, **model_args)
