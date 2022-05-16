"""TorchOK ResNet

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn

from src.constructor import BACKBONES
from src.models.modules.blocks.se import SEModule
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.base_model import BaseModel, FeatureInfo
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
        'first_conv': 'conv1',
        'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet
    'resnet18': _cfg(url=''),
    'resnet34': _cfg(
        url=''),
    'resnet50': _cfg(
        url='',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet101': _cfg(
        url='',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet152': _cfg(
        url='',
        interpolation='bicubic',
        crop_pct=0.95),

    # Squeeze-Excitation ResNets
    'seresnet18': _cfg(interpolation='bicubic'),
    'seresnet34': _cfg(interpolation='bicubic'),
    'seresnet50': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet101': _cfg(interpolation='bicubic'),
    'seresnet152': _cfg(interpolation='bicubic')
}


class BasicBlock(nn.Module):
    """BasicBlock bulding block for ResNet architecture."""

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attn_layer: Optional[nn.Module] = None):
        """Init BasicBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Attention block.
        """
        super().__init__()
        out_block_channels = out_channels * self.expansion

        self.convbnact = ConvBnAct(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv = nn.Conv2d(out_channels, out_block_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_block_channels)
        self.se = attn_layer(out_block_channels) if attn_layer is not None else None
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        """Forward method."""
        identity = x

        out = self.convbnact(x)
        out = self.conv(out)
        out = self.bn(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck building block for ResNet architecture."""

    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 attn_layer: Optional[nn.Module] = None):
        """Init Bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Attention block.
        """
        super().__init__()
        out_block_channels = out_channels * self.expansion

        self.convbnact1 = ConvBnAct(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.convbnact2 = ConvBnAct(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)

        self.conv = nn.Conv2d(out_channels, out_block_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_block_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = attn_layer(out_block_channels) if attn_layer is not None else None

        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        """Forward method."""
        identity = x

        out = self.convbnact1(x)
        out = self.convbnact2(out)

        out = self.conv(out)
        out = self.bn(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class ResNet(BaseModel):
    """ResNet model."""

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: List[int],
                 in_chans: int = 3,
                 block_args: Dict = None):
        """Init ResNet.

        Args:
            block: Type of block.
            layers: Number of layers.
            in_chans: Input channels.
            block_args: Arguments for block_args.
        """
        self.block_args = block_args or dict()
        self.inplanes = 64
        self.channels = [64, 128, 256, 512]
        self.num_features = 512 * block.expansion

        super().__init__()

        self.convbnact = ConvBnAct(in_chans, self.channels[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(block, self.channels[0], layers[0], **self.block_args)
        self.layer2 = self.__make_layer(block, self.channels[1], layers[1], stride=2, **self.block_args)
        self.layer3 = self.__make_layer(block, self.channels[2], layers[2], stride=2, **self.block_args)
        self.layer4 = self.__make_layer(block, self.channels[3], layers[3], stride=2, **self.block_args)

        self.create_hooks(output_channels=self.channels * block.expansion,
                          module_names=['layer1', 'layer2', 'layer3', 'layer4'],
                          strides=[1, 2, 2, 2])

    def __make_layer(self,
                     block: Union[BasicBlock, Bottleneck],
                     in_channel: int,
                     blocks: int,
                     stride: int = 1,
                     **block_args) -> nn.Sequential:
        """Create layer - abstraction for convenient work.

        Args:
            block: Abstraction for convenient work.
            in_channel: Input channels.
            blocks: Number of blocks.
            stride: Stride.
            block_args: Arguments of block.(for example: attn_layer)
        """
        downsample = None

        if stride != 1 or self.inplanes != in_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, in_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channel * block.expansion)
            )

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

    def get_features_info(self, output_channels, module_names, strides):
        features_info = []
        for num_channels, module_name, stride in zip(output_channels, module_names, strides):
            feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=stride)
            features_info.append(feature_info)
        return features_info

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        return self.num_features


def create_resnet(variant, pretrained=False, **kwargs):
    """Is creating ResNet based model."""
    return build_model_with_cfg(ResNet, default_cfg=default_cfgs[variant],
                                pretrained=pretrained, **kwargs)


@BACKBONES.register_class
def resnet18(pretrained=False, **kwargs):
    """Is constructing a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return create_resnet('resnet18', pretrained, **model_args)


@BACKBONES.register_class
def resnet34(pretrained=False, **kwargs):
    """Is constructing a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet34', pretrained, **model_args)


@BACKBONES.register_class
def resnet50(pretrained=False, **kwargs):
    """Is constructing a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet50', pretrained, **model_args)


@BACKBONES.register_class
def resnet101(pretrained=False, **kwargs):
    """Is constructing a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return create_resnet('resnet101', pretrained, **model_args)


@BACKBONES.register_class
def resnet152(pretrained=False, **kwargs):
    """Is constructing a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return create_resnet('resnet152', pretrained, **model_args)


@BACKBONES.register_class
def seresnet18(pretrained=False, **kwargs):
    """Is constructing a SEResNet-18 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=block_args, **kwargs)
    return create_resnet('seresnet18', pretrained, **model_args)


@BACKBONES.register_class
def seresnet34(pretrained=False, **kwargs):
    """Is constructing a SEResNet-34 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet34', pretrained, **model_args)


@BACKBONES.register_class
def seresnet50(pretrained=False, **kwargs):
    """Is constructing a SEResNet-50 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet50', pretrained, **model_args)


@BACKBONES.register_class
def seresnet101(pretrained=False, **kwargs):
    """Is constructing a SEResNet-101 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet101', pretrained, **model_args)


@BACKBONES.register_class
def seresnet152(pretrained=False, **kwargs):
    """Is constructing a SEResNet-152 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': SEModule})
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet152', pretrained, **model_args)