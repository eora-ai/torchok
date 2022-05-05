"""TorchOK ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py'

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman
"""
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn

from src.models.modules.utils.create_attn import create_attn
from src.models.backbones.base import BaseModel, FeatureInfo
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.backbones.utils.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'first_conv': 'conv1',
        'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
        interpolation='bicubic',
        crop_pct=0.95),
    'resnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth',
        interpolation='bicubic',
        crop_pct=0.95),

    # Squeeze-Excitation ResNets
    'seresnet18': _cfg(interpolation='bicubic'),
    'seresnet34': _cfg(interpolation='bicubic'),
    'seresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth',
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
                 attn_layer: str = None):
        """Init BasicBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Type of attention block.
        """
        super(BasicBlock, self).__init__()
        out_block_channels = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_block_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_block_channels)
        self.se = create_attn(attn_layer, out_block_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck building block for ResNet architecture."""

    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Sequential = None,
                 attn_layer: str = None):
        """Init Bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride.
            downsample: Downsample module.
            attn_layer: Type of attention block.
        """
        super(Bottleneck, self).__init__()
        out_block_channels = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_block_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_block_channels)
        self.se = create_attn(attn_layer, out_block_channels)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)

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
            block: Type of BasicBlock.
            layers: Number of layers.
            in_chans: Input channels.
            block_args: Arguments for block_args.
        """
        self.block_args = block_args or dict()
        self.inplanes = 64
        self.channels = [64, 128, 256, 512]
        self.num_features = 512 * block.expansion

        super(ResNet, self).__init__(self.channels)

        self.conv1 = nn.Conv2d(in_chans, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.act1 = nn.ReLU(inplace=True)
        self._feature_info = [FeatureInfo(channel_number=self.channels[0], reduction=2, module_name='act1')]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(block, self.channels[0], layers[0], **self.block_args)
        self._feature_info.append(FeatureInfo(channel_number=self.channels[0], reduction=2, module_name='layer1'))

        self.layer2 = self.__make_layer(block, self.channels[1], layers[1], stride=2, **self.block_args)
        self._feature_info.append(FeatureInfo(channel_number=self.channels[1], reduction=2, module_name='layer2'))

        self.layer3 = self.__make_layer(block, self.channels[2], layers[2], stride=2, **self.block_args)
        self._feature_info.append(FeatureInfo(channel_number=self.channels[2], reduction=2, module_name='layer3'))

        self.layer4 = self.__make_layer(block, self.channels[3], layers[3], stride=2, **self.block_args)
        self._feature_info.append(FeatureInfo(channel_number=self.channels[3], reduction=2, module_name='layer4'))

        self.create_hooks()

        self.init_weights()

    def __make_layer(self,
                     block: Union[BasicBlock, Bottleneck],
                     in_channel: int,
                     blocks: int,
                     stride: int = 1,
                     **kwargs) -> nn.Sequential:

        downsample = None

        if stride != 1 or self.inplanes != in_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, in_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channel * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, in_channel, stride, downsample=downsample, **kwargs))

        self.inplanes = in_channel * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, in_channel, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def create_resnet(variant, pretrained=False, **kwargs):
    """Is creating ResNet based model."""
    return build_model_with_cfg(ResNet, variant, default_cfg=default_cfgs[variant],
                                pretrained=pretrained, **kwargs, pretrained_strict=False)


@register_model
def resnet18(pretrained=False, **kwargs):
    """Is constructing a ResNet-18 model."""
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return create_resnet('resnet18', pretrained, **model_args)


@register_model
def resnet34(pretrained=False, **kwargs):
    """Is constructing a ResNet-34 model."""
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet34', pretrained, **model_args)


@register_model
def resnet50(pretrained=False, **kwargs):
    """Is constructing a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return create_resnet('resnet50', pretrained, **model_args)


@register_model
def resnet101(pretrained=False, **kwargs):
    """Is constructing a ResNet-101 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return create_resnet('resnet101', pretrained, **model_args)


@register_model
def resnet152(pretrained=False, **kwargs):
    """Is constructing a ResNet-152 model."""
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return create_resnet('resnet152', pretrained, **model_args)


@register_model
def seresnet18(pretrained=False, **kwargs):
    """Is constructing a SEResNet-18 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': 'se'})
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=block_args, **kwargs)
    return create_resnet('seresnet18', pretrained, **model_args)


@register_model
def seresnet34(pretrained=False, **kwargs):
    """Is constructing a SEResNet-34 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': 'se'})
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet34', pretrained, **model_args)


@register_model
def seresnet50(pretrained=False, **kwargs):
    """Is constructing a SEResNet-50 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': 'se'})
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet50', pretrained, **model_args)


@register_model
def seresnet101(pretrained=False, **kwargs):
    """Is constructing a SEResNet-101 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': 'se'})
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet101', pretrained, **model_args)


@register_model
def seresnet152(pretrained=False, **kwargs):
    """Is constructing a SEResNet-152 model."""
    block_args = kwargs.pop('block_args', dict())
    block_args.update({'attn_layer': 'se'})
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=block_args, **kwargs)
    return create_resnet('seresnet152', pretrained, **model_args)
