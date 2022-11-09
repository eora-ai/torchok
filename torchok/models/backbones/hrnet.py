"""TorchOK HRNet.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.hrnet import _BN_MOMENTUM, BasicBlock, blocks_dict, Bottleneck, cfg_cls, HighResolutionModule
from torch import Tensor

from torchok.constructor import BACKBONES
from torchok.models.backbones import BaseBackbone


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1',
        'classifier': 'classifier',
        **kwargs
    }


url_base_path = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/'
default_cfgs = {
    'hrnet_w18_small': _cfg(url=f'{url_base_path}/hrnet_w18_small_v1-f460c6bc.pth'),
    'hrnet_w18_small_v2': _cfg(url=f'{url_base_path}/hrnet_w18_small_v2-4c50a8cb.pth'),
    'hrnet_w18': _cfg(url=f'{url_base_path}/hrnetv2_w18-8cb57bb9.pth'),
    'hrnet_w30': _cfg(url=f'{url_base_path}/hrnetv2_w30-8d7f8dab.pth'),
    'hrnet_w32': _cfg(url=f'{url_base_path}/hrnetv2_w32-90d8c5fb.pth'),
    'hrnet_w40': _cfg(url=f'{url_base_path}/hrnetv2_w40-7cd397a4.pth'),
    'hrnet_w44': _cfg(url=f'{url_base_path}/hrnetv2_w44-c9ac8c18.pth'),
    'hrnet_w48': _cfg(url=f'{url_base_path}/hrnetv2_w48-abd2e6ab.pth'),
    'hrnet_w64': _cfg(url=f'{url_base_path}/hrnetv2_w64-b47cc881.pth'),
}


class HighResolutionNet(BaseBackbone):
    """HighResolutionNet model."""

    def __init__(self, cfg: Dict[str, Any], in_channels: int = 3):
        """Init HighResolutionNet.

        Args:
            cfg: Model config.
            in_channels: Input channels.
        """
        super().__init__(in_channels=in_channels,
                         out_channels=cfg['STAGE4']['NUM_CHANNELS'])
        self._out_encoder_channels = cfg['STAGE4']['NUM_CHANNELS']

        stem_width = cfg['STEM_WIDTH']
        self.conv1 = nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self.__make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self.__make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self.__make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self.__make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self.__make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self.__make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, out_channels = self.__make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.init_weights()

    @torch.jit.ignore
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_transition_layer(self, num_channels_pre_layer: List[int],
                                num_channels_cur_layer: List[int]) -> nn.Module:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def __make_layer(self, block: Union[Bottleneck, BasicBlock], in_channels: int,
                     out_channels: int, num_blocks: int, stride: int = 1) -> nn.Module:
        """The method creates layer in the HRNet.

        Args:
            block: Block type.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_blocks: Number of blocks.
            stride: Stride.
        """
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion, momentum=_BN_MOMENTUM),
            )

        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def __make_stage(self, layer_config: Dict[str, Any], in_channels: List[int],
                     multi_scale_output: bool = True) -> nn.Module:
        """The method creates stage in the HRNet.

        Args:
            layer_config: Layer configurations.
            in_channels: Number of input channels.
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, in_channels, num_channels, fuse_method, reset_multi_scale_output)
            )
            in_channels = modules[-1].get_num_in_chs()

        return nn.Sequential(*modules), in_channels

    def forward_stages(self, x: Tensor) -> List[Tensor]:
        """The method forward the tensor through all stages.

        Args:
            x: Input tensor.
        """
        x = self.layer1(x)

        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward method.

        Args:
            x: Input tensor.
        """
        x = self.forward_stem(x)
        yl = self.forward_stages(x)

        return yl

    def forward_stem(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward_features(self, x: Tensor) -> List[Tensor]:
        """Forward backbone features and input tensor.

        Args:
            x: Input tensor.
        """
        return [x] + self.forward(x)

    def get_stages(self, stage: int) -> nn.Module:
        """Return modules corresponding the given model stage and all previous stages.
        For example, `0` must stand for model stem. `1` must stand for models stem and
        the first global layer of the model (`layer1` in the resnet), etc.

        Args:
            stage: index of the models stage.
        """
        output = [self.conv1, self.bn1, self.act1, self.conv2, self.bn2, self.act2]
        layers = [[self.layer1],
                  [self.transition1, self.stage2],
                  [self.transition2, self.stage3],
                  [self.transition3, self.stage4]]
        for i in range(stage):
            output += layers[i]
        return nn.ModuleList(output)


def _create_hrnet(variant: str, pretrained: bool = False, **model_kwargs):
    """Create HighResolutionNet base model.

    Args:
        variant: Backbone type.
        pretrained: If True the pretrained weights will be loaded.
        model_kwargs: Kwargs for model (for example in_channels).
    """
    return build_model_with_cfg(HighResolutionNet, variant, pretrained, model_cfg=cfg_cls[variant],
                                pretrained_strict=False, kwargs_filter=('num_classes', 'global_pool', 'in_chans'),
                                **model_kwargs)


@BACKBONES.register_class
def hrnet_w18_small(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18_small model."""
    return _create_hrnet('hrnet_w18_small', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w18_small_v2(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18_small_v2 model."""
    return _create_hrnet('hrnet_w18_small_v2', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w18(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18 model."""
    return _create_hrnet('hrnet_w18', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w30(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w30 model."""
    return _create_hrnet('hrnet_w30', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w32(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w32 model."""
    return _create_hrnet('hrnet_w32', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w40(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w40 model."""
    return _create_hrnet('hrnet_w40', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w44(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w44 model."""
    return _create_hrnet('hrnet_w44', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w48(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w48 model."""
    return _create_hrnet('hrnet_w48', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w64(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w64 model."""
    return _create_hrnet('hrnet_w64', pretrained, **kwargs)
