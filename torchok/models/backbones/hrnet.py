"""TorchOK HRNet.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from typing import Any, List, Tuple, Union, Dict

import torch.nn as nn
from torch import Tensor

from torchok.constructor import BACKBONES
from torchok.models.base import BaseModel
from torchok.models.modules.bricks.convbnact import ConvBnAct
from torchok.models.modules.blocks.basicblock import BasicBlock
from torchok.models.modules.blocks.bottleneck import Bottleneck
from torchok.models.backbones.utils.helpers import build_model_with_cfg
from torchok.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


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


default_cfgs = {
    'hrnet_w18_small': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w18_small-torchok.pth'),
    'hrnet_w18_small_v2': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w18_small_v2-torchok.pth'),
    'hrnet_w18': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w18-torchok.pth'),
    'hrnet_w30': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w30-torchok.pth'),
    'hrnet_w32': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w32-torchok.pth'),
    'hrnet_w40': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w40-torchok.pth'),
    'hrnet_w44': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w44-torchok.pth'),
    'hrnet_w48': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w48-torchok.pth'),
    'hrnet_w64': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/hrnet_w64-torchok.pth'),
}

cfg_cls = dict(
    hrnet_w18_small=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(1,),
            NUM_CHANNELS=(32,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(16, 32),
        ),
        STAGE3=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(16, 32, 64)
        ),
        STAGE4=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(16, 32, 64, 128)
        ),
    ),

    hrnet_w18_small_v2=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(2,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2),
            NUM_CHANNELS=(18, 36)
        ),
        STAGE3=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2),
            NUM_CHANNELS=(18, 36, 72)
        ),
        STAGE4=dict(
            NUM_MODULES=2,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(2, 2, 2, 2),
            NUM_CHANNELS=(18, 36, 72, 144)
        ),
    ),

    hrnet_w18=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(18, 36)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(18, 36, 72)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(18, 36, 72, 144)
        ),
    ),

    hrnet_w30=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(30, 60)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(30, 60, 120)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(30, 60, 120, 240)
        ),
    ),

    hrnet_w32=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(32, 64)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(32, 64, 128)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(32, 64, 128, 256)
        ),
    ),

    hrnet_w40=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(40, 80)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(40, 80, 160)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(40, 80, 160, 320)
        ),
    ),

    hrnet_w44=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(44, 88)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(44, 88, 176)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(44, 88, 176, 352)
        ),
    ),

    hrnet_w48=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(48, 96)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(48, 96, 192)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(48, 96, 192, 384)
        ),
    ),

    hrnet_w64=dict(
        STEM_WIDTH=64,
        STAGE1=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=1,
            BLOCK='BOTTLENECK',
            NUM_BLOCKS=(4,),
            NUM_CHANNELS=(64,)
        ),
        STAGE2=dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(64, 128)
        ),
        STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(64, 128, 256)
        ),
        STAGE4=dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(64, 128, 256, 512)
        ),
    )
)


class HighResolutionModule(nn.Module):
    """HighResolutionModule is logical module of HighResolutionNet."""
    def __init__(self,
                 num_branches: int,
                 blocks: Union[Bottleneck, BasicBlock],
                 num_blocks: Tuple[int],
                 num_inchannels: Tuple[int],
                 num_outchannels: Tuple[int]):
        """Init HighResolutionModule.

        Args:
            num_branches: Number of branches.
            blocks: Type of block.
            num_blocks: Number of blocks.
            num_inchannels: Number of input channels.
            num_outchannels: Number of output channels.
        """
        super().__init__()

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_outchannels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU()

    def _make_one_branch(self,
                         branch_index: int,
                         block: Union[Bottleneck, BasicBlock],
                         num_blocks: Tuple[int],
                         num_channels: Tuple[int],
                         stride: int = 1) -> nn.Sequential:
        """The method creates one branch in the HRNet.

        Args:
            branch_index: Index of branch.
            block: Block type.
            num_blocks: Number of blocks.
            num_channels: Number of channels.
            stride: Stride.
        """
        downsample = None
        expanded_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != expanded_channels:
            downsample = ConvBnAct(in_channels=self.num_inchannels[branch_index],
                                   out_channels=expanded_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False,
                                   act_layer=None)

        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = expanded_channels
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self,
                       num_branches: int,
                       block: Union[Bottleneck, BasicBlock],
                       num_blocks: Tuple[int],
                       num_channels: Tuple[int]):
        """The method creates branches in the HRNet.

        Args:
            num_branches: Number of branches.
            block: Block type.
            num_blocks: Number of blocks.
            num_channels: Number of channels.
        """
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList:
        """The method creates fuse layers in the HRNet."""
        if self.num_branches == 1:
            return nn.Identity()

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        ConvBnAct(in_channels=num_inchannels[j],
                                  out_channels=num_inchannels[i],
                                  kernel_size=1,
                                  padding=0,
                                  stride=1,
                                  bias=False,
                                  act_layer=None),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        num_outchannels_conv3x3 = num_inchannels[i] if k == i - j - 1 else num_inchannels[j]
                        act_layer = None if k == i - j - 1 else nn.ReLU
                        conv3x3s.append(ConvBnAct(in_channels=num_inchannels[j],
                                                  out_channels=num_outchannels_conv3x3,
                                                  kernel_size=3,
                                                  padding=1,
                                                  stride=2,
                                                  bias=False,
                                                  act_layer=act_layer))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Forward method."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))

        return x_fuse

    def get_num_inchannels(self) -> List[int]:
        """Number of input channels."""
        return self.num_inchannels


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(BaseModel):
    """HighResolutionNet model."""

    def __init__(self,
                 cfg: Dict[str, Any],
                 in_chans: int = 3):
        """Init HighResolutionNet.

        Args:
            cfg: Model config.
            in_chans: Input channels.
        """
        super().__init__()

        stem_width = cfg['STEM_WIDTH']

        self.convbnact1 = ConvBnAct(in_channels=in_chans,
                                    out_channels=stem_width,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2,
                                    bias=False)

        self.convbnact2 = ConvBnAct(in_channels=stem_width,
                                    out_channels=64,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2,
                                    bias=False)

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
        self.stage4, self.out_channels = self.__make_stage(self.stage4_cfg, num_channels)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_transition_layer(self,
                                num_channels_pre_layer: List[int],
                                num_channels_cur_layer: List[int]) -> nn.ModuleList:
        """The method creates transiton_layer in the HRNet.

        Args:
            num_channels_pre_layer: Number of channels in previous layer.
            num_channels_cur_layer: Number of channels in current layer.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        ConvBnAct(in_channels=num_channels_pre_layer[i],
                                  out_channels=num_channels_cur_layer[i],
                                  kernel_size=3,
                                  padding=1,
                                  stride=1,
                                  bias=False))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        ConvBnAct(in_channels=inchannels,
                                  out_channels=outchannels,
                                  kernel_size=3,
                                  padding=1,
                                  stride=2,
                                  bias=False))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def __make_layer(self,
                     block: Union[Bottleneck, BasicBlock],
                     in_channels: int,
                     out_channels: int,
                     num_blocks: int,
                     stride: int = 1) -> nn.Sequential:
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
            downsample = ConvBnAct(in_channels=in_channels,
                                   out_channels=out_channels * block.expansion,
                                   kernel_size=1,
                                   padding=0,
                                   stride=stride,
                                   bias=False,
                                   act_layer=None)

        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def __make_stage(self,
                     layer_config: Dict[str, Any],
                     num_inchannels: Tuple[int]) -> nn.Sequential:
        """The method creates stage in the HRNet.

        Args:
            layer_config: Layer configurations.
            num_inchannels: Number of input channels.
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]

        modules = []
        for i in range(num_modules):
            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels, num_channels)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def __stages(self, x: Tensor) -> List[Tensor]:
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
        x = self.convbnact1(x)
        x = self.convbnact2(x)

        yl = self.__stages(x)

        return yl

    def forward_backbone_features(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward backbone features and input tensor.

        Args:
            x: Input tensor.
        """
        features = self.forward(x)
        features = [x] + features
        return features[1:], features

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.out_channels


def create_hrnet(variant: str, pretrained: bool = False, **model_kwargs):
    """Create HighResolutionNet base model.

    Args:
        variant: Backbone type.
        pretrained: If True the pretrained weights will be loaded.
        model_kwargs: Kwargs for model (for example in_chans).
    """
    return build_model_with_cfg(
        HighResolutionNet, pretrained, default_cfg=default_cfgs[variant],
        model_cfg=cfg_cls[variant], **model_kwargs)


@BACKBONES.register_class
def hrnet_w18_small(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18_small model."""
    return create_hrnet('hrnet_w18_small', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w18_small_v2(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18_small_v2 model."""
    return create_hrnet('hrnet_w18_small_v2', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w18(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w18 model."""
    return create_hrnet('hrnet_w18', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w30(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w30 model."""
    return create_hrnet('hrnet_w30', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w32(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w32 model."""
    return create_hrnet('hrnet_w32', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w40(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w40 model."""
    return create_hrnet('hrnet_w40', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w44(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w44 model."""
    return create_hrnet('hrnet_w44', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w48(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w48 model."""
    return create_hrnet('hrnet_w48', pretrained, **kwargs)


@BACKBONES.register_class
def hrnet_w64(pretrained: bool = False, **kwargs):
    """It's constructing a hrnet_w64 model."""
    return create_hrnet('hrnet_w64', pretrained, **kwargs)
