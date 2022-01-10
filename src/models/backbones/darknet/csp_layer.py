# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
# from mmcv.runner import BaseModule

from src.models.backbones.efficientnet.efficientnet_blocks import ConvBnAct, DepthwiseSeparableConv


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.
    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 norm_kwargs=dict(momentum=0.03, eps=0.001),
                 act_layer=nn.Hardswish):
        super(DarknetBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConv if use_depthwise else ConvBnAct
        self.conv1 = ConvBnAct(
            in_channels,
            hidden_channels,
            1,
            norm_kwargs=norm_kwargs,
            act_layer=act_layer)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            norm_kwargs=norm_kwargs,
            act_layer=act_layer)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 norm_kwargs=dict(momentum=0.03, eps=0.001),
                 act_layer=nn.Hardswish,
                 ):
        super(CSPLayer, self).__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBnAct(
            in_channels,
            mid_channels,
            1,
            norm_kwargs=norm_kwargs,
            act_layer=act_layer)
        self.short_conv = ConvBnAct(
            in_channels,
            mid_channels,
            1,
            norm_kwargs=norm_kwargs,
            act_layer=act_layer)
        self.final_conv = ConvBnAct(
            2 * mid_channels,
            out_channels,
            1,
            norm_kwargs=norm_kwargs,
            act_layer=act_layer)

        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                norm_kwargs=norm_kwargs,
                act_layer=act_layer) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)
