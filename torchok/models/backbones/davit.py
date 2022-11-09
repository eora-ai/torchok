"""TorchOK DaViT.

Adapted from https://github.com/dingmyu/davit/blob/main/mmseg/mmseg/models/backbones/davit.py
Licensed under MIT License [see LICENSE for details]
"""
import itertools
from typing import List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor

from torchok.constructor import BACKBONES
from torchok.models.backbones import BaseBackbone
from torchok.models.modules.bricks.mlp import Mlp


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        # 'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'davit_t': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/davit-t_torchok.pth'),
    'davit_s': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/davit-s_torchok.pth'),
    'davit_b': _cfg(url='https://torchok-hub.s3.eu-west-1.amazonaws.com/davit-b_torchok.pth'),
}


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 96, overlapped: bool = False):
        """Init PatchEmbed.

        Args:
            patch_size: Patch size.
            in_channels: Input channels.
            embed_dim: Embedding dimension.
            overlapped: Overlapping.
        """
        super().__init__()
        self.patch_size = patch_size

        if patch_size == 4:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(7, 7), stride=patch_size, padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        elif patch_size == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel, stride=patch_size, padding=pad)
            self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: Tensor, size: Tuple[int]) -> Tuple[Tensor, Tuple[int]]:
        """Forward method."""
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size[1]))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize


class ConvPosEnc(nn.Module):
    """Convolution positional encoding."""

    def __init__(self, dim: int, kernel_size: int = 3, use_act: bool = False, norm_layer: Optional[str] = None):
        """Init ConvPosEnc.

        Args:
            dim: Dimension.
            kernel_size: Kernel size.
            use_act: If True, will use GELU activation.
            norm_layer: Type of normalization.
        """
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              kernel_size,
                              1,
                              kernel_size // 2,
                              groups=dim)
        self.norm_layer = norm_layer
        if self.norm_layer == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.norm_layer == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if use_act else None

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        """Forward method."""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.norm_layer == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.norm_layer == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)

        if self.activation is not None:
            x = x + self.activation(feat)
        return x


class ChannelAttention(nn.Module):
    """Channel based multi-head self attention module."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        """Init Channel Attention.

        Args:
            dim: Dimension size.
            num_heads: Number of heads.
            qkv_bias: Query-Key-Value bias.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        """Init WindowAttention.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            qkv_bias:  If True, add a learnable bias to query, key, value. Default: True
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: Input tensor.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    """Channel Block of DaViT."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop_path: float = 0., act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,
                 ffn: bool = True, cpe_act: bool = False):
        """Init ChannelBlock.

        Args:
            dim: Dimension.
            num_heads: Number of heads.
            mlp_ratio: Multilayer perceptron ratio for hidden dim.
            qkv_bias: Query-Key-Value bias.
            drop_path: Drop path.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            ffn: If True, will use Mlp.
            cpe_act: If True, ConvPosEnc will use activation.

        """
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act),
                                  ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, inputs):
        """Forward method."""
        x, size = inputs
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


class SpatialBlock(nn.Module):
    """Windows Block.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        drop_path: Stochastic depth rate. Default: 0.0
        act_layer: Activation layer. Default: nn.GELU
        norm_layer: Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_path: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm, ffn: bool = True, cpe_act: bool = False):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act),
                                  ConvPosEnc(dim=dim, kernel_size=3, use_act=cpe_act)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, inputs):
        """Forward method."""
        x, size = inputs
        H, W = size
        B, L, C = x.shape

        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = self._window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)

        x = self._window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

    def _window_partition(self, x: Tensor, window_size: int) -> Tensor:
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class DaViT(BaseBackbone):
    """ Dual Attention Transformer"""

    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 4, depths=(1, 1, 3, 1),
                 embed_dims: Tuple[int] = (64, 128, 192, 256), num_heads: Tuple[int] = (3, 6, 12, 24),
                 window_size: int = 7, mlp_ratio: float = 4., qkv_bias: bool = True, drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm, overlapped_patch: bool = False, ffn: bool = True,
                 cpe_act: bool = False):
        """Init DaViT.

        Args:
            img_size: Input image size.
            in_channels: Number of input image channels.
            patch_size: Patch size.
            embed_dims: Patch embedding dimension.
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__(in_channels, embed_dims[-1])
        self.img_size = img_size

        architecture = [[index] * item for index, item in enumerate(depths)]
        self.attention_types = ('spatial', 'channel')
        self.architecture = architecture
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)
        self._out_encoder_channels = embed_dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=patch_size if i == 0 else 2,
                       in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                       embed_dim=self.embed_dims[i],
                       overlapped=overlapped_patch)
            for i in range(self.num_stages)])

        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))

            block = nn.ModuleList([
                nn.Sequential(*[
                    ChannelBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=norm_layer,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ) if attention_type == 'channel' else
                    SpatialBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=norm_layer,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(self.attention_types)]
                ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block)
        self.main_blocks = nn.ModuleList(main_blocks)

        # add a norm layer for each output
        for i_layer in range(self.num_stages):
            layer = norm_layer(self.embed_dims[i_layer])  # if i_layer != 0 else nn.Identity()
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.init_weights()

    @torch.jit.ignore
    def init_weights(self):
        """Weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _forward_stages(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
        features = [x]
        sizes = [size]
        branches = [0]

        for block_index, block_param in enumerate(self.architecture):
            branch_ids = sorted(set(block_param))
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)
            for layer_index, branch_id in enumerate(block_param):
                inputs = (features[branch_id], sizes[branch_id])
                features[branch_id], _ = self.main_blocks[block_index][layer_index](inputs)
        return features, sizes

    def forward_features(self, x: Tensor) -> List[Tensor]:
        """Forward method for getting backbone feature maps.
           They are mainly used for segmentation and detection tasks.

        Args:
            x: Input tensor.

        Returns:
            This method return tuple of tensors.
            Each tensor has (B, C, H, W) shape structure.
        """
        input_tensor = x
        features, sizes = self._forward_stages(x)

        outs = []
        for i in range(self.num_stages):
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(features[i])
            H, W = sizes[i]
            out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return [input_tensor, *outs]

    def forward(self, x: Tensor) -> Tensor:
        """Forward method"""
        features, sizes = self._forward_stages(x)
        last_stage = self.num_stages - 1

        norm_layer = getattr(self, f'norm{last_stage}')
        x_out = norm_layer(features[last_stage])
        H, W = sizes[last_stage]
        out = x_out.view(-1, H, W, self.embed_dims[last_stage]).permute(0, 3, 1, 2).contiguous()

        return out

    def get_stages(self, stage: int) -> nn.Module:
        """Return modules corresponding the given model stage and all previous stages.
        For example, `0` must stand for model stem. `1` must stand for models stem and
        the first global layer of the model (`layer1` in the resnet), etc.

        Args:
            stage: index of the models stage.
        """
        logging.warning("DaViT does not support `get_stages`. Return the whole model")
        return self


def _create_davit(variant: str, pretrained: bool = False, **kwargs):
    """Create DaViT base model.

    Args:
        variant: Backbone type.
        pretrained: If True the pretrained weights will be loaded.
        kwargs: Kwargs for model (for example in_chans).
    """
    kwargs_filter = ('num_classes', 'global_pool', 'in_chans')
    return build_model_with_cfg(DaViT, variant, pretrained, pretrained_strict=False,
                                pretrained_cfg=default_cfgs[variant], kwargs_filter=kwargs_filter, **kwargs)


@BACKBONES.register_class
def davit_t(pretrained: bool = False, **kwargs):
    """It's constructing a davit_t model."""
    model_kwargs = dict(embed_dims=(96, 192, 384, 768), depths=(1, 1, 3, 1), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_davit('davit_t', pretrained, **model_kwargs)


@BACKBONES.register_class
def davit_s(pretrained: bool = False, **kwargs):
    """It's constructing a davit_s model."""
    model_kwargs = dict(embed_dims=(96, 192, 384, 768), depths=(1, 1, 9, 1), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_davit('davit_s', pretrained, **model_kwargs)


@BACKBONES.register_class
def davit_b(pretrained: bool = False, **kwargs):
    """It's constructing a davit_b model."""
    model_kwargs = dict(embed_dims=(128, 256, 512, 1024), depths=(1, 1, 9, 1), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_davit('davit_b', pretrained, **model_kwargs)
