"""TorchOK Swin Transformer V2

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py
Copyright 2019 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.constructor import BACKBONES
from src.models.base import BaseModel, FeatureInfo
from src.models.modules.weights_init import trunc_normal_
from src.models.modules.blocks.patch_merging import PatchMerging
from src.models.modules.blocks.patch_embedding import PatchEmbed
from src.models.modules.blocks.swin_block import SwinTransformerBlock
from src.models.modules.bricks.mlp import Mlp
from src.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from src.models.backbones.utils.helpers import build_model_with_cfg


default_cfgs = {
    'swinv2_tiny_window8_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_tiny_patch4_window8_256.pth'
    ),
    'swinv2_tiny_window16_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_tiny_patch4_window16_256.pth'
    ),
    'swinv2_small_window8_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_small_patch4_window8_256.pth'
    ),
    'swinv2_small_window16_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_small_patch4_window16_256.pth'
    ),
    'swinv2_base_window8_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_base_patch4_window8_256.pth'
    ),
    'swinv2_base_window16_256': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_base_patch4_window16_256.pth'
    ),
    'swinv2_base_window12_192_22k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_base_patch4_window12_192_22k.pth'
    ),
    'swinv2_base_window12to16_192to256_22kft1k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'
    ),
    'swinv2_base_window12to24_192to384_22kft1k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'
    ),
    'swinv2_large_window12_192_22k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_large_patch4_window12_192_22k.pth'
    ),
    'swinv2_large_window12to16_192to256_22kft1k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth'
    ),
    'swinv2_large_window12to24_192to384_22kft1k': dict(
        url='https://torchok-hub.s3.eu-west-1.amazonaws.com/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth'
    ),
}


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
            self, dim, input_resolution, depth, num_heads, window_size,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        attention_out = x
        downsample_attention = self.downsample(x)
        return downsample_attention, attention_out

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2(BaseModel):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
        feature_out_idxs (tuple(int)): Output from which stages.
    """

    def __init__(
            self, img_size=224, patch_size=4, in_chans=3,
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.encoder_channels = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches

        self.input_resolutions = []
        for i_layer in range(self.num_layers):
            resolution = (
                self.patch_embed.grid_size[0] // (2 ** i_layer), self.patch_embed.grid_size[1] // (2 ** i_layer)
            )
            self.input_resolutions.append(resolution)

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patch_embed.grid_size[0] // (2 ** i_layer),
                    self.patch_embed.grid_size[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            self.layers.append(layer)

        self.feature_norms = nn.ModuleList([norm_layer(chs) for i, chs in enumerate(self.encoder_channels)])

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = {'absolute_pos_embed'}
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    def __forward_patch_emb(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def __normalize_with_bhwc_reshape(self, x: torch.Tensor, leyer_number: int):
        """Convert swin BLC shape to BHWC.

        Args:
            x: input tensor.
            layer_number: number os SWin layer.
        
        Retruns:
            x: tensor with shape BHWC.
        """
        # B L C
        x = self.feature_norms[leyer_number](x)
        H = self.input_resolutions[leyer_number][0]
        W = self.input_resolutions[leyer_number][1]
        C = self.encoder_channels[leyer_number]
        # B H W C
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x):
        downsample_attn = self.__forward_patch_emb(x)
        features = []
        for i, layer in enumerate(self.layers):
            downsample_attn, attn = layer(downsample_attn)
            feature = self.__normalize_with_bhwc_reshape(attn, i)
            features.append(feature)
            
        return tuple(features)

    def forward(self, x):
        x = self.__forward_patch_emb(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.__normalize_with_bhwc_reshape(x, -1)
        return x

    def get_forward_output_channels(self):
        return self.encoder_channels


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table')]):
            continue  # skip buffers that should not be persistent
        out_dict[k] = v
    return out_dict


def _create_swin_transformer_v2(variant, pretrained=False, **model_kwargs):
    return build_model_with_cfg(SwinTransformerV2, pretrained, default_cfg=default_cfgs[variant], **model_kwargs)


@BACKBONES.register_class
def swinv2_tiny_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_tiny_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_tiny_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_tiny_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_small_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_small_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_small_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_small_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_base_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_base_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=192, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_base_window12_192_22k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs
    )
    return _create_swin_transformer_v2(
        'swinv2_base_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=384, window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs
    )
    return _create_swin_transformer_v2(
        'swinv2_base_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=192, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs
    )
    return _create_swin_transformer_v2('swinv2_large_window12_192_22k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=256, window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs
    )
    return _create_swin_transformer_v2(
        'swinv2_large_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=384, window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs
    )
    return _create_swin_transformer_v2(
        'swinv2_large_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)
