"""TorchOK Swin Transformer V2
A PyTorch implementation of `Swin Transformer V2: Scaling Up Capacity and Resolution`
- https://arxiv.org/abs/2111.09883

Adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
(Copyright (c) 2022 Microsoft)
and from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py
(Copyright 2022, Ross Wightman)
Licensed under Apache License 2.0 [see LICENSE for details]
"""

from typing import List, Mapping, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer_v2 import BasicLayer as SwinBasicLayer, checkpoint_filter_fn, PatchEmbed, PatchMerging

from torchok.constructor import BACKBONES
from torchok.models.backbones import BaseBackbone


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


url_base_path = 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/'
default_cfgs = {
    'swinv2_tiny_window8_256': _cfg(url=f'{url_base_path}/swinv2_tiny_patch4_window8_256.pth',
                                    input_size=(3, 256, 256)),
    'swinv2_tiny_window16_256': _cfg(url=f'{url_base_path}/swinv2_tiny_patch4_window16_256.pth',
                                     input_size=(3, 256, 256)),
    'swinv2_small_window8_256': _cfg(url=f'{url_base_path}/swinv2_small_patch4_window8_256.pth',
                                     input_size=(3, 256, 256)),
    'swinv2_small_window16_256': _cfg(url=f'{url_base_path}/swinv2_small_patch4_window16_256.pth',
                                      input_size=(3, 256, 256)),
    'swinv2_base_window8_256': _cfg(url=f'{url_base_path}/swinv2_base_patch4_window8_256.pth',
                                    input_size=(3, 256, 256)),
    'swinv2_base_window16_256': _cfg(url=f'{url_base_path}/swinv2_base_patch4_window16_256.pth',
                                     input_size=(3, 256, 256)),

    'swinv2_base_window12_192_22k': _cfg(url=f'{url_base_path}/swinv2_base_patch4_window12_192_22k.pth',
                                         num_classes=21841, input_size=(3, 192, 192)),
    'swinv2_base_window12to16_192to256_22kft1k': _cfg(
        url=f'{url_base_path}/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
        input_size=(3, 256, 256)),
    'swinv2_base_window12to24_192to384_22kft1k': _cfg(
        url=f'{url_base_path}/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'swinv2_large_window12_192_22k': _cfg(url=f'{url_base_path}/swinv2_large_patch4_window12_192_22k.pth',
                                          num_classes=21841, input_size=(3, 192, 192)),
    'swinv2_large_window12to16_192to256_22kft1k': _cfg(
        url=f'{url_base_path}/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth',
        input_size=(3, 256, 256)),
    'swinv2_large_window12to24_192to384_22kft1k': _cfg(
        url=f'{url_base_path}/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class BasicLayer(SwinBasicLayer):
    """ A basic Swin Transformer layer for one stage adapted for intermediate feature extraction.
    """

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return self.downsample(x), x


class SwinTransformerV2(BaseBackbone):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of input image channels.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            ape: If True, add absolute position embedding to the patch embedding.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            load_attn_mask: If False drop `attn_mask` in layers when load pretrained weights.
    """

    def __init__(
            self, img_size: int = 224, patch_size: int = 4, in_channels: int = 3,
            embed_dim: int = 96, depths: List[int] = (2, 2, 6, 2), num_heads: List[int] = (3, 6, 12, 24),
            window_size: int = 7, mlp_ratio: float = 4., qkv_bias: bool = True,
            drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm, ape: bool = False, patch_norm: bool = True,
            pretrained_window_sizes: List[int] = (0, 0, 0, 0), load_attn_mask: bool = True):
        super().__init__(in_channels=in_channels)
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.encoder_channels = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self._out_channels = self.encoder_channels[-1]
        self._out_encoder_channels = self.encoder_channels
        self.load_attn_mask = load_attn_mask

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.input_resolutions = []
        for i_layer in range(self.num_layers):
            resolution = (
                self.patch_embed.grid_size[0] // (2 ** i_layer),
                self.patch_embed.grid_size[1] // (2 ** i_layer)
            )
            self.input_resolutions.append(resolution)

        self.patches_resolution = self.patch_embed.grid_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
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

        self.init_weights()

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
        for bly in self.layers:
            bly._init_respostnorm()

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        """Create modules names for which weights decay will not be use. See BaseBackbone for more information.

        Returns:
            nod: Module names for which weights decay will not be use.
        """
        nod = ['absolute_pos_embed']
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.append(n)
        return nod

    def _forward_patch_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch embedding part.

        Args:
            x: Input tensor.

        Returns:
            x: Patch embeddings with dropout - self.pos_drop.
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def _normalize_with_bhwc_reshape(self, x: torch.Tensor, layer_number: int, normalize: bool = True) -> torch.Tensor:
        """Convert SWin BLC shape to BCHW.

        Args:
            x: Input tensor.
            layer_number: Number os SWin layer.
            normalize: If `True` do normalization.

        Returns:
            x: Tensor with shape BCHW.
        """
        # B L C
        if normalize:
            x = self.feature_norms[layer_number](x)
        H = self.input_resolutions[layer_number][0]
        W = self.input_resolutions[layer_number][1]
        C = self.encoder_channels[layer_number]
        # B C H W
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        downsample_attn = self._forward_patch_emb(x)

        for i, layer in enumerate(self.layers):
            downsample_attn, attn = layer(downsample_attn)
            feature = self._normalize_with_bhwc_reshape(attn, layer_number=i, normalize=True)
            features.append(feature)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_patch_emb(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self._normalize_with_bhwc_reshape(x, -1)
        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not self.load_attn_mask:
            state_dict = dict(state_dict)
            for k in list(state_dict.keys()):
                if "attn_mask" in k:
                    state_dict.pop(k)
        return super(SwinTransformerV2, self).load_state_dict(state_dict, strict)

    def get_stages(self, stage: int) -> nn.Module:
        """Return modules corresponding the given model stage and all previous stages.
        For example, `0` must stand for model stem. `1` must stand for models stem and
        the first global layer of the model (`layer1` in the resnet), etc.

        Args:
            stage: index of the models stage.
        """
        output = [self.patch_embed, self.pos_drop]
        return nn.ModuleList(output + list(self.layers[:stage]))


def _create_swin_transformer_v2(variant, pretrained=False, **kwargs):
    kwargs_filter = ('num_classes', 'global_pool', 'in_chans')
    model = build_model_with_cfg(SwinTransformerV2, variant, pretrained, pretrained_strict=False,
                                 kwargs_filter=kwargs_filter, pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model


@BACKBONES.register_class
def swinv2_tiny_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_tiny_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_tiny_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_tiny_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_small_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_small_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_small_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_small_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window16_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window8_256', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window12_192_22k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_base_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_base_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_base_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2('swinv2_large_window12_192_22k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_large_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def swinv2_large_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_large_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)
