"""TorchOK Swin Transformer V2

Adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
Copyright (c) 2022 Microsoft
Licensed under The MIT License [see LICENSE for details]
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import List, Tuple, Union, Optional

from torchok.constructor import BACKBONES
from torchok.models.backbones.base_backbone import BaseBackbone
from torchok.models.modules.weights_init import trunc_normal_
from torchok.models.modules.blocks.patch_merging import PatchMerging
from torchok.models.modules.blocks.patch_embedding import PatchEmbed
from torchok.models.modules.blocks.swin_block import SwinTransformerBlock
from torchok.models.backbones.utils.helpers import build_model_with_cfg


# The weights was taken from https://github.com/rwightman/pytorch-image-models
# Licensed under The Apache 2.0 License [see LICENSE for details]
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
    """A basic Swin Transformer layer for one stage."""
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: Union[float, List[float]] = 0., norm_layer: nn.Module = nn.LayerNorm,
                 downsample: Optional[nn.Module] = None, use_checkpoint: bool = False, pretrained_window_size: int = 0):
        """Init BasicLayer.
        
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            downsample: Downsample layer at the end of the layer.
            use_checkpoint: Whether to use checkpointing to save memory.
            pretrained_window_size: Local window size in pre-training.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
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
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        attention_out = x
        if self.downsample is not None:
            downsample_attention = self.downsample(x)
        else:
            downsample_attention = x
        return downsample_attention, attention_out

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerV2(BaseBackbone):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` -
        https://arxiv.org/pdf/2103.14030
    """
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_channels: int = 3,
                 embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2], num_heads: List[int] = [3, 6, 12, 24],
                 window_size: int = 7, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm, ape: bool = False, patch_norm: bool = True,
                 use_checkpoint: bool = False, pretrained_window_sizes: List[int] = [0, 0, 0, 0], **kwargs):
        """Init SwinTransformerV2.

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
            use_checkpoint: Whether to use checkpointing to save memory.
            pretrained_window_sizes: Pretrained window sizes of each layer.
        """
        super().__init__(in_channels=in_channels)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.encoder_channels = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self._out_channels = self.encoder_channels[-1]
        self._out_feature_channels = [in_channels] + self.encoder_channels
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.input_resolutions = []
        for i_layer in range(self.num_layers):
            resolution = (
                self.patch_embed.patches_resolution[0] // (2 ** i_layer),
                self.patch_embed.patches_resolution[1] // (2 ** i_layer)
            )
            self.input_resolutions.append(resolution)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
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
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py
    # Copyright 2019 Ross Wightman
    # Licensed under The Apache 2.0 License [see LICENSE for details]
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

    def _normalize_with_bhwc_reshape(self, x: torch.Tensor, leyer_number: int, normalize: bool = True) -> torch.Tensor:
        """Convert SWin BLC shape to BHWC.

        Args:
            x: Input tensor.
            layer_number: Number os SWin layer.
            normalize: If do normalization.

        Retruns:
            x: Tensor with shape BHWC.
        """
        # B L C
        if normalize:
            x = self.feature_norms[leyer_number](x)
        H = self.input_resolutions[leyer_number][0]
        W = self.input_resolutions[leyer_number][1]
        C = self.encoder_channels[leyer_number]
        # B H W C
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        downsample_attn = self._forward_patch_emb(x)
        stem = self._normalize_with_bhwc_reshape(downsample_attn, leyer_number=0, normalize=False)
        features.append(stem)
        for i, layer in enumerate(self.layers):
            downsample_attn, attn = layer(downsample_attn)
            feature = self._normalize_with_bhwc_reshape(attn, leyer_number=i, normalize=True)
            features.append(feature)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_patch_emb(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self._normalize_with_bhwc_reshape(x, -1)
        return x


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
