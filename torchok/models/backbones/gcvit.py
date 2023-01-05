"""TorchOK Swin Transformer V2
A PyTorch implementation of `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
(Copyright (c) 2022 Microsoft)
and from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py
(Copyright 2022, Ross Wightman)
Licensed under Apache License 2.0 [see LICENSE for details]
"""

from functools import partial
from typing import Optional, Tuple, List, Any, Mapping

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.gcvit import Stem, GlobalContextVitStage
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.layers import get_act_layer, to_ntuple, to_2tuple, get_norm_layer

from torchok.constructor import BACKBONES
from torchok.models.backbones import BaseBackbone


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True, **kwargs
    }


url_base_path = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit'
default_cfgs = {
    'gcvit_xxtiny': _cfg(url=f'{url_base_path}/gcvit_xxtiny_224_nvidia-d1d86009.pth'),
    'gcvit_xtiny': _cfg(url=f'{url_base_path}/gcvit_xtiny_224_nvidia-274b92b7.pth'),
    'gcvit_tiny': _cfg(url=f'{url_base_path}/gcvit_tiny_224_nvidia-ac783954.pth'),
    'gcvit_small': _cfg(url=f'{url_base_path}/gcvit_small_224_nvidia-4e98afa2.pth'),
    'gcvit_base': _cfg(url=f'{url_base_path}/gcvit_base_224_nvidia-f009139b.pth'),
}


class GlobalContextVit(BaseBackbone):
    def __init__(
            self,
            in_channels: int = 3,
            img_size: Tuple[int, int] = 224,
            window_ratio: Tuple[int, ...] = (32, 32, 16, 32),
            window_size: Tuple[int, ...] = None,
            embed_dim: int = 64,
            depths: Tuple[int, ...] = (3, 4, 19, 5),
            num_heads: Tuple[int, ...] = (2, 4, 8, 16),
            mlp_ratio: float = 3.0,
            qkv_bias: bool = True,
            layer_scale: Optional[float] = None,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init='',
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_layer_cl: str = 'layernorm',
            norm_eps: float = 1e-5,
            load_relative_position_bias_table: bool = True
    ):
        num_stages = len(depths)
        super().__init__(in_channels=in_channels, out_channels=int(embed_dim * 2 ** (num_stages - 1)))
        self.encoder_channels = [int(embed_dim * 2 ** i) for i in range(num_stages)]
        self._out_encoder_channels = self.encoder_channels
        self.load_relative_position_bias_table = load_relative_position_bias_table
        act_layer = get_act_layer(act_layer)
        norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        norm_layer_cl = partial(get_norm_layer(norm_layer_cl), eps=norm_eps)

        img_size = to_2tuple(img_size)
        feat_size = tuple(d // 4 for d in img_size)  # stem reduction by 4
        self.drop_rate = drop_rate
        if window_size is not None:
            window_size = to_ntuple(num_stages)(window_size)
        else:
            assert window_ratio is not None
            window_size = tuple([(img_size[0] // r, img_size[1] // r) for r in to_ntuple(num_stages)(window_ratio)])

        self.stem = Stem(
            in_chs=in_channels,
            out_chs=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for i in range(num_stages):
            last_stage = i == num_stages - 1
            stage_scale = 2 ** max(i - 1, 0)
            stages.append(GlobalContextVitStage(
                dim=embed_dim * stage_scale,
                depth=depths[i],
                num_heads=num_heads[i],
                feat_size=(feat_size[0] // stage_scale, feat_size[1] // stage_scale),
                window_size=window_size[i],
                downsample=i != 0,
                stage_norm=last_stage,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                layer_scale=layer_scale,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
            ))
        self.stages = nn.Sequential(*stages)

        self.init_weights(scheme=weight_init)

    def init_weights(self, scheme=''):
        named_apply(partial(self._init_weights, scheme=scheme), self)

    def _init_weights(self, module, name, scheme='vit'):
        # note Conv2d left as default init
        if scheme == 'vit':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
        else:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)'
        )
        return matcher

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not self.load_relative_position_bias_table:
            state_dict = dict(state_dict)
            for k in list(state_dict.keys()):
                if "relative_position_bias_table" in k:
                    state_dict.pop(k)
        return super(GlobalContextVit, self).load_state_dict(state_dict, strict)

    def get_stages(self, stage: int) -> nn.Module:
        """Return modules corresponding the given model stage and all previous stages.
        For example, `0` must stand for model stem. `1` must stand for models stem and
        the first global layer of the model (`layer1` in the resnet), etc.

        Args:
            stage: index of the models stage.
        """
        return nn.ModuleList([self.stem, *self.stages[:stage]])


def _create_gcvit(variant, pretrained=False, **kwargs):
    kwargs_filter = tuple(['num_classes', 'global_pool', 'in_chans'])
    model = build_model_with_cfg(GlobalContextVit, variant, pretrained, pretrained_strict=False,
                                 kwargs_filter=kwargs_filter, **kwargs)
    return model


@BACKBONES.register_class
def gcvit_xxtiny(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_xxtiny', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def gcvit_xtiny(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=(3, 4, 6, 5),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_xtiny', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def gcvit_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(2, 4, 8, 16),
        **kwargs)
    return _create_gcvit('gcvit_tiny', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def gcvit_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(3, 6, 12, 24),
        embed_dim=96,
        mlp_ratio=2,
        layer_scale=1e-5,
        **kwargs)
    return _create_gcvit('gcvit_small', pretrained=pretrained, **model_kwargs)


@BACKBONES.register_class
def gcvit_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        depths=(3, 4, 19, 5),
        num_heads=(4, 8, 16, 32),
        embed_dim=128,
        mlp_ratio=2,
        layer_scale=1e-5,
        **kwargs)
    return _create_gcvit('gcvit_base', pretrained=pretrained, **model_kwargs)
