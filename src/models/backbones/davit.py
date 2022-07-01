"""TorchOK DaViT.

Adapted from https://github.com/dingmyu/davit/blob/main/mmseg/mmseg/models/backbones/davit.py
Licensed under MIT License [see LICENSE for details]
"""
import itertools
from typing import Tuple, Dict, Any, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModel
from src.constructor import BACKBONES
from src.models.modules.blocks.spatial_block import SpatialBlock
from src.models.modules.blocks.channel_block import ChannelBlock
from src.models.modules.blocks.patch_embed import PatchEmbed
from src.models.backbones.utils.helpers import build_model_with_cfg
from src.models.backbones.utils.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        **kwargs
    }


default_cfgs = {
    'davit_t': _cfg(url=''),
    'davit_s': _cfg(url=''),
    'davit_b': _cfg(url=''),
}


cfg_cls = dict(
    davit_t=dict(
        EMBED_DIMS=(96, 192, 384, 768),
        DEPTHS=(1, 1, 3, 1),
        NUM_HEADS=(3, 6, 12, 24),
        WINDOW_SIZE=7,
        MLP_RATIO=4.,
        DROP_PATH_RATE=0.1,
    ),
    davit_s=dict(
        EMBED_DIMS=(96, 192, 384, 768),
        DEPTHS=(1, 1, 9, 1),
        NUM_HEADS=(3, 6, 12, 24),
        WINDOW_SIZE=7,
        MLP_RATIO=4.,
        DROP_PATH_RATE=0.1,
    ),
    davit_b=dict(
        EMBED_DIMS=(128, 256, 512, 1024),
        DEPTHS=(1, 1, 9, 1),
        NUM_HEADS=(4, 8, 16, 32),
        WINDOW_SIZE=7,
        MLP_RATIO=4.,
        DROP_PATH_RATE=0.1,
    )
)


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DaViT(BaseModel):
    """ Dual Attention Transformer"""

    def __init__(self,
                 cfg: Dict[str, Any],
                 in_chans: int = 3):
        """Init DaViT.

        Args:
            cfg: Model config.
            in_chans: Input channels.
        """
        super().__init__()

        architecture = [[index] * item for index, item in enumerate(cfg['DEPTHS'])]
        self.attention_types = ('spatial', 'channel')
        self.architecture = architecture
        self.embed_dims = cfg['EMBED_DIMS']
        self.num_heads = cfg['NUM_HEADS']
        self.num_stages = len(self.embed_dims)
        dpr = [x.item() for x in torch.linspace(0, cfg['DROP_PATH_RATE'], 2 * len(list(itertools.chain(*self.architecture))))]

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=4 if i == 0 else 2,
                       in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                       embed_dim=self.embed_dims[i],
                       overlapped=False)
            for i in range(self.num_stages)])

        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))

            block = nn.ModuleList([
                Sequential(*[
                    ChannelBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=cfg['MLP_RATIO'],
                        qkv_bias=True,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=True,
                        cpe_act=False
                    ) if attention_type == 'channel' else
                    SpatialBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=cfg['MLP_RATIO'],
                        qkv_bias=True,
                        drop_path=dpr[2 * (layer_id + layer_offset_id) + attention_id],
                        norm_layer=nn.LayerNorm,
                        ffn=True,
                        cpe_act=False,
                        window_size=cfg['WINDOW_SIZE'],
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(self.attention_types)]
                ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block)
        self.main_blocks = nn.ModuleList(main_blocks)

        for i_layer in range(self.num_stages):
            layer = nn.LayerNorm(self.embed_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """Forward method."""
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
                features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id], sizes[branch_id])

        outs = []
        for i in range(self.num_stages):
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(features[i])
            H, W = sizes[i]
            out = x_out.view(-1, H, W, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return tuple(outs)

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.embed_dims


def create_davit(variant: str, pretrained: bool = False, **model_kwargs):
    """Create DaViT base model.

    Args:
        variant: Backbone type.
        pretrained: If True the pretrained weights will be loaded.
        model_kwargs: Kwargs for model (for example in_chans).
    """
    return build_model_with_cfg(
        DaViT, pretrained, default_cfg=default_cfgs[variant],
        model_cfg=cfg_cls[variant], **model_kwargs)


@BACKBONES.register_class
def davit_t(pretrained: bool = False, **kwargs):
    """It's constructing a davit_t model."""
    return create_davit('davit_t', pretrained, **kwargs)


@BACKBONES.register_class
def davit_s(pretrained: bool = False, **kwargs):
    """It's constructing a davit_s model."""
    return create_davit('davit_s', pretrained, **kwargs)


@BACKBONES.register_class
def davit_b(pretrained: bool = False, **kwargs):
    """It's constructing a davit_b model."""
    return create_davit('davit_b', pretrained, **kwargs)
