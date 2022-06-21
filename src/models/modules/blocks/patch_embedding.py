""" TorchOK DropBlock, DropPath modules.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
from torch import nn as nn

from src.models.modules.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size[0]:
            raise ValueError(f"PatchEmbed module. Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        if W != self.img_size[1]:
            raise ValueError(f"PatchEmbed module. Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x