import math
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_
from torch import Tensor

from torchok.constructor import BACKBONES, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SimMIMTask(BaseTask):
    """A class for image classification task."""

    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            pooling_name: str,
            backbone_params: dict = None,
            pooling_params: dict = None,
            encoder_stride: int = 32,
            mask_patch_size: int = 32,
            mask_ratio: float = 0.6,
            inputs: dict = None
    ):
        """Init ClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            pooling_name: name of the backbone architecture in the POOLINGS registry.
            backbone_params: parameters for backbone constructor.
            pooling_params: parameters for neck constructor. `in_channels` will be set automatically based on neck or
                backbone if neck is absent.
            encoder_stride:
            mask_patch_size:
            mask_ratio:
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, inputs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        self.backbone_stride = encoder_stride
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        # POOLING
        pooling_params = pooling_params or dict()
        self.pooling = POOLINGS.get(pooling_name)(in_channels=self.backbone.out_channels, **pooling_params)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.out_channels,
                out_channels=self.backbone_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.backbone_stride),
        )

        self.input_size = self.backbone.img_size[0]
        self.patch_size = self.backbone.patch_embed.patch_size[0]

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(math.ceil(self.token_count * self.mask_ratio))

        self.hook_handle = self.backbone.patch_embed.register_forward_hook(self.hook)
        self.backbone.patch_embed.mask_token = nn.Parameter(torch.zeros(1, 1, self.backbone.embed_dim))
        trunc_normal_(self.backbone.patch_embed.mask_token, mean=0., std=.02)

    def hook(self, module, inp, out):
        B, L, _ = out.shape
        mask_tokens = module.mask_token.expand(B, L, -1)
        w = self.mask.type_as(mask_tokens).flatten()[None, :, None]
        out.mul_(1. - w)
        out.add_(mask_tokens * w)

    def forward(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Forward method."""
        mask = self._prepare_mask()
        z = self.backbone(x)
        embeddings = self.pooling(z)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)
        mask = mask[None, None].contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / ((mask.sum() + 1e-5) * self.backbone.in_channels * x.size(0))
        return loss, embeddings

    def _prepare_mask(self):
        mask_idx = torch.randperm(self.token_count)[:self.mask_count]
        mask = torch.zeros(self.token_count, dtype=torch.bool, device=self.device, requires_grad=False)
        mask[mask_idx] = True

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat_interleave(self.scale, dim=0).repeat_interleave(self.scale, dim=1)
        self.register_buffer('mask', mask)
        return mask

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        target = batch.get('target')

        loss, embeddings = self.forward(input_data)
        output = {'embeddings': embeddings, 'loss': loss}

        if target is not None:
            output['target'] = target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(self.backbone, self.pooling)
