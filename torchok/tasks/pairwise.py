from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


@TASKS.register_class
class PairwiseLearnTask(BaseTask):
    """A class for pairwise learning task."""

    def __init__(self, hparams: DictConfig):
        """Init PairwiseLearnTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)
        # BACKBONE
        backbone_name = self._hparams.task.params.get('backbone_name')
        backbones_params = self._hparams.task.params.get('backbone_params', dict())
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
        neck_in_channels = self.backbone.out_channels

        if neck_name is not None:
            self.neck = NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)
            pooling_in_channels = self.neck.out_channels
        else:
            self.neck = nn.Identity()
            pooling_in_channels = neck_in_channels

        # POOLING
        pooling_params = self._hparams.task.params.get('pooling_params', dict())
        pooling_name = self._hparams.task.params.get('pooling_name')
        self.pooling = POOLINGS.get(pooling_name)(in_channels=pooling_in_channels, **pooling_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_channels = self.pooling.out_channels
        self.head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        anchor = batch.get('anchor')
        positive = batch.get('positive')
        negative = batch.get('negative')

        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)

        output = {'anchor': anchor, 'positive': positive, 'negative': negative}

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(self.backbone, self.neck, self.pooling, self.head)
