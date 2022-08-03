from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


@TASKS.register_class
class ClassificationTask(BaseTask):
    """A class for image classification task."""

    def __init__(self, hparams: DictConfig):
        """Init ClassificationTask.

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
            self.neck = None
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
        if self.neck is not None:
            x = self.neck(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        target = batch.get('target')
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            features = self.backbone(input_data)
        if self.neck is not None:
            features = self.neck(features)
        embeddings = self.pooling(features)
        prediction = self.head(embeddings, target)
        output = {'embeddings': embeddings, 'prediction': prediction}

        if target is not None:
            output['target'] = target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        modules_for_checkpoint = []
        modules_for_checkpoint.append(self.backbone)
        if self.neck is not None:
            modules_for_checkpoint.append(self.neck)
        modules_for_checkpoint.append(self.pooling)
        modules_for_checkpoint.append(self.head)
        return nn.Sequential(*modules_for_checkpoint)
