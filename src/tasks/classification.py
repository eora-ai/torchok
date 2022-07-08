from typing import Dict, Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from src.tasks.base import BaseTask


@TASKS.register_class
class ClassificationTask(BaseTask):
    """A class for image classification task."""

    def __init__(self, hparams: DictConfig):
        """Init ClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.

        Raises:
            NotImplementedError: if backbone, neck, pooling or head is not implemented.
        """
        super().__init__(hparams)
        # BACKBONE
        backbone_name = self._hparams.task.params.get('backbone_name')
        backbones_params = self._hparams.task.params.get('backbone_params', dict())

        backbone = BACKBONES.get(backbone_name)
        self.backbone = backbone(**backbones_params)

        # NECK
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
        neck_in_features = self.backbone.get_forward_output_channels()

        if neck_name is not None:
            neck = NECKS.get(neck_name)
            self.neck = neck(in_features=neck_in_features, **neck_params)
            pooling_in_features = self.neck.get_forward_output_channels()
        else:
            self.neck = None
            pooling_in_features = neck_in_features

        # POOLING
        pooling_params = self._hparams.task.params.get('pooling_params', dict())
        pooling_name = self._hparams.task.params.get('pooling_name')

        pooling = POOLINGS.get(pooling_name)
        self.pooling = pooling(in_features=pooling_in_features, **pooling_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_features = self.pooling.get_forward_output_channels()

        head = HEADS.get(head_name)
        self.head = head(in_features=head_in_features, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            features = self.backbone(input_data)
        embeddings = self.pooling(features)
        prediction = self.head(embeddings, target)
        output = {'target': target, 'embeddings': embeddings, 'prediction': prediction}
        return output
