from typing import Dict, Union

import torch
from omegaconf import DictConfig

from src.constructor import BACKBONES, HEADS, POOLINGS, TASKS
from src.tasks.base import BaseTask


@TASKS.register_class
class ClassificationTask(BaseTask):
    """A class for image classification task."""

    def __init__(self, hparams: DictConfig):
        """Init ClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)
        backbones_params = self._hparams.task.params.backbone_params
        self.backbone = BACKBONES.get(self._hparams.task.params.backbone_name)(**backbones_params)
        self._hparams.task.params.pooling_params['in_features'] = self.backbone.get_forward_output_channels()

        pooling_params = self._hparams.task.params.pooling_params
        self.pooling = POOLINGS.get(self._hparams.task.params.pooling_name)(**pooling_params)
        self._hparams.task.params.head_params['in_features'] = self.pooling.get_forward_output_channels()

        head_params = self._hparams.task.params.head_params
        self.head = HEADS.get(self._hparams.task.params.head_name)(**head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        with torch.set_grad_enabled(not self._hparams.task.params.freeze_backbone and self.training):
            features = self.backbone(input_data)
        features = self.pooling(features)
        prediction = self.head(features, target)
        output = {'target': target, 'embeddings': features, 'prediction': prediction}
        return output
