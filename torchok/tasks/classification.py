from typing import Dict, Union

import torch
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, HEADS, POOLINGS, TASKS
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
        backbones_params = self._hparams.task.params.backbone_params
        self.backbone = BACKBONES.get(self._hparams.task.params.backbone_name)(**backbones_params)

        pooling_params = self._hparams.task.params.get('pooling_params', dict())
        pooling_in_features = self.backbone.get_forward_output_channels()
        pooling_name = self._hparams.task.params.get('pooling_name', 'Identity')
        self.pooling = POOLINGS.get(pooling_name)(in_features=pooling_in_features, **pooling_params)
        
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_features = self.pooling.get_forward_output_channels()

        head_name = self._hparams.task.params.get('head_name', 'Identity')
        self.head = HEADS.get(head_name)(in_features=head_in_features, **head_params)

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
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            features = self.backbone(input_data)
        embeddings = self.pooling(features)
        prediction = self.head(embeddings, target)
        output = {'target': target, 'embeddings': embeddings, 'prediction': prediction}
        return output
