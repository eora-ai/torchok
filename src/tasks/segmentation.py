from typing import Dict, Union

import torch
from omegaconf import DictConfig

from src.constructor import BACKBONES, HEADS, NECKS, TASKS
from src.tasks.base import BaseTask


@TASKS.register_class
class SegmentationTask(BaseTask):
    def __init__(self, hparams: DictConfig):
        """Init SegmentationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)

        self._start_block_number = hparams.task.get('begin_block_number', 2)

        # BACKBONE
        backbone_name = self._hparams.task.params.get('backbone_name')
        backbones_params = self._hparams.task.params.get('backbone_params', dict())
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
        neck_in_channels = self.backbone.out_feature_channels[self._start_block_number:]
        self.neck = NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_channels = self.neck.out_channels
        self.head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone.forward_features(x)
        x = self.neck(x, self._start_block_number)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            features = self.backbone.forward_features(input_data)
        neck_out = self.neck(features, self._start_block_number)
        prediction = self.head(neck_out)
        output = {'target': target, 'prediction': prediction}
        return output
