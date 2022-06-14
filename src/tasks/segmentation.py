from typing import Dict, Union

import torch
from omegaconf import DictConfig

from src.constructor import BACKBONES, HEADS, NECKS, TASKS
from src.tasks.base import BaseTask


@TASKS.register_class
class SegmentationTask(BaseTask):
    """A class for segmentation task."""

    def __init__(self, hparams: DictConfig):
        """Init SegmentationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)
        backbones_params = self._hparams.task.params.backbone_params
        self.backbone = BACKBONES.get(self._hparams.task.params.backbone_name)(**backbones_params)

        neck_in_features = self.backbone.get_forward_output_channels()
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
    
        if neck_name is not None:
            self.neck = NECKS.get(neck_name)(in_features=neck_in_features, **neck_params)
            head_in_features = self.neck.get_forward_output_channels()
        else:
            self.neck = None
            head_in_features = neck_in_features

        head_params = self._hparams.task.params.head_params 
        self.head = HEADS.get(self._hparams.task.params.head_name)(in_features=head_in_features, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            backbone_features = self.backbone(input_data)
        if self.neck is not None:
            neck_features = self.neck(backbone_features)
        prediction = self.head(neck_features)
        output = {'target': target, 'prediction': prediction}
        return output
