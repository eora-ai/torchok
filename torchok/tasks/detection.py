from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor.config_structure import Phase
from torchok.constructor import BACKBONES, HEADS, NECKS, TASKS
from torchok.models.backbones import BackboneWrapper
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SingleStageDetectionTask(BaseTask):
    def __init__(self, hparams: DictConfig):
        """Init SingleStageDetectionTask.

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
        neck_in_channels = self.backbone.out_encoder_channels
        self.neck = NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_channels = self.neck.out_channels
        self.head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone.forward_features(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        features = self.backbone.forward_features(input_data)
        neck_out = self.neck(features)
        prediction = self.head(neck_out)
        output = {'prediction': prediction}

        if 'bboxes' in batch:
            output['bboxes'] = batch.get('bboxes')
            output['labels'] = batch.get('labels')

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(BackboneWrapper(self.backbone), self.neck, self.head)


    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete training loop."""
        output = self.forward_with_gt(batch[0])
        total_loss, tagged_loss_values = self.losses(**output)
        self.metrics_manager.update(Phase.TRAIN, **output)
        output_dict = {'loss': total_loss}
        output_dict.update(tagged_loss_values)
        return output_dict

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        self.metrics_manager.update(Phase.VALID, **output)

        # In arcface classification task, if we try to compute loss on test dataset with different number
        # of classes we will crash the train study.
        if self._hparams.task.compute_loss_on_valid:
            total_loss, tagged_loss_values = self.losses(**output)
            output_dict = {'loss': total_loss}
            output_dict.update(tagged_loss_values)
        else:
            output_dict = {}

        return output_dict

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        self.metrics_manager.update(Phase.TEST, **output)

    def predict_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> None:
        """Complete predict loop."""
        output = self.forward_with_gt(batch)
        return output
