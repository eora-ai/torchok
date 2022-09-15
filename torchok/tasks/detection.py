from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, DETECTION_NECKS, HEADS, TASKS
from torchok.constructor.config_structure import Phase
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
        self.num_scales = self._hparams.task.params.get('num_scales', len(self.backbone.out_encoder_channels))

        # NECK
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
        neck_in_channels = self.backbone.out_encoder_channels[-self.num_scales:][::-1]
        self.neck = DETECTION_NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_channels = self.neck.out_channels
        self.bbox_head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone.forward_features(x)[-self.num_scales:]
        x = self.neck(x)
        x = self.bbox_head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, Any]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        features = self.backbone.forward_features(input_data)[-self.num_scales:]
        neck_out = self.neck(features)
        prediction = self.bbox_head(neck_out)
        output = self.bbox_head.format_dict(prediction)
        output['image_shape'] = input_data.shape[-2:]

        if 'bboxes' in batch:
            output['gt_bboxes'] = batch.get('bboxes')
            output['gt_labels'] = batch.get('label')

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(BackboneWrapper(self.backbone), self.neck, self.bbox_head)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete training loop."""
        output = self.forward_with_gt(batch)
        total_loss, tagged_loss_values = self.bbox_head.loss(self.losses, **output)

        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.TRAIN, **output)
        output_dict = {'loss': total_loss}
        output_dict.update(tagged_loss_values)
        return output_dict

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]],
                        batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        total_loss, tagged_loss_values = self.bbox_head.loss(self.losses, **output)

        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.VALID, **output)
        output_dict = {'loss': total_loss}
        output_dict.update(tagged_loss_values)
        return output_dict

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.TEST, **output)

    def predict_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete predict loop."""
        output = self.forward_with_gt(batch)
        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        return output
