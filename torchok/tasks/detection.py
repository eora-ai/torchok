from typing import Any, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, DETECTION_NECKS, HEADS, TASKS
from torchok.constructor.config_structure import Phase
from torchok.models.backbones import BackboneWrapper
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SingleStageDetectionTask(BaseTask):
    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            head_name: str,
            neck_name: str,
            num_scales: int = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            head_params: dict = None,
            **kwargs
    ):
        """Init SingleStageDetectionTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            neck_name: name of the head architecture in the DETECTION_NECKS registry.
            head_name: name of the neck architecture in the HEADS registry.
            num_scales: number of feature maps that will be passed from backbone to the neck
                starting from the last one.
                Example: for backbone output `[layer1, layer2, layer3, layer4]` and `num_scales=3`
                neck will get `[layer2, layer3, layer4]`.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, **kwargs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)
        self.num_scales = num_scales or len(self.backbone.out_encoder_channels)

        # NECK
        neck_params = neck_params or dict()
        neck_in_channels = self.backbone.out_encoder_channels[-self.num_scales:][::-1]
        self.neck = DETECTION_NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)

        # HEAD
        head_params = head_params or dict()
        self.bbox_head = HEADS.get(head_name)(in_channels=self.neck.out_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        features = self.backbone.forward_features(x)[-self.num_scales:]
        features = self.neck(features)
        features = self.bbox_head(features)
        output = self.bbox_head.format_dict(features)
        output = self.bbox_head.get_bboxes(**output, image_shape=x.shape[-2:])
        return output

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
        if 'gt_bboxes' in output:
            output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        return output
