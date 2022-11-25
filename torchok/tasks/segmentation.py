from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, HEADS, NECKS, TASKS
from torchok.models.backbones import BackboneWrapper
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SegmentationTask(BaseTask):
    # ToDo: write documentation for the task parameters
    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            head_name: str,
            neck_name: str,
            backbone_params: dict = None,
            neck_params: dict = None,
            head_params: dict = None,
            **kwargs
    ):
        """Init SegmentationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams, **kwargs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        neck_params = neck_params or dict()
        neck_params['in_channels'] = self.backbone.out_encoder_channels
        self.neck = NECKS.get(neck_name)(**neck_params)

        # HEAD
        head_params = head_params or dict()
        head_params['in_channels'] = self.neck.out_channels
        self.head = HEADS.get(head_name)(**head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone.forward_features(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        target = batch.get('target')
        features = self.backbone.forward_features(input_data)
        neck_out = self.neck(features)
        prediction = self.head(neck_out)
        output = {'prediction': prediction}

        if target is not None:
            output['target'] = target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(BackboneWrapper(self.backbone), self.neck, self.head)
