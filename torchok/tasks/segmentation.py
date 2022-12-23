from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, HEADS, NECKS, TASKS
from torchok.models.backbones import BackboneWrapper
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SegmentationTask(BaseTask):
    """A class for image segmentation task."""

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
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            neck_name: name of the head architecture in the DETECTION_NECKS registry.
            head_name: name of the neck architecture in the HEADS registry.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, **kwargs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        neck_params = neck_params or dict()
        self.neck = NECKS.get(neck_name)(in_channels=self.backbone.out_encoder_channels, **neck_params)

        # HEAD
        head_params = head_params or dict()
        self.head = HEADS.get(head_name)(in_channels=self.neck.out_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            x: torch.Tensor of shape [B, C, H, W]. Batch of input images

        Returns:
            torch.Tensor of shape [B, num_classes, H, W], representing logits masks per each image.
        """
        x = self.backbone.forward_features(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (torch.Tensor):
                    tensor of shape (B, C, H, W), representing input images.
                - `target` (torch.Tensor):
                    tensor of shape [B, H, W], target class or labels masks per each image.

        Returns:
            Dictionary with the following keys and values

            - 'prediction': torch.Tensor of shape [B, num_classes], representing logits masks per each image.
            - 'target': torch.Tensor of shape [B, H, W], target class or labels masks per each image. May absent.
        """
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
        """Method for model representation as sequential of modules (required for checkpointing)."""
        return nn.Sequential(BackboneWrapper(self.backbone), self.neck, self.head)
