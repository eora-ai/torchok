from typing import Dict, Union, Tuple

import torch
from torch import Tensor
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
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
        # BACKBONE
        backbone_name = self._hparams.task.params.get('backbone_name')
        backbones_params = self._hparams.task.params.get('backbone_params', dict())
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        neck_name = self._hparams.task.params.get('neck_name')
        neck_params = self._hparams.task.params.get('neck_params', dict())
        neck_in_channels = self.backbone.out_channels

        if neck_name is not None:
            self.neck = NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)
            pooling_in_channels = self.neck.out_channels
        else:
            self.neck = None
            pooling_in_channels = neck_in_channels

        # POOLING
        pooling_params = self._hparams.task.params.get('pooling_params', dict())
        pooling_name = self._hparams.task.params.get('pooling_name')
        self.pooling = POOLINGS.get(pooling_name)(in_channels=pooling_in_channels, **pooling_params)

        # HEAD
        head_name = self._hparams.task.params.get('head_name')
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_channels = self.pooling.out_channels
        self.head = HEADS.get(head_name)(in_channels=head_in_channels, **head_params)

        # check forward outputs is allowed or no.
        self.forward_allowed_outputs = {'embeddings', 'predictions'}
        self.forward_outputs = set(self._hparams.task.params.get('forward_outputs', ['predictions']))

        if len(self.forward_allowed_outputs.intersection(self.forward_outputs)) == 0:
            raise ValueError(f'forward_outputs = {self.forward_outputs} is not allowed. '
                             f'Available names: {self.forward_allowed_outputs}')

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward method."""
        outputs = []

        features = self.backbone(x)

        if self.neck is not None:
            features = self.neck(features)

        embeddings = self.pooling(features)

        if 'embeddings' in self.forward_outputs:
            outputs.append(embeddings)

        if 'predictions' in self.forward_outputs:
            predictions = self.head(embeddings)
            outputs.append(predictions)

        outputs = (*outputs, )
        return outputs

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        features = self.backbone(input_data)
        if self.neck is not None:
            features = self.neck(features)
        embeddings = self.pooling(features)
        prediction = self.head(embeddings, target)
        output = {'target': target, 'embeddings': embeddings, 'prediction': prediction}
        return output
