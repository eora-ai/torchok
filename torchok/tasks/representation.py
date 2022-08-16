from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


@TASKS.register_class
class RepresentationLearnTask(BaseTask):
    """
    Deep Metric Learning task for pairwise losses.
    """

    def __init__(self, hparams: DictConfig):
        """Init PairwiseLearnTask.

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
            self.neck = nn.Identity()
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pooling(x)
        x = self.head(x)

        return x

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        target = batch.get('target')

        embedding = self.forward(input_data)

        output = {'emb1': embedding, 'emb2': embedding}

        if target is not None:
            output['R'] = self.calc_relevance_matrix(target)
            output['target'] = target

        return output

    def calc_relevance_matrix(self, y: Tensor) -> Tensor:
        """
        Calculates binary relevance matrix given multi-label matrice y
        Args:
            y: Multi-label matrix of shape (N, L) representing labels for N samples, where L - number of classes.
                Values are either 0 or 1, where y1[i, k] = 1 indicate that i-th sample belongs to k-th class

        Returns:
            Binary relevance matrix R of shape (N, M) where R[i, j] = 1 means that
                samples i and j are relevant to each other, dtype=float32
        """
        y = y.float()

        if y.ndim == 1:
            bs = y.shape[0]
            nc = self._hparams.task.params.num_classes
            input_label = torch.zeros(bs, nc, device=y.device)
            y = input_label.scatter_(1, y[:, None], 1)

        intersections = torch.matmul(y, y.transpose(1, 0))
        rel_matrix = (intersections > 0).type(torch.float32)

        return rel_matrix

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(self.backbone, self.neck, self.pooling, self.head)
