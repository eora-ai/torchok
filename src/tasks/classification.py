<<<<<<< HEAD
=======
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig

from src.registry import TASKS, POOLINGS, HEADS, BACKBONES
from src.tasks.base import BaseTask


@TASKS.register_class
class ClassificationTask(BaseTask):
    """A class for image classification task."""

    def __init__(self, hparams: DictConfig):
        """Init ClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)

        self.backbone = BACKBONES.get(self._hparams.backbone_name)(**self._hparams.backbone_params)

        self._hparams.pooling_params['in_features'] = self.backbone.get_forward_output_channels()
        self.pooling = POOLINGS.get(self._hparams.pooling_name)(**self._hparams.pooling_params)

        self._hparams.head_params['in_features'] = self.pooling.get_forward_output_channels()
        self.head = HEADS.get(self._hparams.head_name)(**self._hparams.head_params)

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """Forward through backbone and pooling modules."""
        x = self.backbone(x)
        x = self.pooling(x)
        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method."""
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def configure_optimizers(self) -> Union[List, Tuple[List, List]]:
        """Define optimizers and LR schedulers."""
        modules = [self.pooling, self.head]

        if not self._hparams.freeze_backbone:
            modules.append(self.backbone)
        optimizers, schedulers = super().configure_optimizers()

        if schedulers[0] is not None:
            return [optimizers[0]], [schedulers[0]]
        else:
            return [optimizers[0]]

    def forward_with_gt(self, batch: dict) -> dict:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        with torch.set_grad_enabled(not self._hparams.freeze_backbone and self.training):
            features = self.backbone(input_data)
        features = self.pooling(features)
        prediction = self.head(features, target)
        output = {'target': target, 'embeddings': features, 'prediction': prediction}
        return output

    def training_step(self, batch: dict) -> torch.tensor:
        """The complete training loop."""
        output = self.forward_with_gt(batch)
        loss = self._criterion(**output)
        self._metric_manager.update('train', **output)
        return loss

    def validation_step(self, batch: dict) -> torch.tensor:
        """The complete validation loop."""
        output = self.forward_with_gt(batch)
        loss = self._criterion(**output)
        self._metric_manager.update('valid', **output)
        return loss

    def test_step(self, batch: dict) -> torch.tensor:
        """The complete test loop."""
        output = self.forward_with_gt(batch)
        loss = self._criterion(**output)
        self._metric_manager.update('test', **output)
        return loss
>>>>>>> after review
