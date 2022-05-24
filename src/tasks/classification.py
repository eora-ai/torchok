from typing import Dict, Union

import torch
from omegaconf import DictConfig

from src.constructor.config_structure import Phase
from src.constructor import BACKBONES, HEADS, POOLINGS, TASKS
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

        self.backbone = BACKBONES.get(self._hparams.task.params.backbone_name)(**self._hparams.task.params.backbone_params)

        self._hparams.task.params.pooling_params['in_features'] = self.backbone.get_forward_output_channels()
        self.pooling = POOLINGS.get(self._hparams.task.params.pooling_name)(**self._hparams.task.params.pooling_params)

        self._hparams.task.params.head_params['in_features'] = self.pooling.get_forward_output_channels()
        self.head = HEADS.get(self._hparams.task.params.head_name)(**self._hparams.task.params.head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels."""
        input_data = batch['image']
        target = batch['target']
        with torch.set_grad_enabled(not self._hparams.task.params.freeze_backbone and self.training):
            features = self.backbone(input_data)
        features = self.pooling(features)
        prediction = self.head(features, target)
        output = {'target': target, 'embeddings': features, 'prediction': prediction}
        return output

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizers, schedulers = super().configure_optimizers()

        if schedulers[0] is not None:
            return optimizers[0], schedulers[0]
        else:
            return optimizers[0]

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> Dict:
        """Complete training loop."""
        output = self.forward_with_gt(batch[0])
        loss = self._losses(**output)
        self._metrics_manager.forward(Phase.TRAIN, **output)
        return {'loss': loss[0], 'tagged_loss_values': loss[1]}

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> Dict:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        loss = self._losses(**output)
        self._metrics_manager.forward(Phase.VALID, **output)
        return {'loss': loss[0], 'tagged_loss_values': loss[1]}

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch[0])
        self._metrics_manager.forward(Phase.TEST, **output)
