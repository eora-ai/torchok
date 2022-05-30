from typing import Dict, List, Tuple, Union

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

        pooling_params = self._hparams.task.params.get('pooling_params', dict())
        pooling_in_features = self.backbone.get_forward_output_channels()
        pooling_name = self._hparams.task.params.get('pooling_name', 'IdentetyPooling')
        self.pooling = POOLINGS.get(pooling_name)(in_features=pooling_in_features, **pooling_params)
        
        head_params = self._hparams.task.params.get('head_params', dict())
        head_in_features = self.pooling.get_forward_output_channels()
        # TODO write IdentetyHead
        head_name = self._hparams.task.params.get('head_name', 'IdentetyHead')
        self.head = HEADS.get(head_name)(in_features=head_in_features, **head_params)

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
        # May be need add config structure
        freeze_backbone = self._hparams.task.params.get('freeze_backbone', False)
        with torch.set_grad_enabled(not freeze_backbone and self.training):
            features = self.backbone(input_data)
        features = self.pooling(features)
        prediction = self.head(features, target)
        output = {'target': target, 'embeddings': features, 'prediction': prediction}
        return output

    def configure_optimizers(self) -> Union[List, Tuple[List, List]]:
        """Define optimizers and LR schedulers."""
        optimizers, schedulers = super().configure_optimizers()

        if schedulers[0] is not None:
            return [optimizers[0]], [schedulers[0]]
        else:
            return [optimizers[0]]

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> torch.Tensor:
        """Complete training loop."""
        output = self.forward_with_gt(batch[0])
        loss = self._losses(**output)
        self._metrics_manager(Phase.TRAIN, **output)
        return {'loss': loss[0], 'tagged_loss_values': loss[1]}

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> torch.Tensor:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        loss = self._losses(**output)
        self._metrics_manager(Phase.VALID, **output)
        return {'loss': loss[0], 'tagged_loss_values': loss[1]}

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        self._metrics_manager(Phase.TEST, **output)
