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

        self.backbone = BACKBONES.get(self._hparams.backbone_name)(**self._hparams.backbone_params)

        self._hparams.pooling_params['in_features'] = self.backbone.get_forward_output_channels()
        self.pooling = POOLINGS.get(self._hparams.pooling_name)(**self._hparams.pooling_params)

        self._hparams.head_params['in_features'] = self.pooling.get_forward_output_channels()
        self.head = HEADS.get(self._hparams.head_name)(**self._hparams.head_params)

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
        with torch.set_grad_enabled(not self._hparams.freeze_backbone and self.training):
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

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]]) -> torch.Tensor:
        """Complete training loop."""
        output = self.forward_with_gt(batch)
        loss = self._losses(**output)
        self._metrics_manager.update(Phase.TRAIN, **output)
        return loss

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]]) -> torch.Tensor:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        loss = self._losses(**output)
        self._metrics_manager.update(Phase.VALID, **output)
        return loss

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]]) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        self._metrics_manager.update(Phase.TEST, **output)
