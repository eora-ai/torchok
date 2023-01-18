from typing import Dict, Union

from torch import Tensor
from omegaconf import DictConfig

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask
from torchok.constructor.config_structure import Phase


@TASKS.register_class
class TripletLearnTask(ClassificationTask):
    """A class for triplet learning task."""

    # ToDo: write documentation for the task parameters
    def __init__(self, hparams: DictConfig, **kwargs):
        """Init TripletLearnTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams, **kwargs)

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        anchor = batch.get('anchor')
        positive = batch.get('positive')
        negative = batch.get('negative')

        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)

        output = {'anchor': anchor, 'positive': positive, 'negative': negative}

        return output

    def validation_step(self, batch: Dict[str, Union[Tensor, int]], batch_idx: int) -> Dict[str, Tensor]:
        """Complete validation loop."""
        output = super(TripletLearnTask, self).forward_with_gt(batch)
        self.metrics_manager.update(Phase.VALID, **output)

        if self._hparams.task.compute_loss_on_valid:
            total_loss, tagged_loss_values = self.losses(**output)
            output_dict = {'loss': total_loss}
            output_dict.update(tagged_loss_values)
        else:
            output_dict = {}

        return output_dict
