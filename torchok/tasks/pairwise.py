from typing import Dict, Union

from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import  TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class PairwiseLearnTask(ClassificationTask):
    """A class for pairwise learning task."""

    def __init__(self, hparams: DictConfig):
        """Init PairwiseLearnTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)

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
