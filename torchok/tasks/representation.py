from typing import Dict, Union

import torch
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class RepresentationLearnTask(ClassificationTask):
    """
    Deep Metric Learning task for pairwise losses.
    """

    def __init__(self, hparams: DictConfig):
        """Init PairwiseLearnTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)

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
        if y.ndim == 1:
            bs = y.shape[0]
            nc = self._hparams.task.params.num_classes
            input_label = torch.zeros(bs, nc, device=y.device)
            y = input_label.scatter_(1, y[:, None], 1)

        intersections = torch.matmul(y, y.transpose(1, 0))
        rel_matrix = (intersections > 0).type(torch.float32)

        return rel_matrix
