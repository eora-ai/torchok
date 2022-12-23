from typing import Dict, Union

import torch
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class PairwiseLearnTask(ClassificationTask):
    """
    Deep Metric Learning task for pairwise losses.

    This task use `ClassificationDataset` in multilabel mode, to detect similar images.
    An example config for this task is `torchok/examples/configs/pairwise_sop.yaml`.
    """

    def __init__(
            self,
            hparams: DictConfig,
            num_classes: int,
            backbone_name: str,
            pooling_name: str,
            head_name: str = None,
            neck_name: str = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            pooling_params: dict = None,
            head_params: dict = None,
            inputs: dict = None
    ):
        """Init PairwiseLearnTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            num_classes: number of all classes in train multilabel dataset.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            pooling_name: name of the backbone architecture in the POOLINGS registry.
            head_name: name of the neck architecture in the HEADS registry.
            neck_name: if present, name of the head architecture in the NECKS registry. Otherwise, model will be created
                without neck.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            pooling_params: parameters for neck constructor. `in_channels` will be set automatically based on neck or
                backbone if neck is absent.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, backbone_name, pooling_name, head_name, neck_name,
                         backbone_params, neck_params, pooling_params, head_params, inputs)
        self.num_classes = num_classes

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (torch.Tensor):
                    tensor of shape (B, C, H, W), representing input images.
                - `target` (torch.Tensor):
                    tensor of shape (B), target class or labels per each image.

        Returns:
            Dictionary with the following keys and values

            - 'emb1': model forward method output.
            - 'emb2': model forward method output same as `emb1`.
            - 'target': target value batch if key `target` in batch, otherwise this key not in output dictionary.
            - 'R': calculated relevance matrix if key `target` in batch, otherwise this key not in output dictionary.
        """
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
        Calculates binary relevance matrix given multi-label matrix y.

        Args:
            y: Multi-label matrix of shape (N, L) representing labels for N samples, where L - number of classes.
            Values are either 0 or 1, where y1[i, k] = 1 indicate that i-th sample belongs to k-th class.

        Returns:
            Binary relevance matrix R of shape (N, M) where R[i, j] = 1 means that samples i and j are relevant
            to each other, dtype=float32.
        """
        if y.ndim == 1:
            bs = y.shape[0]
            input_label = torch.zeros(bs, self.num_classes, device=y.device)
            y = input_label.scatter_(1, y[:, None], 1)

        intersections = torch.matmul(y, y.transpose(1, 0))
        rel_matrix = torch.where(intersections > 0, 1., 0.)

        return rel_matrix
