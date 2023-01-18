from typing import Dict
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class SimCLRTask(ClassificationTask):
    """
    Task-agnostic part of the SimCLR v2 approach described in paper
    Big Self-Supervised Models are Strong Semi-Supervised Learners: https://arxiv.org/abs/2006.10029

    This task use `UnsupervisedContrastiveDataset` dataset.
    """

    def __init__(
        self,
        hparams: DictConfig,
        backbone_name: str,
        pooling_name: str,
        head_name: str,
        neck_name: str = None,
        backbone_params: dict = None,
        neck_params: dict = None,
        pooling_params: dict = None,
        head_params: dict = None,
        inputs: dict = None,
    ):
        """Init SimCLRTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
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
        super().__init__(
            hparams,
            backbone_name,
            pooling_name,
            head_name,
            neck_name,
            backbone_params,
            neck_params,
            pooling_params,
            head_params,
            inputs,
        )

    def forward_with_gt(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image_0` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.
                - `image_1` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.

        Returns:
            Dictionary with the following keys and values

            - 'emb1': torch.Tensor of shape `(B, num_features)`, representing embeddings for batch['image_0'].
            - 'emb2': torch.Tensor of shape `(B, num_features)`, representing embeddings for batch['image_1'].
        """
        x1, x2 = batch["image_0"], batch["image_1"]
        output = dict()
        output["emb1"] = self.forward(x1)
        output["emb2"] = self.forward(x2)

        return output
