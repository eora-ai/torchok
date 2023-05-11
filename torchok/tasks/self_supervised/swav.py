from typing import Dict, List
from omegaconf import DictConfig
from torch import Tensor, nn
from lightly.models.modules import SwaVPrototypes

from torchok.constructor import TASKS
from torchok.tasks.classification import ClassificationTask


@TASKS.register_class
class SwaVTask(ClassificationTask):
    """
    Task-agnostic part of the SimCLR v2 approach described in paper
    Big Self-Supervised Models are Strong Semi-Supervised Learners: https://arxiv.org/abs/2006.10029

    This task use `UnsupervisedContrastiveDataset` dataset.
    """

    def __init__(
        self,
        hparams: DictConfig,
        backbone_name: str,
        pooling_name: str = None,
        head_name: str = None,
        neck_name: str = None,
        backbone_params: dict = None,
        neck_params: dict = None,
        pooling_params: dict = None,
        head_params: dict = None,
        inputs: dict = None,
        n_prototypes: int = 3000,
        num_highres_crops: int = 2,
        num_lowres_crops: int = 6
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
            n_prototypes: Number of prototypes used in SwaV method.
        """
        super().__init__(
            hparams=hparams,
            backbone_name=backbone_name,
            pooling_name=pooling_name,
            head_name=head_name,
            neck_name=neck_name,
            backbone_params=backbone_params,
            neck_params=neck_params,
            pooling_params=pooling_params,
            head_params=head_params,
            inputs=inputs,
        )
        self.prototypes = SwaVPrototypes(self.head.out_channels, n_prototypes=n_prototypes)
        self.num_highres_crops = num_highres_crops
        self.num_lowres_crops = num_lowres_crops

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def forward_with_gt(self, batch: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (List[torch.Tensor]):
                    list of tensor of shape `(B, C, H, W)`, each tensor contains cross-batch view of images.

        Returns:
            Dictionary with the following keys and values

            - 'high_resolution_features': list of torch.Tensor of shape `(B, num_features)`,
                representing embeddings per each image view in high resolution.
            - 'low_resolution_features': list of torch.Tensor of shape `(B, num_classes)`,
                representing logits per each image view in low resolution.
        """
        self.prototypes.normalize()
        crops = batch['image']
        multi_crop_features = [self.forward(crop) for crop in crops]
        high_resolution_features = multi_crop_features[:self.num_highres_crops]
        low_resolution_features = multi_crop_features[self.num_highres_crops:]
        output = {"high_resolution_features": high_resolution_features,
                  "low_resolution_features": low_resolution_features}

        return output
