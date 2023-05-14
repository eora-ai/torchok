from typing import Dict, List, Any
from collections import namedtuple

import torch
from omegaconf import DictConfig
from torch import nn

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


@TASKS.register_class
class MultiHeadClassificationTask(BaseTask):
    """A class for multi-head classification task."""

    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            heads: List[Dict[str, Any]],
            neck_name: str = None,
            pooling_name: str = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            pooling_params: dict = None,
            inputs: dict = None
    ):
        """Init MultiHeadClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            pooling_name: name of the backbone architecture in the POOLINGS registry.
            heads: list of dicts containing information about model heads. Format of the dict

                - `type` (str): class of the head model.
                - `name` (str): name of the head.
                - `target` (str): name of the target from dataset.
                - `params` (dict): dict containing parameters of the head.

            neck_name: if present, name of the head architecture in the NECKS registry. Otherwise, model will be created
                without neck.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            pooling_params: parameters for neck constructor. `in_channels` will be set automatically based on neck or
                backbone if neck is absent.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, inputs)
        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        if neck_name is None:
            self.neck = nn.Identity()
            pooling_in_channels = self.backbone.out_channels
        else:
            neck_params = neck_params or dict()
            self.neck = NECKS.get(neck_name)(in_channels=self.backbone.out_encoder_channels, **neck_params)
            pooling_in_channels = self.neck.out_channels

        # POOLING
        if pooling_name is None:
            self.pooling = nn.Identity()
            head_in_channels = self.backbone.out_channels
        else:
            pooling_params = pooling_params or dict()
            self.pooling = POOLINGS.get(pooling_name)(in_channels=pooling_in_channels, **pooling_params)
            head_in_channels = self.pooling.out_channels

        # HEADS
        self.heads = nn.ModuleDict()
        self.target_mapping = {}
        for head in heads:
            head_type = HEADS.get(head['type'])
            head_name = head['name']
            target = head['target']

            self.heads[head_name] = head_type(in_channels=head_in_channels, **head['params'])
            self.target_mapping[head_name] = target

        self.head_tuple = namedtuple('HeadOutput', list(self.target_mapping.keys()))

    def forward(self, x: torch.Tensor) -> namedtuple:
        """Forward method.

        Args:
            x: torch.Tensor of shape `(B, C, H, W)`. Batch of input images.

        Returns:
            Namedtuple with string keys representing head name
            and torch.Tensor values representing output of corresponding head.
        """
        features = self.backbone(x)
        features = self.pooling(features)
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(features)
        return self.head_tuple(**head_outputs)

    def forward_with_gt(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.
                - `target_*` (torch.Tensor):
                    tensor of shape `(B)`, target class or labels per each image for *-named head.
                - `condition_*` (Optional[torch.Tensor]):
                    boolean tensor of shape `(B)`, condition to select corresponding backbone
                    output features for the *-named head.

        Returns:
            Dictionary with the following keys and values

            - 'embeddings': torch.Tensor of shape `(B, num_features)`, representing embeddings per each image.
            - 'prediction_*':
                torch.Tensor of shape `(b, num_classes)`, representing logits per each image for *-named head.
                `b` is less or equal than `B` and depends on `condition_*`.
            - 'target_*':
                torch.Tensor of shape `(b)`, target class or labels per each image. May absent.
                `b` is less or equal than `B` and depends on `condition_*`.
        """

        features = self.backbone(batch['image'])
        features = self.pooling(features)
        output = {'embeddings': features}
        for head_name, head in self.heads.items():
            target_name = self.target_mapping[head_name]
            head_target = batch[f'target_{target_name}']
            condition = batch.get(f'condition_{target_name}', None)

            if condition is not None:
                head_target = head_target[condition]
                out = head(features[condition], head_target)
            else:
                out = head(features, head_target)

            output[f'prediction_{head_name}'] = out
            output[f'target_{target_name}'] = head_target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for onnx checkpointing)."""
        raise NotImplementedError()
