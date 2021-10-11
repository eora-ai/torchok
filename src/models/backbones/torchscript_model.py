from typing import Any

import torch

from .base import BackboneBase
from .utils.registry import register_model


class TorchscriptModel(BackboneBase):

    def __init__(self, path_to_torchscript, out_features, **kwargs):
        super().__init__()
        self.model = torch.jit.load(path_to_torchscript, map_location='cpu')
        self.num_features = out_features
        self.set_neck = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


@register_model
def torchscript_model(**kwargs: Any):
    return TorchscriptModel(**kwargs)
