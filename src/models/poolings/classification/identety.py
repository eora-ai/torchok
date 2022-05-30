import torch
from torch import nn, Tensor

from src.models.base_model import BaseModel
from src.constructor import POOLINGS


@POOLINGS.register_class
class IdentityPooling(BaseModel):
    def __init__(self, in_features):
        super().__init__()
        self.identity = nn.Identity()
        self.out_features = in_features

    def forward(self, x: Tensor) -> Tensor:
        x = self.identity(x)
        return x

    def get_forward_output_channels(self):
        return self.out_features
