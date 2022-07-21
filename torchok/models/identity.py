from torch import nn, Tensor

from torchok.models.base import BaseModel


class Identity(BaseModel):
    def __init__(self, in_features):
        super().__init__()
        self.identity = nn.Identity()
        self.out_features = in_features

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        x = self.identity(x)
        return x

    def get_forward_output_channels(self):
        return self.out_features
