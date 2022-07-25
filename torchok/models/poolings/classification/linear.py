import torch
import torch.nn as nn

from torchok.constructor import POOLINGS
from torchok.models.base import BaseModel
from torchok.models.poolings.classification import Pooling


@POOLINGS.register_class
class PoolingLinear(BaseModel):
    def __init__(self, in_channels, out_channels, pooling_type='avg', bias=True):
        super().__init__(in_channels, out_channels)

        self.global_pool = Pooling(in_channels=in_channels, pooling_type=pooling_type)
        self.fc = nn.Linear(self.global_pool._out_channels, out_channels, bias=bias)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
