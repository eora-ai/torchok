import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constructor import POOLINGS
from src.models.base import BaseModel

from . import Pooling


@POOLINGS.register_class
class PoolingLinear(BaseModel):
    def __init__(self, in_features, out_features, pooling_type='avg', bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.global_pool = Pooling(in_features=in_features, pooling_type=pooling_type)
        self.fc = nn.Linear(self.global_pool.out_features, self.out_features, bias=bias)
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
    
    def get_forward_output_channels(self):
        return self.out_features
    