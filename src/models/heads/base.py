from torch import nn, Tensor
from abc import ABC, abstractmethod



class AbstractHead(nn.Module, ABC):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        pass

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
