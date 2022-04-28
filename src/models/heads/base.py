from torch import nn, Tensor
from abc import ABC, abstractmethod



class AbstractHead(nn.Module, ABC):
    """An abstract class for head."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        pass

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features
