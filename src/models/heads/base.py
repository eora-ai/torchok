from typing import List, Union

import torch
from torch import nn
from abc import ABC, abstractmethod
from src.models.base_model import BaseModel, FeatureInfo




class AbstractHead(BaseModel, ABC):
    """An abstract class for head."""

    def __init__(self, in_features, out_features):
        """Init AbstractHead.
        
        Args:
            in_features: Input features.
            out_features: Output features.
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward method."""
        pass

    def init_weights(self) -> None:
        """Weights initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self._out_features

    @property
    def in_features(self) -> int:
        """Input features."""
        return self._in_features

    @property
    def out_features(self) -> int:
        """Output features."""
        return self._out_features
