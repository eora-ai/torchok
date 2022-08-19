from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Neck, Pooling and Head."""

    def __init__(self,
                 in_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None):
        """Init BaseModel.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels - channels after forward method.
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward method."""
        pass

    def no_weight_decay(self) -> List[str]:
        """Create module names for which weight decay will not be used.

        Returns: Module names for which weight decay will not be used.
        """
        return list()

    @property
    def in_channels(self) -> Union[int, List[int], Tuple[int, ...]]:
        """Number of input channels."""
        if self._in_channels is None:
            raise ValueError('TorchOk Models must have self._in_channels attribute.')
        return self._in_channels

    @property
    def out_channels(self) -> Union[int, List[int], Tuple[int, ...]]:
        """Number of output channels - channels after forward method."""
        if self._out_channels is None:
            raise ValueError('TorchOk Models must have self._out_channels attribute.')
        return self._out_channels

    def init_weights(self):
        """Initialize model weights"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
