import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from typing import List, Tuple, Union, Optional


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Neck, Pooling and Head."""
    def __init__(self,
                 in_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None):
        """Init BaseModel.

        Args:
            in_channels: Number of input channels.
            out_features: Number of output channels - channels after forward method.
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
