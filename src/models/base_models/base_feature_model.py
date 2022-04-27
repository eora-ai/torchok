import torch.nn as nn

from abc import ABC, abstractmethod
from typing import List


class BaseChannelsModel(nn.Module, ABC):
    """Interface for using input and output channels.

    Args:
        input_channels: Channels of input tensors.
    """
    def __init__(self, input_channels: List[int]):
        super().__init__()
        self._input_channels = input_channels


    @abstractmethod
    def get_output_channels(self) -> List[int]:
        """Method for obtain output channels of current model."""
        pass

    @property
    def input_channels(self):
        return self._input_channels
