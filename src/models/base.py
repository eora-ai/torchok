import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Union


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Backbone, Neck, Pooling and Head."""
    def __init__(self):
        """Initialize BaseModel."""
        super().__init__()

    @abstractmethod
    def get_forward_channels(self) -> Union[int, List[int]]:
        """Set output channels for Module forward pass.

        Returns: Outputs channels.
        """
        pass
    
    @abstractmethod
    def no_weight_decay(self) -> List[str]:
        """Create module names for which weights decay will not be used.

        Returns: Module names for which weights decay will not be used.
        """
        pass
