from abc import ABC, abstractmethod
from typing import List, Union

from src.models.base import BaseModel


class BaseBackbone(BaseModel, ABC):
    """Base model for all TorchOk Backbone."""
    def __init__(self):
        """Initialize BaseModel."""
        super().__init__()

    @abstractmethod
    def forward_features(self, *args, **kwargs):
        """Forward method for getting backbone feature maps.
           They are mainly used for segmentation and detection tasks.
        """
        pass

    @abstractmethod
    def get_forward_feature_channels(self) -> Union[int, List[int]]:
        """Set output channels for forward features pass.

        Returns: Outputs channels.
        """
        pass
