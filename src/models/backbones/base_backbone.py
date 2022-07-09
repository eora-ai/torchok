from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

from src.models.base import BaseModel


class BaseBackbone(BaseModel, ABC):
    """Base model for TorchOk Backbones"""
    def __init__(self,
                 in_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_feature_channels: Optional[Union[List[int], Tuple[int, ...]]] = None):
        """Init BaseBackbone.

        Args:
            in_channels: Number of input channels.
            out_features: Number of output channels - channels after forward method.
            out_feature_channels: Number of output feature channels - channels after forward_features method.
        """
        super().__init__(in_channels, out_channels)
        self._out_feature_channels = out_feature_channels

    @abstractmethod
    def forward_features(self, *args, **kwargs):
        """Forward method for getting backbone feature maps.
           They are mainly used for segmentation and detection tasks.
        """
        pass

    @property
    def out_feature_channels(self) -> List[int]:
        """Number of output feature channels - channels after forward_features method."""
        if self._out_feature_channels is None:
            raise ValueError('TorchOk Backbones must have self._out_feature_channels attribute.')
        if isinstance(self._out_feature_channels, tuple):
            return list(self._out_feature_channels)
        return self._out_feature_channels
