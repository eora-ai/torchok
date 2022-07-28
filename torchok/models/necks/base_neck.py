from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

from torchok.models.base import BaseModel


class BaseNeck(BaseModel, ABC):
    """Base model for TorchOk Necks.

    Input tensors for neck obtained from backbone.forward_feature method.
    Default features is:
        [
            backbone input tensor,
            backbone layer_1 out,
            backbone layer_2 out,
            ***
        ].
    """
    def __init__(self,
                 in_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None):
        """Init BaseNeck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels - channels after forward method.
        """
        super().__init__(in_channels, out_channels)
