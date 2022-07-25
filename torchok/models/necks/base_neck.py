from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

from torchok.models.base import BaseModel


class BaseNeck(BaseModel, ABC):
    """Base model for TorchOk Necks.

    Input tensors for neck obtained from backbone.forward_feature method.
    Default features is:
        [
            backbone input tensor,
            backbone stem out,
            backbone layer_1 out,
            backbone layer_2 out,
            backbone layer_3 out,
            backbone layer_4 out,  
        ].
    So every neck must choice from which feature block do compute.
    """
    def __init__(self,
                 start_block: int = 2,
                 in_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                 out_channels: Optional[Union[int, List[int], Tuple[int, ...]]] = None):
        """Init BaseNeck.

        Args:
            start_block: Input tensors index from which do compute.
            in_channels: Number of input channels.
            out_features: Number of output channels - channels after forward method.
        """
        super().__init__(in_channels, out_channels)
        self._start_block = start_block
