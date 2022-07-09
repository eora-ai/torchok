import torch.nn as nn
<<<<<<< HEAD
from abc import ABC, abstractmethod
from torch import Tensor
from typing import List, Union, Optional
=======

from dataclasses import dataclass
from functools import partial
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Generator, Optional, Any
from enum import Enum


class HookType(Enum):
    FORWARD = 'forward'
    FORWARD_PRE = 'forward_pre'


@dataclass
class FeatureInfo:
    """Gather feature hooks params.
    
    Args:
        module_name: Name of hooked module.
        num_channels: Number of image channels.
        stride: Downscale value for get image shape in current module.
        hook_type: Hook type (forward hook or forward pre hook).
    """
    module_name: str
    num_channels: int
    stride: int
    hook_type: HookType = HookType.FORWARD


class FeatureHooks:
    """ Feature Hook Helper.

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torch script.
    """
    def __init__(self, features_info: List[FeatureInfo], named_modules: Generator[Tuple[str, nn.Module], None, None]):
        """Initialize feature hooks.
        
        Args:
            features_info: Hooks features info that must be registered.
            named_modules: Result of self.named_modules() function call.
        """
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for feature_info in features_info:
            hook_name = feature_info.module_name
            module = modules[hook_name]
            hook_fn = partial(self._collect_hook_features, hook_name)
            # register hooks
            if feature_info.hook_type == HookType.FORWARD_PRE:
                module.register_forward_pre_hook(hook_fn)
            else:
                module.register_forward_hook(hook_fn)
           
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_hook_features(self, hook_name: str, *args):
        """Collect hooks features in self._feature_outputs.
        
        Args:
            hook_name: Name of saved hooked module.
        """
        # tensor we want is last argument, output for fwd, input for fwd_pre
        x = args[-1]  
        if isinstance(x, (tuple, list)) and len(x) == 1:
            # unwrap input tuple
            x = x[0]  
            device = x.device
        else:
            device = x[0].device
        self._feature_outputs[device][hook_name] = x

    def get_features(self, device: torch.device) -> Dict[str, torch.tensor]:
        """Return hooks features.

        Args:
            device: Saved hooks device.

        Returns:
            output: Dict of module hook name to its output tensor.
        """
        output = self._feature_outputs[device]
        # clear after reading
        self._feature_outputs[device] = OrderedDict()  
        return output
>>>>>>> origin/dev


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Neck, Pooling and Head."""
    def __init__(self,
                 in_channels: Optional[Union[int, List[int]]] = None,
                 out_channels: Optional[Union[int, List[int]]] = None):
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
        """Create module names for which weights decay will not be used.

        Returns: Module names for which weights decay will not be used.
        """
        return list()

    @property
    def in_channels(self) -> Union[int, List[int]]:
        """Number of input channels."""
        if self._in_channels is None:
            raise ValueError('TorchOk Models must have self._in_channels attribute.')
        return self._in_channels

    @property
    def out_channels(self) -> Union[int, List[int]]:
        """Number of output channels - channels after forward method."""
        if self._out_channels is None:
            raise ValueError('TorchOk Models must have self._out_channels attribute.')
        return self._out_channels
