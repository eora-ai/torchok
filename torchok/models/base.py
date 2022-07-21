from abc import ABC, abstractmethod
import torch
import torch.nn as nn

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
    redesign for torcscript.
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


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Backbone, Neck, Pooling and Head.

    This class supports adding feature hooks to some model layers.
    Class has feature info to describe hooks.
    To create hooks, method `self.get_features_info()` must be overridden, and the function `self.create_hooks()` 
    must be called in the end line of initialization method of inherited class!
    Method `self.create_hooks(*args, **kwargs)` called `self.get_features_info(*args, **kwargs)` with 
    the same parameters! 
    Also class contains method get_output_channels which returns forward pass and hooks output channels.

    Example:
        >>> class ModelWithHooks(BaseModel):
        >>>     def __init__(self, input_channel: int = 3):
        >>>         super().__init__()
        >>>         self.out_features = [5, 10, 15]
        >>>         self.conv1 = nn.Conv2d(input_channel, 5, 3)
        >>>         self.conv2 = nn.Conv2d(5, 10, 3)
        >>>         self.conv3 = nn.Conv2d(10, 15, 3)
        >>>         # Call create_hooks after all modules are initialized, with parameters as in get_features_info.
        >>>         self.create_hooks(output_channels=[10, 15], module_names=['conv2', 'conv3'])
        >>>
        >>>     def get_features_info(self, output_channels, module_names):
        >>>         # Overwrite get_features_info.
        >>>         features_info = []
        >>>         for num_channels, module_name in zip(output_channels, module_names):
        >>>             feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=1)
        >>>             features_info.append(feature_info)
        >>>         return features_info
        >>>
        >>>     def get_forward_output_channels(self):
        >>>         return self.out_features[-1]
        >>>
        >>> model = ModelWithHooks()
        >>> print(model.get_output_channels())
        (15, [10, 15])
    """
    def __init__(self):
        """Initialize BaseModel.
        
        Set features_info and feature_hooks as None.
        """
        super().__init__()
        self.__features_info = None
        self.__feature_hooks = None

    def get_features_info(self, *args: Any, **kwargs: Any) -> List[FeatureInfo]:
        """Initialize feature hooks.
        
        It should be overriden in an inherited class if a user is expected to have access to feature hooks 
        of the implemented Module.

        Retruns: Hooks FeatureInfo list.
        """
        return []

    def create_hooks(self, *args: Any, **kwargs: Any) -> Tuple[List[FeatureInfo], FeatureHooks]:
        """Call `self.get_features_info()` method to generate features_info list and then generate feature hooks.
        
        If `self.get_features_info()` isn't overrided in an inherited class then the features_info and hooks 
        will be None and no feature hooks will be created.

        Raises:
            ValueError: If overriden `self.get_features_info()` method return not FeatureInfo list.
        """
        features_info = self.get_features_info(*args, **kwargs)
        # Check if all self._features_info is FeatureInfo types.
        self.__check_features_info_types(features_info)
        self.__features_info = features_info
        self.__feature_hooks = FeatureHooks(features_info, self.named_modules())

    @staticmethod
    def __check_features_info_types(features_info: Any):
        """Check if all self._features_info is FeatureInfo class.

        Args:
            features_info: Hooks feature info list that must be checked.

        Raises:
            ValueError: If features_info is not list, or if any value in features_info is not FeatureInfo class.
        """
        if not isinstance(features_info, list):
            raise ValueError('All features_info must be list.')

        for feature in features_info:
            if type(feature) != FeatureInfo:
                raise ValueError('All features_info must be FeatureInfo class.')

    def forward_stage_features(self, *input: Any) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return model output with hooks features.
        
        Returns:
            last_features: Module forward method output.
            hooks_features: Hooks outputs.
        """
        device = list(*input)[0].device
        last_features = self.forward(*input)
        if self.__feature_hooks is not None:
            hooks_features = self.__feature_hooks.get_features(device)
            hooks_features = list(hooks_features.values())
        else:
            hooks_features = None
        return last_features, hooks_features

    def _get_hooks_output_channels(self) -> List[int]:
        """Generate hooks output channels numbers.
        
        Returns:
            output_hooks_channels: Hooks output channels numbers.
        """
        if self.__features_info is None:
            output_hooks_channels = None
        else:
            output_hooks_channels = [feature.num_channels for feature in self.__features_info]
        return output_hooks_channels

    @abstractmethod
    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Set output channels for Module forward pass.
        
        Returns: Outpus channels.
        """
        pass
    
    def get_output_channels(self) -> Tuple[Union[int, List[int]], List[int]]:
        """Create forward an hooks channels numbers.
        
        Returns:
            forward_channels: Forward pass output channels.
            hooks_channels: Hooks output channels.
        """
        forward_channels = self.get_forward_output_channels()
        hooks_channels = self._get_hooks_output_channels()
        return forward_channels, hooks_channels

    @property
    def features_info(self):
        return self.__features_info

    @property
    def feature_hooks(self):
        return self.__feature_hooks
