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
    """
    module_name: str
    num_channels: int
    stride: int
    hook_type: HookType = HookType.FORWARD


@dataclass
class Hook:
    """Gather hook params.
    
    Args:
        module_name: Hooked module name.
        hook_type: Hook type (forward hook or forward pre hook).
    """
    module_name: str
    hook_type: HookType


class FeatureHooks:
    """ Feature Hook Helper.

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torcscript.
    """

    def __init__(self, hooks: List[Hook], named_modules: Generator[Tuple[str, nn.Module], None, None]):
        """Initialize feature hooks.
        
        Args:
            hooks: Hooks to be registered.
            named_modules: Result of self.named_modules() function call.
        """
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for i, hook in enumerate(hooks):
            hook_name = hook.module_name
            module = modules[hook_name]
            hook_fn = partial(self._collect_hook_features, hook_name)
            # register hooks
            if hook.hook_type == HookType.FORWARD_PRE:
                module.register_forward_pre_hook(hook_fn)
            else:
                module.register_forward_hook(hook_fn)
           
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_hook_features(self, hook_name: str, *args):
        """Collect hooks features in self._hook_features.
        
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
            output: Dict of module hook name two its output tensor.
        """
        output = self._feature_outputs[device]
        # clear after reading
        self._feature_outputs[device] = OrderedDict()  
        return output


class BaseModel(nn.Module, ABC):
    """Base model for all TorchOk Models - Backbone, Neck, Pooling and Head.

    This class supports adding feature hooks to some model layers.
    Class has feature info to describe hooks.
    To create hooks, method _get_features_info(self, *args, **kwargs) must be overriden, 
    and the function _create_hooks(self, *args, **kwargs) must be called in the end line 
    of initialization method of inherited class! 
    Method _create_hooks(self, *args, **kwargs) called _get_features_info(self, *args, **kwargs) 
    with the same parameters!
    Also class contains method get_output_channels which returns forward pass and hooks output channels.

    Example:
        >>> class ModelWithHooks(BaseModel):
        >>>     def __init__(self, input_channel: int):
        >>>         super().__init__()
        >>>         out_features = [5, 10, 15]
        >>>         self.conv1 = nn.Conv2d(input_channel, 5, 3)
        >>>         self.conv2 = nn.Conv2d(5, 10, 3)
        >>>         self.conv3 = nn.Conv2d(10, 15, 3)
        >>>         # Call _create_hooks after all modules is initialize, with parameters as in _get_features_info.
        >>>         self._create_hooks(output_channels=[10, 15], module_names=['conv2', 'conv3'])

        >>>     def _get_features_info(self, output_channels, module_names):
        >>>         # Overwrite get_features_info.
        >>>         features_info = []
        >>>         for num_channels, module_name in zip(output_channels, module_names):
        >>>             feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=1)
        >>>             features_info.append(feature_info)
        >>>         return features_info
    """
    def __init__(self):
        """Initialize BaseModel.
        
        Set features_info and feature_hooks as None.
        """
        super().__init__()
        self._features_info = None
        self._feature_hooks = None

    def _get_features_info(self) -> List[FeatureInfo]:
        """Method for initialize Hooks.
        
        It should be overriden in an inherited class if a user is expected to have access to feature hooks 
        of the implemented Module.

        Retruns: Hooks FeatureInfo list.
        """
        return None

    def _create_hooks(self) -> Tuple[List[FeatureInfo], FeatureHooks]:
        """Call self._get_features_info() method to generate features_info list and then generate feature hooks.
        
        If self._get_features_info() isn't overrided in an inherited class then the features_info and hooks 
        will be None and no feature hooks will be created.

        Args:
            args: Any list which will be passed to self._get_features_info function.
            kwargs: Any dict which will be passed to self._get_features_info function.

        Returns:
            features_info: Hooks feature info list.
            feature_hooks: Generated hooks.

        Raises:
            ValueError: If overriden self._get_features_info() method return not FeatureInfo list.
        """
        self._features_info = self._get_features_info()
        # Check if all self._features_info is FeatureInfo types.
        if self._features_info is not None:
            self.__check_features_info_types()
            hooks = [
                Hook(module_name=feature.module_name, hook_type=feature.hook_type) 
                for feature in self._features_info
                ]
            self._feature_hooks = FeatureHooks(hooks, self.named_modules())

    def __check_features_info_types(self):
        """Check if all self._features_info is FeatureInfo class.

        Raises:
            ValueError: If any value in features_info is not FeatureInfo class.
        """
        for feature in self._features_info:
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
        if self._feature_hooks is not None:
            hooks_features = self._feature_hooks.get_features(device)
            hooks_features = list(hooks_features.values())
        else:
            hooks_features = None
        return last_features, hooks_features

    def _get_hooks_output_channels(self) -> List[int]:
        """Generate hooks output channels numbers.
        
        Returns:
            output_hooks_channels: Hooks output channels numbers.
        """
        if self._features_info is None:
            output_hooks_channels = None
        else:
            output_hooks_channels = [feature.num_channels for feature in self._features_info]
        return output_hooks_channels

    @abstractmethod
    def _get_forward_output_channels(self) -> Union[int, List[int]]:
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
        forward_channels = self._get_forward_output_channels()
        hooks_channels = self._get_hooks_output_channels()
        return forward_channels, hooks_channels

    @property
    def features_info(self):
        return self._features_info

    @property
    def feature_hooks(self):
        return self._feature_hooks
