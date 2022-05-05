import torch
import torch.nn as nn

from dataclasses import dataclass
from functools import partial
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Generator, Optional
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

    def __init__(self, hooks: List[Hook], named_modules: Generator[Tuple[str, nn.Module]]):
        """Init FeatureHooks class.
        
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
        self._hook_features[device][hook_name] = x

    def get_features(self, device: torch.device) -> Dict[str, torch.tensor]:
        """Return hooks features.

        Args:
            device: Saved hooks device.

        Returns:
            output: Dict of module hook name two its output tensor.
        """
        output = self._hook_features[device]
        # clear after reading
        self._hook_features[device] = OrderedDict()  
        return output


class BaseModel(nn.Module):
    """Base model for all TorchOk Models - Backbone, Neck, Pooling and Head.

    This class supports adding feature hooks to some model layers.
    Also it's have feature info list for every hook.
    To create hooks, method self.get_features_info() must be rewrited!
    """
    def __init__(self):
        """Inits BaseModel class and it's hooks.

        Raises:
            ValueError: If rewrited self.get_features_info() method return not FeatureInfo list.
        """
        super().__init__()

        # Create feature_info list with hooks
        self._features_info, self._feature_hooks = self._create_hooks()


    def get_features_info(self) -> List[FeatureInfo]:
        """Method for initialize Hooks.
        
        It must be rewrite in inherited classes, if Module would need hooks.

        Retruns: Hooks FeatureInfo list.
        """
        return None

    def _create_hooks(self) -> Tuple[List[FeatureInfo], FeatureHooks]:
        """Call self.get_features_info() method to generate feature_info list and then generate feature hooks.
        
        If self.get_features_info() not rewrited in inheritable class the features_info and hooks would be None.

        Returns:
            feature_info: Hooks feature info list.
            feature_hooks: Generated hooks.
        """
        features_info = self.get_features_info()
        # Check if all self._feature_info is FeatureInfo types.
        if features_info is not None:
            self.__check_features_info_types()
            hooks = [
                Hook(module_name=feature.module_name, hook_type=feature.hook_type) 
                for feature in features_info
                ]
            feature_hooks = FeatureHooks(hooks, self.named_modules())
        else:
            feature_hooks = None

        return features_info, feature_hooks

    def __check_features_info_types(self):
        """Check if all self._feature_info is FeatureInfo class.
            
        Raises:
            ValueError: If any value in feature_info is not FeatureInfo class.
        """
        for feature in self._features_info:
            if type(feature) != FeatureInfo:
                raise ValueError('All feature_info must be FeatureInfo class.')

    def forward(self, x: torch.Tensor):
        # Return features for classification.
        raise NotImplementedError

    def forward_stage_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return model output with hooks features.
        
        Returns:
            last_features: Module forward method output.
            hooks_features: Hooks outputs.
        """
        last_features = self.forward(x)
        hooks_features = self._feature_hooks.get_features(x.device)
        hooks_features = list(hooks_features.values())
        hooks_features = [x] + hooks_features
        return last_features, hooks_features

    def get_output_hooks_channels(self) -> List[int]:
        """Generate hooks output channels numbers.
        
        Returns:
            output_hooks_channels: Hooks output channels numbers.
        """
        output_hooks_channels = [feature.num_channels for feature in self._feature_info]
        return output_hooks_channels

    @property
    def feature_info(self):
        return self._features_info

    @property
    def feature_hooks(self):
        return self._feature_hooks
