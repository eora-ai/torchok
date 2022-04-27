from email import generator
import torch
import torch.nn as nn

from dataclasses import dataclass
from functools import partial
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from enum import Enum

from base_feature_model import BaseChannelsModel

@dataclass
class FeatureInfo:
    """Gather feature hooks params.
    
    Args:
        module_name: Name of hooked module.
        channel_number: Number of image channels.
        reduction: Downscale value for get image shape in current module.
    """
    module_name: str
    channel_number: int
    reduction: int


class HookType(Enum):
    FORWARD = 'forward'
    FORWARD_PRE = 'forward_pre'


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
    """ Feature Hook Helper

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torcscript.

    Args:
        hooks: Hooks to be registrate.
        named_modules: nn.Model answer of self.named_modules() function call.
    """

    def __init__(self, hooks: List[Hook], named_modules: generator):
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for i, hook in enumerate(hooks):
            hook_name = hook.module_name
            module = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            # registrate hooks
            if hook.hook_type == HookType.FORWARD_PRE:
                module.register_forward_pre_hook(hook_fn)
            else:
                module.register_forward_hook(hook_fn)
           
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, hook_name: str, *args):
        """Collect hooks outputs in self._feature_outputs.
        
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

    def get_output(self, device: torch.device) -> Dict[str, torch.tensor]:
        """Return hooks output.

        Args:
            device: Saved hooks device.

        Returns:
            output: Dict of module hook name two its output tensor.
        """
        output = self._feature_outputs[device]
        # clear after reading
        self._feature_outputs[device] = OrderedDict()  
        return output


class BaseHookModel(nn.Module):

    def create_hooks(self):
        """Generate feature hooks."""
        self._stage_names = [feature.module_name for feature in self.feature_info]
        self._encoder_channels = [feature.channel_number for feature in self.feature_info]
        hooks = [Hook(module_name=name, hook_type=HookType.FORWARD) for name in self.stage_names]
        self._feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x: torch.Tensor):
        # Return features for classification.
        raise NotImplementedError

    def forward_backbone_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return model output with hooks features."""
        last_features = self.forward(x)
        backbone_features = self._feature_hooks.get_output(x.device)
        backbone_features = list(backbone_features.values())
        backbone_features = [x] + backbone_features
        return last_features, backbone_features

    def forward_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return only hooks features."""
        x = self.forward(x)
        return list(self._feature_hooks.get_output(x.device).values())

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @feature_info.setter
    def feature_info(self, feature_info: List[FeatureInfo]):
        """Check if all feature_info is FeatureInfo class.
        
        Args:
            feature_info: Value to be feature_info class parameter.
            
        Raises:
            ValueError: If any value in feature_info is not FeatureInfo class.
        """
        checked_feature_info = []
        for feature in feature_info:
            if type(feature) == FeatureInfo:
                checked_feature_info.append(feature)
            else:
                raise ValueError('All feature_info must be FeatureInfo class.')
        self.feature_info = checked_feature_info

    @property
    def encoder_channels(self):
        return self._encoder_channels

    @property
    def stage_names(self):
        return self._stage_names

    @property
    def feature_hooks(self):
        return self._feature_hooks


# How to do class without methods =)
class BaseChannelsHookModel(BaseHookModel, BaseChannelsModel):
    pass