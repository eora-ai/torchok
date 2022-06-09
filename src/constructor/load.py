import functools
import operator
from typing import Optional, Dict, List, Callable, Union

import torch
from pytorch_lightning.utilities.cloud_io import load


def load_state_dict(checkpoint_path: str, map_location: Optional[Union[str, Callable, torch.device]] = 'cpu'):
    """Loads a checkpoint state_dict.
    
    Args:
        checkpoint_path: Path or URL of the checkpoint, it can be S3 on AWS, GCS on Google Cloud, or ADL on Azure.
        map_location: How to remap storage locations.

    Returns:
        state_dict: Downloaded state dict.
    """
    checkpoint = load(checkpoint_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    return state_dict


def create_straighten_keys(require_key: str, model_keys: List[str]):
    """Create straighten model keys from require.
    
    Args:
        require_key: The key to being straightened.
        model_keys: The model keys.
    """
    straighten_keys = []
    for key in model_keys:
        if key.startswith(require_key):
            straighten_keys.append(key)
    return straighten_keys


class StateDictWithDepthStructure:
    """Structure for state dict and model key depth.
    
    Contains 2 dictionary:
        First normal state dict self.__state_dict.
        Second dict with model key and added state dict depth key. Where the depth is the count of dots.
    """
    def __init__(self):
        """Initialize structure with empty dicts."""
        self.__state_dict = dict()
        self.__state_dict_depth = dict()

    def __add(self, key: str, weight: torch.Tensor, depth: int):
        """Add weight and depth to dicts by key.
        
        Args:
            key: Added weight key.
            weight: Added weight.
            depth: Added depth.
        """
        self.__state_dict[key] = weight
        self.__state_dict_depth[key] = depth

    def add(self, key: str, weight: torch.Tensor, depth: int):
        """Add weight and depth to dicts by key if current depth more than previous.
        
        Args:
            key: Added weight key.
            weight: Added weight.
            depth: Added depth.
        """
        if key in self.__state_dict_depth:
            old_depth = self.__state_dict_depth[key]
            if depth > old_depth:
                self.__add(key, weight, depth)
        else:
            self.__add(key, weight, depth)
        
    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.__state_dict

    @property
    def state_dict_depth(self) -> Dict[str, int]:
        return self.__state_dict_depth

    
def create_override_state_dict(override_name2state_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Create one state dict from dictionary of override state dicts.
    
    If override state dicts have many weights of one straighten key then the weight with the biggest depth
    would be written in final state dict. There is the depth is the count of dots in the dictionary key.

    Example:
        >>> backbone = {
        >>>     'backbone.linear.1': torch.tensor(5),
        >>>     'backbone.linear.2': torch.tensor(3)
        >>> }
        >>> backbone_linear_1 = {
        >>>     'backbone.linear.1': torch.tensor(10)
        >>> }
        >>> override_name2state_dict = {
        >>>     'backbone': backbone,
        >>>     'backbone.linear.1': backbone_linear_1
        >>> }
        >>> create_override_state_dict(override_name2state_dict)
        {'backbone.linear.1': tensor(10), 'backbone.linear.2': tensor(3)}

    Args: 
        override_name2state_dict: Dictionary of module name to it's state_dict.

    Returns:
        override_state_dict: Created state dictionary.
    """
    state_dict_with_depth = StateDictWithDepthStructure()
    for key, state_dict in override_name2state_dict.items():
        straighten_keys = create_straighten_keys(key)
        depth = len(key.split('.'))
        for straighten_key in straighten_keys:
            weight = state_dict[straighten_key]
            state_dict_with_depth.add(straighten_key, weight, depth)

    override_state_dict = state_dict_with_depth.state_dict
    return override_state_dict


def generate_require_state_dict(model_state_dict: Dict[str, torch.Tensor], base_state_dict: Dict[str, torch.Tensor], 
                                override_name2state_dict: Dict[str, torch.Tensor], exclude_names: List[str]):
    """Create load state dict from base_state_dict, override_name2state_dict, exclude_names.

    Args:
        model_state_dict: Current model state dict.
        base_state_dict: Base state dict that must be loaded.
        override_name2state_dict: State dicts that must be override base_state_dict.
        exclude_names: Keys that should not be loaded.

    Returns:
        base_state_dict: Generated final state dict.
    """
    model_keys = list(model_state_dict.keys())
    straighten_exclude_names = [create_straighten_keys(name, model_keys) for name in exclude_names]
    # Concatenate list of list
    straighten_exclude_names = functools.reduce(operator.iconcat, straighten_exclude_names, [])
    override_state_dict = create_override_state_dict(override_name2state_dict)
    # Merge state dicts
    base_state_dict.update(override_state_dict)
    # Remove exclude keys
    for exclude_key in straighten_exclude_names:
        base_state_dict.pop(exclude_key, None)

    return base_state_dict


def load_checkpoint(model: "pl.LightningModule", base_ckpt_path: Optional[str] = None, 
                    override_name2ckpt_path: Optional[Dict[str, str]] = None, 
                    exclude_names: Optional[List[str]] = None):
    """Load checkpoint to model.
    
    Args:
        model: Module to load checkpoint into.
        base_ckpt_path: Base checkpoint path that must be loaded.
        override_name2ckpt_path: Dicts module key to checkpoint path, that must be override base checkpoint.
        exclude_names: Module keys that should not be loaded.
    """
    # If no checkpoints to load
    if base_ckpt_path is None and override_name2ckpt_path is None:
        return

    model_state_dict = model.state_dict()

    # Load state dicts
    base_state_dict = load_state_dict(base_ckpt_path, map_location='cpu') if base_ckpt_path is not None else dict()

    if override_name2ckpt_path is None:
        override_name2state_dict = dict()
    else:
        override_name2state_dict = {name: load_state_dict(ckpt_path) 
                                    for name, ckpt_path in override_name2ckpt_path.items()}

    load_state_dict = generate_require_state_dict(model_state_dict, base_state_dict, 
                                                  override_name2state_dict, exclude_names)

    model.load_state_dict(load_state_dict, strict=False)
