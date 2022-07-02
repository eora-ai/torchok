import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load
from typing import Optional, Dict, List, Callable, OrderedDict, Union


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


def sort_state_dict_by_depth(override_name2state_dict: Dict[str, str]) -> List[List[OrderedDict[str, torch.Tensor]]]:
    """Generate sorted by depth list of state dict list with the current depth.
    Where depth is calculated as the number of dots in the dictionary key.

    Args:
        override_name2state_dict: Dicts of module key to state dict, which should override base checkpoint.

    Returns:
        sorted_state_dicts: Sorted by depth list of state dicts list. Where the positional index means the depth 
            for each state dicts at this position.
    """
    max_depth = 0
    sorted_state_dicts = [[]] * len(override_name2state_dict)

    for override_key, override_state_dict in override_name2state_dict.items():
        depth = len(override_key.split('.')) - 1
        sorted_state_dicts[depth].append(override_state_dict)
        max_depth = depth if depth > max_depth else max_depth

    sorted_state_dicts = sorted_state_dicts[: max_depth + 1]
    return sorted_state_dicts
    

def state_dict_with_prefix(prefix: str, state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Generate state dict with prefixed keys. If input state dict keys startswith prefix then no prefix added.

    Args:
        prefix: Prefix for state dict keys that should be added, then state dict keys not begin with prefix.
        state_dict: State dict whose keys should be prefixed.

    Returns:
        state_dict_with_prefix: Prefixed state dict.
    """
    state_dict_with_prefix = OrderedDict()
    # remove spaces and dots
    prefix = prefix.strip(' .')
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            key = prefix + '.' + key
        state_dict_with_prefix[key] = value
    return state_dict_with_prefix


def get_straighten_keys(require_key: str, model_keys: List[str]) -> List[str]:
    """Create straighten model keys from require.
    
    Args:
        require_key: The key to being straightened.
        model_keys: The model keys.
    
    Returns:
        straighten_keys: Straightened keys.
    """
    straightened_keys = []
    for key in model_keys:
        if key.startswith(require_key):
            straightened_keys.append(key)
    return straightened_keys


def generate_required_state_dict(base_state_dict: OrderedDict[str, torch.Tensor], 
                                 overridden_name2state_dict: Dict[str, OrderedDict[str, torch.Tensor]], 
                                 exclude_keys: List[str],
                                 model_keys: List[str]) -> OrderedDict[str, torch.Tensor]:
    """Generate state dict, which should be loaded from 3 main components: base state dict, overridden state dicts and
    exclude keys.
    
    Args:
        base_state_dict: Base state dict that should be loaded.
        overridden_name2state_dict: Dicts of module key to state dict, which should override base state dict.
        exclude_keys: Module keys that should not be loaded.
        model_keys: Model state dict keys.

    Returns:
        required_state_dict: State dict obtained by override base state dict by overridden state dict, which not 
            contain the keys starting with exclude keys.
    """
    required_state_dict = OrderedDict()
    overridden_full_name2state_dict = dict()
    # Add prefix for every overridden state dict
    for overridden_name, state_dict in overridden_name2state_dict.items():
        overridden_full_name2state_dict[overridden_name] = state_dict_with_prefix(prefix=overridden_name,
                                                                                  state_dict=state_dict)
    # Get sorted by depth state dict list that must be overridden
    sorted_state_dicts = sort_state_dict_by_depth(overridden_name2state_dict)
    
    # Firstly change model state dict by base state dict
    required_state_dict.update(base_state_dict)
    
    # Then change model state dict with overridden state dicts, in order of it's depths
    for depth in range(len(sorted_state_dicts)):
        for state_dicts in sorted_state_dicts[depth]:
            for state_dict in state_dicts:
                required_state_dict.update(base_state_dict)

    straightened_exclude_keys = [get_straighten_keys(exclude_key, model_keys) for exclude_key in exclude_keys]
    required_state_dict.pop(straightened_exclude_keys, None)
    return required_state_dict


def load_checkpoint(model: pl.LightningModule, base_ckpt_path: Optional[str] = None, 
                    overridden_name2ckpt_path: Optional[Dict[str, str]] = None, 
                    exclude_keys: Optional[List[str]] = None):
    """Load checkpoint to model.
    
    Args:
        model: Module to load checkpoint into.
        base_ckpt_path: Base checkpoint path that should be loaded.
        override_name2ckpt_path: Dicts of module key to checkpoint path, which should override base checkpoint.
        exclude_names: Module keys that should not be loaded.
    """
    # If no checkpoints to load
    if base_ckpt_path is None and overridden_name2ckpt_path is None:
        logging.warning('Load checkpoint function. You wrote checkpoint parameters in yaml config without base '
                        'checkpoint path and overridden checkpoint paths!')
        return

    model_state_dict = model.state_dict()
    model_keys = list(model_state_dict.keys())

    if exclude_keys is None:
        exclude_keys = list()

    # Load base state dict
    base_state_dict = load_state_dict(base_ckpt_path, map_location='cpu') if base_ckpt_path is not None else dict()
    
    # Load overridden state dicts
    if overridden_name2ckpt_path is None:
        overridden_name2state_dict = dict()
    else:
        overridden_name2state_dict = {name: load_state_dict(ckpt_path)
                                      for name, ckpt_path in overridden_name2ckpt_path.items()}

    required_state_dict = generate_required_state_dict(base_state_dict, overridden_name2state_dict, 
                                                       exclude_keys, model_keys)

    model_state_dict.update(required_state_dict)
    model.load_state_dict(model_state_dict, strict=True)
