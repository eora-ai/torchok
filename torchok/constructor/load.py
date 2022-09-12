import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
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


def sort_state_dict_by_depth(override_name2state_dict: Dict[str, str]) -> List[List[Dict[str, torch.Tensor]]]:
    """Generate sorted by depth list of state dict list with the current depth.
    Where depth is calculated as the number of dots in the dictionary key.

    Args:
        override_name2state_dict: Dicts of module key to state dict, which should override base checkpoint.

    Returns:
        depth2override_state_dicts: Sorted by depth dict, where key - depth, value - list of all state dicts with
            current depth.
    """
    if len(override_name2state_dict) == 0:
        return dict()

    depth2override_state_dicts = defaultdict(list)

    for override_key, override_state_dict in override_name2state_dict.items():
        depth = len(override_key.split('.')) - 1
        depth2override_state_dicts[depth].append(override_state_dict)

    # Sort depth2override_state_dicts by it key - depth
    depth2override_state_dicts = sorted(depth2override_state_dicts.items())
    return depth2override_state_dicts


def get_state_dict_with_prefix(prefix: str, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Generate state dict with prefixed keys. If input state dict keys startswith prefix then no prefix added.

    Args:
        prefix: Prefix for state dict keys that should be added, then state dict keys not begin with prefix.
        state_dict: State dict whose keys should be prefixed.

    Returns:
        state_dict_with_prefix: Prefixed state dict.
    """
    state_dict_with_prefix = dict()
    # Remove spaces and dots
    prefix = prefix.strip(' .')
    prefix = prefix + '.'
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            key = prefix + key
        state_dict_with_prefix[key] = value
    return state_dict_with_prefix


def get_absolute_keys(require_key: str, model_keys: List[str]) -> List[str]:
    """Create absolute model keys from require.

    The keys belong to the model state dict called absolute.

    Args:
        require_key: The key to being straightened.
        model_keys: The model keys.

    Returns:
        absolute_keys: Absolute keys.
    """
    absolute_keys = []
    for key in model_keys:
        if key.startswith(require_key):
            absolute_keys.append(key)
    return absolute_keys


def generate_required_state_dict(base_state_dict: Dict[str, torch.Tensor],
                                 overridden_name2state_dict: Dict[str, Dict[str, torch.Tensor]],
                                 exclude_keys: List[str],
                                 model_keys: List[str],
                                 initial_state_dict: Dict[str, torch.Tensor]
                                 ) -> Dict[str, torch.Tensor]:
    """Generate state dict, which should be loaded from 4 main components: base state dict, overridden state dicts,
    exclude keys and model_keys.

    Base state dict values are overridden with overridden_name2state_dict, where the key in overridden_name2state_dict -
    module name which should be overridden.
    If overridden_name2state_dict has many state dicts for one key in the base state dict, then the state dict is
    selected whose overridden_name module is closer to the key i.e. choose the deepest overridden_name, where depth
    is the number of dots.
    After the base state dict had been overridden, it is necessary to remove keys whose names begin with any
    key in exclude_keys.

    Example:
        >>> model_keys = ['backbone.linear.1', 'backbone.linear.2', 'head.linear.1', 'head.linear.2']
        >>> exclude_keys = ['head.linear.2']
        >>> initial_state_dict = {
        >>>    'backbone.linear.1': torch.tensor(0),
        >>>    'backbone.linear.2': torch.tensor(0),
        >>>    'head.linear.1': torch.tensor(0),
        >>>    'head.linear.2': torch.tensor(0)
        >>> }
        >>> base_state_dict = {
        >>>    'backbone.linear.1': torch.tensor(1),
        >>>    'backbone.linear.2': torch.tensor(1),
        >>>    'head.linear.1': torch.tensor(1),
        >>>    'head.linear.2': torch.tensor(1)
        >>> }
        >>> override_backbone = {
        >>>     'backbone.linear.1': torch.tensor(5),
        >>>     'backbone.linear.2': torch.tensor(3)
        >>> }
        >>> override_backbone_linear_1 = {
        >>>     'backbone.linear.1': torch.tensor(10)
        >>> }
        >>> override_name2state_dict = {
        >>>     'backbone': override_backbone,
        >>>     'backbone.linear.1': override_backbone_linear_1
        >>> }
        >>> generate_required_state_dict(base_state_dict, override_name2state_dict,
        >>>                              exclude_keys, model_keys, initial_state_dict)
        {'backbone.linear.1': tensor(10), 'backbone.linear.2': tensor(3),
         'head.linear.1': tensor(1), 'head.linear.2': tensor(0)}

    Args:
        base_state_dict: Base state dict that should be loaded.
        overridden_name2state_dict: Dicts of module key to state dict, which should override base state dict.
        exclude_keys: Module keys that would be loaded from the initial model state dict.
        model_keys: Model state dict keys.
        initial_state_dict: Model initial state dict.

    Returns:
        required_state_dict: State dict obtained by override base state dict by overridden state dict, which not
            contain the keys starting with exclude keys.

    Raises:
        ValueError: If any exclude_key not in model.
    """
    required_state_dict = dict()

    # Add prefix for every overridden state dict
    overridden_full_name2state_dict = dict()
    for overridden_name, state_dict in overridden_name2state_dict.items():
        prefixed_state_dict = get_state_dict_with_prefix(prefix=overridden_name, state_dict=state_dict)
        overridden_full_name2state_dict[overridden_name] = prefixed_state_dict

    # Get sorted by depth state dict list that must be overridden
    depth2state_dicts = sort_state_dict_by_depth(overridden_full_name2state_dict)

    # Firstly change model state dict by base state dict
    required_state_dict.update(base_state_dict)

    # Then change model state dict with overridden state dicts, in order of it's depths
    for _, depth_state_dicts in depth2state_dicts.items():
        for state_dict in depth_state_dicts:
            required_state_dict.update(state_dict)

    # Create absolute exclude keys
    absolute_exclude_keys = []
    for exclude_key in exclude_keys:
        absolute_keys = get_absolute_keys(exclude_key, model_keys)
        if len(absolute_keys) == 0:
            raise ValueError(f'Load checkpoint. Found exclude key {exclude_key} which not in model_keys.')
        absolute_exclude_keys += absolute_keys

    # Create exclude state dict
    exclude_state_dict = dict()
    for key in absolute_exclude_keys:
        exclude_state_dict[key] = initial_state_dict[key]

    # Update require state dict with exclude state dict
    required_state_dict.update(exclude_state_dict)

    return required_state_dict


def load_checkpoint(model: pl.LightningModule, base_ckpt_path: Optional[str] = None,
                    overridden_name2ckpt_path: Optional[Dict[str, str]] = None,
                    exclude_keys: Optional[List[str]] = None, strict: bool = True):
    """Load checkpoint to model.

    Args:
        model: Module to load checkpoint into.
        base_ckpt_path: Base checkpoint path that should be loaded.
        overridden_name2ckpt_path: Dicts of module key to checkpoint path, which should override base checkpoint.
        exclude_keys: Module keys that should not be loaded.
    """
    # If no checkpoints to load
    if base_ckpt_path is None and overridden_name2ckpt_path is None:
        logging.info('Load checkpoint function. You wrote checkpoint parameters in yaml config without base '
                     'checkpoint path and overridden checkpoint paths!')
        return
    initial_state_dict = model.state_dict()
    model_keys = list(initial_state_dict.keys())

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
                                                       exclude_keys, model_keys, initial_state_dict)

    model.load_state_dict(required_state_dict, strict=strict)
