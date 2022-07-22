import torch.nn as nn
import logging
from typing import List, Set, Union, Dict, Tuple


def change_train_type(model: nn.Module, module_names: Union[Set[str], List[str]], freeze: bool = True):
    """Freeze or unfreeze modules.

    Args:
        model: The model in which the modules will be manipulated.
        module_names: The module names which will be manipulated.
        freeze: If freeze or unfreeze.
    """
    if len(module_names) == 0:
        return

    found_modules = set()
    for module_name, module in model.named_modules():
        if module_name in module_names:
            found_modules.add(module_name)
            if freeze:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
            else:
                for param in module.parameters():
                    param.requires_grad = True
                module.train()

    not_found_modules = set(module_names) - found_modules
    if len(not_found_modules) != 0:
        logging.warning(f'change_train_type method, not found modules = {not_found_modules}')


def get_freeze_unfreeze_by_epoch(epoch2module_names: Dict[int, List[str]],
                                 current_epoch: int) -> Tuple[List[str], List[str]]:
    """Generate module names which need freeze or unfreeze depending on the train epoch.
    
    Args:
        epoch2module_names: Dict with unfreeze epoch keys and it's list of module names as values.
        current_epoch: Train epoch number.

    Returns:
        freeze_module_names: List module names which need to freeze.
        unfreeze_module_names: List module names which need to unfreeze.

    Raises:
        ValueError: If epoch2module_names have a module name with different unfreeze epochs.
    """
    module_name2epoch = dict()
    for epoch, module_names in epoch2module_names.items():
        for module_name in module_names:
            if module_name in module_name2epoch and module_name2epoch[module_name] != epoch:
                raise ValueError(f'Error in get_freeze_unfreeze_by_epoch freeze method. '
                                 f'Have a module - {module_name} with different unfreeze epochs.')
            module_name2epoch[module_name] = epoch

    freeze_module_names = []
    unfreeze_module_names = []
    if epoch == 0:
        freeze_module_names = list(module_name2epoch.keys())
    else:
        for module_name, epoch in module_name2epoch.items():
            if epoch > current_epoch:
                freeze_module_names.append(module_name)
            else:
                unfreeze_module_names.append(module_name)
    
    return freeze_module_names, unfreeze_module_names
