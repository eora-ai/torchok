import torch.nn as nn
from torch.optim.optimizer import Optimizer
import logging
from typing import Dict, Iterable, Union, Set
from pytorch_lightning.callbacks import BaseFinetuning

from torchok.constructor import CALLBACKS


def get_modules_by_names(module_names: Union[str, Iterable[str]], module: nn.Module) -> Set[nn.Module]:
    """Return modules by it's names.

    Args:
        module_names: Searched module names.
        module: Search module.
    
    Returns:
        found_modules: All found modules.
    """
    if isinstance(module_names, str):
        module_names = [module_names]
    module_names = set(module_names)
    found_modules = set()
    found_module_names = set()
    for module_name, curr_module in module.named_modules():
        if module_name in module_names:
            found_modules.add(curr_module)
            found_module_names.add(module_name)

    not_found_modules = module_names - found_module_names
    if len(not_found_modules) != 0:
        logging.warning(f'Get modules_by_names function, can`t found modules with names = {not_found_modules}')
    return found_modules


@CALLBACKS.register_class
class FreezeUnfreeze(BaseFinetuning):
    def __init__(self, epoch2module_names: Dict[int, str]):
        super().__init__()
        self._epoch2module_names = epoch2module_names

    def freeze_before_training(self, pl_module: nn.Module):
        # Get all module names from self._epoch2module_names
        freeze_module_names = []
        for _, module_names in self._epoch2module_names.items():
            freeze_module_names += module_names

        # Get modules by module names
        freeze_modules = get_modules_by_names(freeze_module_names, pl_module)
        # Freeze every module
        for freeze_module in freeze_modules:
            self.freeze(freeze_module)

    def finetune_function(self, pl_module: nn.Module, current_epoch: int, optimizer: Optimizer, optimizer_idx: int):
        """Unfreeze modules from self._epoch2module_names dictionary.
        
        Args:
            pl_module: Module which contain unfreeze modules.
            current_epoch: Current epoch.
            optimizer: Optimizer.
            optimizer_idx: Optimizer index.
        """
        if current_epoch in self._epoch2module_names:
            # Get modules to unfreeze
            unfreeze_modules = get_modules_by_names(self._epoch2module_names[current_epoch], pl_module)
            # Unfreeze every module
            for unfreeze_module in unfreeze_modules:
                self.unfreeze_and_add_param_group(
                    modules=unfreeze_module,
                    optimizer=optimizer,
                    train_bn=True,
                )
