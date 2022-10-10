import logging
from typing import Dict, Iterable, List, Set, Union, Optional

import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer

from torchok.constructor import CALLBACKS


def get_modules_by_names(module_names: Union[str, Iterable[str]], module: nn.Module) -> Set[nn.Module]:
    """Return modules by its names.

    Args:
        module_names: Searched module names.
        module: The module in which it is searched.

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
        logging.warning(f'get_modules_by_names function can`t find modules with names {not_found_modules}')
    return found_modules


@CALLBACKS.register_class
class FreezeUnfreeze(BaseFinetuning):
    """Callback to freeze modules and incremental unfreeze this modules during training."""

    def __init__(self, epoch2module_names: Dict[int, List[str]], freeze_bn: bool = True):
        """Init FreezeUnfreeze.

        Args:
            epoch2module_names: Incremental unfreeze dictionary. Keys - unfreeze epoch,
                values - module names to unfreeze. By default, all the modules named in this dictionary
                will be frozen before training.
            freeze_bn: If freeze batch norm layers.
        """
        super().__init__()
        self.epoch2module_names = epoch2module_names
        self.freeze_bn = freeze_bn

    @staticmethod
    def make_trainable(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        """Unfreezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for module in modules:
            if isinstance(module, _BatchNorm):
                module.track_running_stats = True
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in module.parameters(recurse=False):
                param.requires_grad = True

    @staticmethod
    def freeze_module(module: Module):
        if isinstance(module, _BatchNorm):
            module.track_running_stats = False
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = False

    @staticmethod
    def freeze(modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode

        Returns:
            None
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                FreezeUnfreeze.make_trainable(mod)
            else:
                FreezeUnfreeze.freeze_module(mod)

    @staticmethod
    def unfreeze_and_add_param_group(
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ) -> None:
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.
        """
        FreezeUnfreeze.make_trainable(modules)
        params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.0
        params = FreezeUnfreeze.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = FreezeUnfreeze.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({"params": params, "lr": params_lr / denom_lr})

    def freeze_before_training(self, pl_module: nn.Module):
        """Freeze modules before training.

        Freeze all the modules named in self.epoch2module_names.

        Args:
            pl_module: Module which contain unfreeze modules.
        """
        # Get all module names from self.epoch2module_names
        freeze_module_names = []
        for module_names in self.epoch2module_names.values():
            freeze_module_names += module_names

        # Get modules by module names
        freeze_modules = get_modules_by_names(freeze_module_names, pl_module)

        train_bn = not self.freeze_bn

        # Freeze every module
        for freeze_module in freeze_modules:
            self.freeze(freeze_module, train_bn=train_bn)

    def finetune_function(self, pl_module: nn.Module, current_epoch: int, optimizer: Optimizer, optimizer_idx: int):
        """Unfreeze modules from self.epoch2module_names dictionary.

        Args:
            pl_module: Module which contain unfreeze modules.
            current_epoch: Current epoch.
            optimizer: Optimizer.
            optimizer_idx: Optimizer index.
        """
        if current_epoch in self.epoch2module_names:
            # Get modules to unfreeze
            unfreeze_modules = get_modules_by_names(self.epoch2module_names[current_epoch], pl_module)
            # Unfreeze every module
            for unfreeze_module in unfreeze_modules:
                self.unfreeze_and_add_param_group(
                    modules=unfreeze_module,
                    optimizer=optimizer,
                    train_bn=True,
                )
