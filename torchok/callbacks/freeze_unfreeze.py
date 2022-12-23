import inspect
from typing import Dict, Iterable, List, Union, Optional, Any

import torch.nn as nn
from pytorch_lightning.callbacks import BaseFinetuning
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

from torchok.constructor import CALLBACKS


def get_modules(module_dict: Dict[str, Any], module: nn.Module) -> List[nn.Module]:
    """Return modules by its names.

    Args:
        module_dict: dict with information about searched modules.
        module: The module in which it is searched.

    Returns:
        found_modules: List of all found modules.
    """
    module_name = module_dict['module_name']
    target_module = module
    if module_name != '':  # empty `module_name` stands for the whole model
        for block_name in module_name.split('.'):
            children = dict(target_module.named_children())
            target_module = children.get(block_name, None)
            if target_module is None:
                raise ValueError(f"Module `{module_name}` is not found")
    if 'stages' in module_dict:
        if not hasattr(target_module, 'get_stages'):
            raise ValueError(f"You specified `stages` in `{module_name}` "
                             f"but this module does not have `get_stages` method")
        target_module = target_module.get_stages(module_dict['stages'])

    if 'module_class' in module_dict:
        module_class = module_dict['module_class']
        filtered_modules = nn.ModuleList()
        for mod in target_module.modules():
            parents = [i.__name__ for i in inspect.getmro(type(mod))]
            if module_class in parents:
                filtered_modules.append(mod)
        if len(filtered_modules) == 0:
            raise ValueError(f"Module `{module_name}` does not have submodules of `{module_class}` type.")
        target_module = filtered_modules

    return target_module


@CALLBACKS.register_class
class FreezeUnfreeze(BaseFinetuning):
    """Callback to freeze modules and incremental unfreeze these modules during training."""

    def __init__(self, freeze_modules: List[Dict[str, Any]], top_down_freeze_order: bool = True):
        """Init FreezeUnfreeze.

        Args:
            freeze_modules: List with dictionaries of models to be frozen-unfrozen with possible keys of dictionaries:

                - `module_name` (str):
                    module name relative to the task on which freeze will be applied.
                    For example `backbone.layer1`. Empty string in the `module_name` stands for the whole model.
                - `epoch` (int, optional):
                    number of epochs when module to be frozen. If not specified then module will be frozen forever.
                - `stages` (int, optional):
                    if specified with module_name that has ``get_stage(int)`` attribute,
                    apply freeze only on modules returned from ``get_stage(int)``. Usually used with
                    a backbone: stage 0 refers to stem layer, stage `i` refers to `i-1` model layer block and all
                    preceding blocks. If not specified, all blocks will be frozen.
                - `module_class` (str, optional):
                    if specified apply freeze only on the modules of the same type or
                    successors of specified type.
                - `bn_requires_grad` (bool, optional):
                    if specified batch norms' gradients will be set up separately
                    from other blocks. If not specified processed as the other modules.
                - `bn_track_running_stats` (bool, optional):
                    if specified batch norms train mode will be set up
                    separately from other blocks. If not specified processed as the other modules.
            top_down_freeze_order: If true freeze policy will be applied firstly on top modules, e.g. `aa` > `aa.bb`.
                In this case freeze policy `aa.bb` will overwrite freeze policy in `aa` related to `aa.bb`.
                Otherwise, freeze policy for bottom layers will be applied first.
        """
        super().__init__()
        self.freeze_modules = sorted(freeze_modules, key=lambda x: x['module_name'], reverse=not top_down_freeze_order)

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
    def freeze(modules: Union[Module, Iterable[Union[Module, Iterable]]], module_dict: Dict[str, Any]) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            module_dict: If True, leave the BatchNorm layers in training mode
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm):
                for param in mod.parameters(recurse=False):
                    param.requires_grad = module_dict.get("bn_requires_grad", False)
                mod.track_running_stats = module_dict.get("bn_track_running_stats", False)
            else:
                for param in mod.parameters(recurse=False):
                    param.requires_grad = False

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
        for module_dict in self.freeze_modules:
            freeze_module = get_modules(module_dict, pl_module)
            self.freeze(freeze_module, module_dict)

    def finetune_function(self, pl_module: nn.Module, current_epoch: int, optimizer: Optimizer, optimizer_idx: int):
        """Unfreeze modules from self.epoch2module_names dictionary.

        Args:
            pl_module: Module which contain unfreeze modules.
            current_epoch: Current epoch.
            optimizer: Optimizer.
            optimizer_idx: Optimizer index.
        """
        for module_dict in self.freeze_modules:
            if ('epoch' in module_dict) and (module_dict['epoch'] <= current_epoch):
                # Get modules to unfreeze
                unfreeze_modules = get_modules(module_dict, pl_module)
                self.unfreeze_and_add_param_group(
                    modules=unfreeze_modules,
                    optimizer=optimizer,
                    train_bn=True,
                )

        # Freeze blocks with overlapping policies that have later unfreeze epoch
        for module_dict in self.freeze_modules:
            if ('epoch' not in module_dict) or ('epoch' in module_dict and module_dict['epoch'] > current_epoch):
                freeze_module = get_modules(module_dict, pl_module)
                self.freeze(freeze_module, module_dict)
