from typing import Any, Dict, List, Optional, Union

import albumentations as A
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn import GroupNorm, LayerNorm, Parameter
from torch.nn import Module, ModuleList
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchok.constructor import DATASETS, LOSSES, OPTIMIZERS, SCHEDULERS, TRANSFORMS
from torchok.constructor.config_structure import Phase
from torchok.data.datasets.base import ImageDataset
from torchok.losses.base import JointLoss
from torchok.metrics.metrics_manager import MetricsManager


class Constructor:
    """Provides factory features for optimizers, schedulers, loss functions, data loaders and metrics."""

    def __init__(self, hparams: DictConfig):
        """Init Constructor with hparams.

        Args:
            hparams: Configuration dictionary for all the mentioned components.
            The dictionary should contain at least the following parameter groups (see configuration for details):
            - optimization
            - data
            - losses
            - metrics
        """
        self._hparams = hparams

    def configure_optimizers(self, modules: Union[Module, List[Union[Module]]],
                             optim_idx: int = -1) -> List[Dict[str, Union[Optimizer, Dict[str, Any]]]]:
        """Create optimizers and learning rate schedulers from a pre-defined configuration.

        Note: optimization parameters are split into two groups: decay and no-decay specifying which parameters can be
        weight decayed and which not. Each module is checked on having an attribute `no_weight_decay`,
        specifying a list of submodules which must be not weight decayed (useful in transformer models such as Swin).
        All *.bias parameters, 1D tensors and scalars are put into no-decay group according to the best practice.

        Args:
            modules: Modules for optimization
            optim_idx: Optimizer and scheduler index if specific optimizer/scheduler group is needed to be created.
            Default is -1 meaning all the optimizers/schedulers are created.

        Returns:
            List of dicts in the PyTorch Lightning accepted format:
            - optimizer: PyTorch-like optimizer
            - lr_scheduler: dict
                scheduler: PyTorch-like scheduler
                <any PyTorch Lightning parameters. See `their documentation`_>

        Raises:
            - ValueError: When a requested optimizer/scheduler group with optim_idx isn't present in configuration
            - ValueError: When parameters type is out of supported types list (see typing)

        .. _their documentation:
        https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_module.html#configure-optimizers
        """
        optims_params = self._hparams.optimization
        if 0 <= optim_idx < len(optims_params):
            optims_params = [optims_params[optim_idx]]
        elif optim_idx >= len(optims_params):
            raise ValueError(f'You requested optimization with index {optim_idx} while '
                             f'there\'re only {len(optims_params)} optimization parameters are specified')

        opt_sched_list = []
        for optim_params in optims_params:
            optimizer = self.create_optimizer(modules, optim_params.optimizer)
            opt_sched = {'optimizer': optimizer}

            if optim_params.scheduler is not None:
                scheduler_dict = self._create_scheduler(optimizer, optim_params.scheduler)
                opt_sched['lr_scheduler'] = scheduler_dict

            opt_sched_list.append(opt_sched)

        return opt_sched_list

    @staticmethod
    def create_optimizer(modules: Union[Module, List[Module]],
                         optimizer_params: DictConfig) -> Optimizer:
        """Default constructor for optimizers.

        By default, each parameter share the same optimizer settings, and we
        provide an attribute ``paramwise_cfg`` in `optimizer_params` to specify parameter-wise settings.
        It is a dict and may contain the following fields:

        - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
          one of the keys in ``custom_keys`` is a substring of the name of one
          parameter, then the setting of the parameter will be specified by
          ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
          be ignored. It should be noted that the aforementioned ``key`` is the
          longest key that is a substring of the name of the parameter. If there
          are multiple matched keys with the same length, then the key with lower
          alphabet order will be chosen.
          ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
          and ``decay_mult``. See Example 2 below.
        - ``bias_lr_mult`` (float): It will be multiplied to the learning
          rate for all bias parameters (except for those in normalization
          layers and offset layers of DCN).
        - ``bias_decay_mult`` (float): It will be multiplied to the weight
          decay for all bias parameters (except for those in
          normalization layers, depth-wise conv layers, offset layers of DCN).
        - ``norm_decay_mult`` (float): It will be multiplied to the weight
          decay for all weight and bias parameters of normalization
          layers.
        - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
          decay for all weight and bias parameters of depth-wise conv
          layers.
        - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
          rate for parameters of offset layer in the deformable convolutions
          of a model.

        Note:

            1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            override the effect of ``bias_lr_mult`` in the bias of offset layer.
            So be careful when using both ``bias_lr_mult`` and
            ``dcn_offset_lr_mult``. If you wish to apply both of them to the offset
            layer in deformable convs, set ``dcn_offset_lr_mult`` to the original
            ``dcn_offset_lr_mult`` * ``bias_lr_mult``.

            2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
            apply it to all the DCN layers in the model. So be careful when the
            model contains multiple DCN layers in places other than backbone.

        Args:
            modules: The module or list of modules with parameters to be optimized.
            optimizer_params: The config dict of the optimizer.
                Positional fields are
                    - `name`: class name of the optimizer.

                Optional fields are
                    - `params`: dict with parameters to set up an optimize, e.g., lr, weight_decay, momentum.
                    - `paramwise_cfg`:
        """

        optimizer_class = OPTIMIZERS.get(optimizer_params.name)
        paramwise_cfg = optimizer_params.paramwise_cfg
        optimizer_cfg = optimizer_params.params

        if isinstance(modules, (tuple, list)):
            modules = ModuleList(modules)

        if not paramwise_cfg:
            parameters: List[Union[Dict, Parameter]] = list(modules.parameters())
        else:
            # set param-wise lr and weight decay recursively
            parameters: List[Union[Dict, Parameter]] = []
            Constructor.add_params(parameters, modules, optimizer_cfg, paramwise_cfg)

        optimizer = optimizer_class(parameters, **optimizer_cfg)

        return optimizer

    @staticmethod
    def add_params(
            parameters: List[Dict],
            module: nn.Module,
            optimizer_cfg: Dict,
            paramwise_cfg: Optional[Dict] = None,
            prefix: str = '',
            is_dcn_module: Union[int, float, None] = None
    ) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            parameters: A list of param groups, it will be modified in place.
            module: The module to be added.
            optimizer_cfg: Dict with optimizer params.
            paramwise_cfg: Dict with setting for certain model parameters.
            prefix: The prefix of the module
            is_dcn_module: If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """

        if paramwise_cfg is None:
            paramwise_cfg = {}

        base_lr = optimizer_cfg.get('lr', None)
        base_wd = optimizer_cfg.get('weight_decay', None)
        # get param-wise options
        custom_keys = paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = paramwise_cfg.get('dwconv_decay_mult', 1.)
        dcn_offset_lr_mult = paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                parameters.append(param_group)
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = base_lr * lr_mult
                    if base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = base_lr * bias_lr_mult

                if prefix.find('conv_offset') != -1 and is_dcn_module and isinstance(module, torch.nn.Conv2d):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group['weight_decay'] = base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group['weight_decay'] = base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        param_group['weight_decay'] = base_wd * bias_decay_mult
            parameters.append(param_group)

        is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            Constructor.add_params(parameters, child_mod, optimizer_cfg, paramwise_cfg,
                                   prefix=child_prefix, is_dcn_module=is_dcn_module)

    @staticmethod
    def _create_scheduler(optimizer: Optimizer, scheduler_params: DictConfig) -> Dict[str, Any]:
        scheduler_class = SCHEDULERS.get(scheduler_params.name)
        scheduler = scheduler_class(optimizer, **scheduler_params.params)
        pl_params = scheduler_params.pl_params

        return {
            'scheduler': scheduler,
            **pl_params
        }

    def create_dataloaders(self, phase: Phase) -> List[DataLoader]:
        """Create data loaders.

        Each data loader is based on a dataset while dataset consists
        augmentations and transformations in the `albumentations`_ format.

        Args:
            phase: Phase for which the data loaders are to be built.

        Returns:
            List of data loaders to be used inside `PyTorch Lightning Module`_
        .. _albumentations: https://albumentations.ai/docs
        .. _PyTorch Lightning Module:
        https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_module.html#train-dataloader

        Raises:
            - ValueError: When transforms are not specified for composition augmentations of albumentation
            - ValueError: When OneOrOther composition is passed that isn't supported
        """
        if phase in self.hparams.data:
            dataloaders = [
                self._prepare_dataloader(phase_params.dataset, phase_params.dataloader)
                for phase_params in self.hparams.data[phase] if phase_params is not None
            ]
            dataloaders = dataloaders if len(dataloaders) > 1 else dataloaders[0]
            return dataloaders
        else:
            return []

    @staticmethod
    def _prepare_dataloader(dataset_params: DictConfig, dataloader_params: DictConfig) -> DataLoader:
        dataset = Constructor._create_dataset(dataset_params)
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None

        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn,
                            **dataloader_params)

        return loader

    @staticmethod
    def _create_dataset(dataset_params: DictConfig) -> ImageDataset:
        transform = Constructor._create_transforms(dataset_params.transform)
        # TODO remove when OmegaConf is fixing the bug, write to issue to Omegaconf!
        # Config structure had 'augment' parameter with default value = None, but in loaded config
        # 'augment' is not in keys of dataset_params dictionary. So it must be written like
        # augment_params = dataset_params.augment
        augment_params = dataset_params.get('augment', None)
        augment = Constructor._create_transforms(augment_params)

        dataset_class = DATASETS.get(dataset_params.name)

        return dataset_class(transform=transform, augment=augment, **dataset_params.params)

    @staticmethod
    def _prepare_transforms_recursively(transforms: ListConfig) -> List[Union[A.Compose, A.BaseCompose]]:
        transforms_list = []

        for transform_info in transforms:
            transform_name = transform_info.name
            transform_params = transform_info.get('params', dict())

            if transform_name in ['Compose', 'OneOf', 'SomeOf', 'PerChannel', 'Sequential']:
                transform = Constructor._prepare_base_compose(transform_name, **transform_params)
            elif transform_name == 'OneOrOther':
                raise ValueError('OneOrOther composition is currently not supported')
            else:
                transform = TRANSFORMS.get(transform_name)(**transform_params)

            transforms_list.append(transform)

        return transforms_list

    @staticmethod
    def _prepare_base_compose(compose_name: str, **kwargs) -> A.Compose:
        transforms = kwargs.pop('transforms', None)
        if transforms is None:
            raise ValueError(f'There are transforms must be specified for {compose_name} composition')

        transforms_list = Constructor._prepare_transforms_recursively(transforms)
        transform = TRANSFORMS.get(compose_name)(transforms=transforms_list, **kwargs)

        return transform

    @staticmethod
    def _create_transforms(transforms_params: ListConfig) -> Optional[A.Compose]:
        if transforms_params is None:
            return None

        return Constructor._prepare_base_compose('Compose', transforms=transforms_params)

    def configure_metrics_manager(self):
        """Create list of metrics wrapping them into a MetricManager module.

        Returns: MetricManager module.
        """
        return MetricsManager(self._hparams.metrics)

    def configure_losses(self) -> JointLoss:
        """Create list of loss modules wrapping them into a JointLoss module.

        Returns: JointLoss module
        """
        loss_modules, mappings, tags, weights = [], [], [], []
        for loss_config in self._hparams.joint_loss.losses:
            loss_module = LOSSES.get(loss_config.name)(**loss_config.params)
            loss_modules.append(loss_module)
            mappings.append(loss_config.mapping)
            tags.append(loss_config.tag)
            weights.append(loss_config.weight)

        normalize_weights = self._hparams.joint_loss.normalize_weights

        return JointLoss(loss_modules, mappings, tags, weights, normalize_weights)

    @property
    def hparams(self) -> DictConfig:
        """Return configuration dictionary.

        Returns:
            Dict, containing at least the following parameter groups (see configuration for details):
                - optimization
                - data
                - losses
                - metrics
        """
        return self._hparams
