from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Union

import albumentations as A
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.nn import Module, Parameter
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
        self.__hparams = hparams

    def configure_optimizers(self, parameters: Union[Module, Tensor, List[Union[Module, Tensor]]],
                             optim_idx: int = -1) -> List[Dict[str, Union[Optimizer, Dict[str, Any]]]]:
        """Create optimizers and learning rate schedulers from a pre-defined configuration.

        Note: optimization parameters are split into two groups: decay and no-decay specifying which parameters can be
        weight decayed and which not. Each module is checked on having an attribute `no_weight_decay`,
        specifying a list of submodules which must be not weight decayed (useful in transformer models such as Swin).
        All *.bias parameters, 1D tensors and scalars are put into no-decay group according to the best practice.

        Args:
            parameters: Parameters for optimization
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
        optims_params = self.__hparams.optimization
        if 0 <= optim_idx < len(optims_params):
            optims_params = [optims_params[optim_idx]]
        elif optim_idx >= len(optims_params):
            raise ValueError(f'You requested optimization with index {optim_idx} while '
                             f'there\'re only {len(optims_params)} optimization parameters are specified')

        opt_sched_list = []
        for optim_params in optims_params:
            optimizer = self.__create_optimizer(parameters, optim_params.optimizer)
            opt_sched = {'optimizer': optimizer}

            if optim_params.scheduler is not None:
                scheduler_dict = self.__create_scheduler(optimizer, optim_params.scheduler)
                opt_sched['lr_scheduler'] = scheduler_dict

            opt_sched_list.append(opt_sched)

        return opt_sched_list

    @staticmethod
    def __create_optimizer(parameters: Union[Module, Tensor, List[Union[Module, Tensor]]],
                           optimizer_params: DictConfig) -> Optimizer:
        optimizer_class = OPTIMIZERS.get(optimizer_params.name)
        parameters = Constructor.__set_weight_decay_for_parameters(parameters)
        optimizer = optimizer_class(parameters, **optimizer_params.params)

        return optimizer

    @staticmethod
    def __create_scheduler(optimizer: Optimizer, scheduler_params: DictConfig) -> Dict[str, Any]:
        scheduler_class = SCHEDULERS.get(scheduler_params.name)
        scheduler = scheduler_class(optimizer, **scheduler_params.params)
        pl_params = scheduler_params.pl_params

        return {
            'scheduler': scheduler,
            **pl_params
        }

    @staticmethod
    def __set_weight_decay_for_parameters(parameters: Union[Module, Tensor, List[Union[Module, Tensor]]]) -> List[
            Dict[str, Union[Tensor, float]]]:
        if not isinstance(parameters, Iterable) and not isinstance(parameters, Module) and \
           not isinstance(parameters, Tensor):
            raise ValueError(f'Unsupported parameters type for optimizer: {type(parameters)}')
        elif not isinstance(parameters, Iterable):
            parameters = [parameters]

        param_groups = []
        for model in parameters:
            if isinstance(model, Module):
                param_groups.extend(Constructor.__param_groups_weight_decay(model))
            elif isinstance(model, Tensor) and model.requires_grad:
                param_groups.append({'params': model})

        return param_groups

    # Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
    # Copyright 2019 Ross Wightman
    # Licensed under The Apache 2.0 License [see LICENSE for details]
    @staticmethod
    def __param_groups_weight_decay(model: Module) -> List[Dict[str, Union[Parameter, float]]]:
        # TODO: add description of no_weight_decay method into BaseModel
        no_weight_decay_list = []
        if hasattr(model, 'no_weight_decay'):
            no_weight_decay_list = model.no_weight_decay()

        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith('.bias') or name in no_weight_decay_list:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay}]

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
            return [
                self.__prepare_dataloader(phase_params.dataset, phase_params.dataloader)
                for phase_params in self.hparams.data[phase] if phase_params is not None
            ]
        else:
            return []

    @staticmethod
    def __prepare_dataloader(dataset_params: DictConfig, dataloader_params: DictConfig) -> DataLoader:
        dataset = Constructor.__create_dataset(dataset_params)
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None

        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn,
                            **dataloader_params)

        return loader

    @staticmethod
    def __create_dataset(dataset_params: DictConfig) -> ImageDataset:
        transform = Constructor.__create_transforms(dataset_params.transform)
        # TODO remove when OmegaConf is fixing the bug, write to issue to Omegaconf!
        # Config structure had 'augment' parameter with default value = None, but in loaded config 
        # 'augment' is not in keys of dataset_params dictionary. So it must be written like
        # augment_params = dataset_params.augment
        augment_params = dataset_params.get('augment', None)
        augment = Constructor.__create_transforms(augment_params)

        dataset_class = DATASETS.get(dataset_params.name)

        return dataset_class(transform=transform, augment=augment, **dataset_params.params)

    @staticmethod
    def __prepare_transforms_recursively(transforms: ListConfig) -> List[Union[A.Compose, A.BaseCompose]]:
        transforms_list = []

        for transform_info in transforms:
            transform_name = transform_info.name
            transform_params = transform_info.get('params', dict())

            if transform_name == 'Compose' or transform_name == 'OneOf' or transform_name == 'SomeOf' or \
               transform_name == 'PerChannel' or transform_name == 'Sequential':
                transform = Constructor.__prepare_base_compose(transform_name, **transform_params)
            elif transform_name == 'OneOrOther':
                raise ValueError('OneOrOther composition is currently not supported')
            else:
                transform = TRANSFORMS.get(transform_name)(**transform_params)

            transforms_list.append(transform)

        return transforms_list

    @staticmethod
    def __prepare_base_compose(compose_name: str, **kwargs) -> A.Compose:
        transforms = kwargs.pop('transforms', None)
        if transforms is None:
            raise ValueError(f'There are transforms must be specified for {compose_name} composition')

        transforms_list = Constructor.__prepare_transforms_recursively(transforms)
        transform = TRANSFORMS.get(compose_name)(transforms=transforms_list, **kwargs)

        return transform

    @staticmethod
    def __create_transforms(transforms_params: ListConfig) -> Optional[A.Compose]:
        if transforms_params is None:
            return None

        return Constructor.__prepare_base_compose('Compose', transforms=transforms_params)

    def configure_metrics_manager(self):
        """Create list of metrics wrapping them into a MetricManager module.

        Returns: MetricManager module.
        """
        return MetricsManager(self.__hparams.metrics)

    def configure_losses(self) -> JointLoss:
        """Create list of loss modules wrapping them into a JointLoss module.

        Returns: JointLoss module
        """
        loss_modules, mappings, tags, weights = [], [], [], []
        for loss_config in self.__hparams.joint_loss.losses:
            loss_module = LOSSES.get(loss_config.name)(**loss_config.params)
            loss_modules.append(loss_module)
            mappings.append(loss_config.mapping)
            tags.append(loss_config.tag)
            weights.append(loss_config.weight)

        normalize_weights = self.__hparams.joint_loss.normalize_weights

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
        return self.__hparams
