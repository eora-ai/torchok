import sys
sys.path.append('../../')

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import albumentations as A
import numpy as np
from torchmetrics import Accuracy
import torch
from torch.nn import Conv2d, Module, Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from dataclasses import dataclass, field
from enum import Enum
from typing import *

from src.constructor.config_structure import ConfigParams, Phase
from src.constructor.constructor import Constructor
from src.constructor import DATASETS, OPTIMIZERS, SCHEDULERS, TRANSFORMS, METRICS
from src.data.datasets.base import ImageDataset


OPTIMIZERS.register_class(Adam)
SCHEDULERS.register_class(OneCycleLR)
TRANSFORMS.register_class(A.Compose)
TRANSFORMS.register_class(A.OneOf)
TRANSFORMS.register_class(A.Blur)
TRANSFORMS.register_class(A.RandomCrop)
METRICS.register_class(Accuracy)


def is_equal(value, answer, test_name=''):
    if value == answer:
        print('OK')
    else:
        print('Failed')


@DATASETS.register_class
class TrainDataset(ImageDataset):
    def __getitem__(self, item: int) -> dict:
        pass

    def __len__(self) -> int:
        return 10


@hydra.main(config_path="configs/", config_name="config_for_constructor_test.yaml")
def test_constructor_with_config_structure_when_full_config_was_define(config):
    config_params = ConfigParams(**config)
    # create constructor with its modules
    constructor = Constructor(config_params)
    # Check test like in Constructor, because config was create from constructor test data
    print('Optimizer!')

    print('test_optimizers_configuration_when_module_is_passed')
    module = Conv2d(16, 32, (3, 3))
    opt_sched_list = constructor.configure_optimizers(module)
    optimizer_params = opt_sched_list[0]['optimizer']
    is_equal(optimizer_params.defaults, {
            'lr': 0.0001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0.0001,
            'amsgrad': False
        })
    
    print('test_optimizers_configuration_when_iterable_is_passed')
    modules_list = [Conv2d(16, 32, (3, 3)), Conv2d(16, 64, (3, 3))]
    opt_sched_list = constructor.configure_optimizers(modules_list)
    optimizer_params = opt_sched_list[0]['optimizer']
    is_equal(len(optimizer_params.param_groups), 4)

    print('test_lr_scheduler_configuration_when_scheduler_specified')
    module = Conv2d(16, 32, (3, 3))
    opt_sched_list = constructor.configure_optimizers(module)

    base_lrs = opt_sched_list[0]['lr_scheduler']['scheduler'].base_lrs
    try:
        np.testing.assert_allclose(base_lrs, [4e-5, 4e-5])
        print('OK')
    except:
        print('Failed')

    print('test_weight_decay_groups_in_optimizers_when_module_contains_bias')
    module = Conv2d(16, 32, (3, 3))
    opt_sched_list = constructor.configure_optimizers(module)

    param_groups = opt_sched_list[0]['optimizer'].param_groups
    weight_decays = [param_groups[i]['weight_decay'] for i in range(2)]
    try:
        np.testing.assert_allclose(weight_decays, [0.0, 1e-4])
        print('OK')
    except:
        print('Failed')

    print('test_weight_decay_groups_in_optimizers_when_module_contains_1d_tensor')
    class Module1dTensor(Module):
        def __init__(self):
            super().__init__()
            self.conv = Conv2d(16, 32, (3, 3))
            self.param = Parameter(torch.ones(123))

    module = Module1dTensor()
    opt_sched_list = constructor.configure_optimizers(module)

    param_groups = opt_sched_list[0]['optimizer'].param_groups
    weight_decays = [param_groups[i]['weight_decay'] for i in range(2)]
    try:
        np.testing.assert_allclose(weight_decays, [0.0, 1e-4])
        print('OK')
    except:
        print('Failed')

    print('test_weight_decay_groups_in_optimizers_when_module_implements_no_weight_decay')
    class Module1dTensor(Module):
        def __init__(self):
            super().__init__()
            self.conv = Conv2d(16, 32, (3, 3))
            self.param = Parameter(torch.ones(123))

        def no_weight_decay(self):
            return ['conv.weight']

    module = Module1dTensor()
    opt_sched_list = constructor.configure_optimizers(module)

    param_groups = opt_sched_list[0]['optimizer'].param_groups
    weight_decays = [param_groups[i]['weight_decay'] for i in range(2)]

    # no weight decay group should have 3 parameters, while weight decay group should be empty
    is_correct_len_groups = len(param_groups[0]['params']) ==  3 and len(param_groups[1]['params']) ==  0
    if is_correct_len_groups:
        try:
            np.testing.assert_allclose(weight_decays, [0.0, 1e-4])
            print('OK')
        except:
            print('Failed')
    else:
        print('Failed group length')


    # Dataloader
    print('Dataloader')
    
    print('test_dataloader_creation_when_dataloader_params_specified')
    dataloaders = constructor.create_dataloaders(Phase.TRAIN)
    actual_params = {
            'batch_size': dataloaders[0].batch_size,
            'num_workers': dataloaders[0].num_workers,
            'drop_last': dataloaders[0].drop_last,
            'shuffle': dataloaders[0].sampler is not None
        }
    is_equal(actual_params, config_params.data[Phase.TRAIN][0].dataloader)

    print('test_transforms_when_transforms_specified')
    dataloaders = constructor.create_dataloaders(Phase.TRAIN)
    actual_transforms = dataloaders[0].dataset.transform
    actual_list = [
        actual_transforms.__class__.__name__,
        actual_transforms.transforms[0].__class__.__name__,
        actual_transforms.transforms[0].transforms[0].__class__.__name__,
        actual_transforms.transforms[0].transforms[1].__class__.__name__
    ]
    answer_list = ['Compose', 'OneOf', 'Blur', 'RandomCrop']
    is_equal(actual_list, answer_list)
    

    # Losses
    print('Losses')

    print('test_losses_creation_when_multiple_losses_specified')
    losses = constructor.configure_losses()

    print('Class names')
    actual_list = [
        losses._JointLoss__losses[0].__class__.__name__,
        losses._JointLoss__losses[1].__class__.__name__
    ]
    answer_list = ['SmoothL1Loss', 'CrossEntropyLoss']
    is_equal(actual_list, answer_list)

    print('Mapping')
    is_equal(losses._JointLoss__mappings, [
            {
                'input': 'emb_student',
                'target': 'emb_teacher'
            }, {
                'input': 'logist',
                'target': 'target'
            }
        ])

    print('Tags')
    is_equal(losses._JointLoss__tags, ['representation', 'classification'])

    print('Weights')
    is_equal(losses._JointLoss__weights, [0.3, 0.7])

    print('Loss representation beta param')
    is_equal(losses['representation'].beta, 0.5)


    # Metrics
    print('Metrics')
    
    print('test_metrics_manager_creation_when_multiple_metrics_specified')
    metric_manager = constructor.configure_metrics_manager()
    phase2metrics = metric_manager.phase2metrics

    print('Phases name')
    is_equal(list(phase2metrics.keys())[0], Phase.TRAIN.name)

    print('Mapping')
    mapping = dict(input='logits', target='target')
    is_equal(phase2metrics[Phase.TRAIN.name][0].mapping, mapping)

    print('Log name')
    is_equal(phase2metrics[Phase.TRAIN.name][0].log_name, 'Cls_Accuracy')


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config_for_constructor_test", node=ConfigParams)
    test_constructor_with_config_structure_when_full_config_was_define()
