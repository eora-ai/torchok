import unittest

import albumentations as A
import numpy as np
from omegaconf import OmegaConf
from torchmetrics import Accuracy
import torch
from torch.nn import Conv2d, Module, Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.constructor import DATASETS, OPTIMIZERS, SCHEDULERS, TRANSFORMS, METRICS
from src.constructor.constructor import Constructor
from src.constructor.config_structure import Phase
from src.data.datasets.base import ImageDataset


OPTIMIZERS.register_class(Adam)
SCHEDULERS.register_class(OneCycleLR)
TRANSFORMS.register_class(A.Compose)
TRANSFORMS.register_class(A.OneOf)
TRANSFORMS.register_class(A.Blur)
TRANSFORMS.register_class(A.RandomCrop)
METRICS.register_class(Accuracy)


@DATASETS.register_class
class TestDataset(ImageDataset):
    def __getitem__(self, item: int) -> dict:
        pass

    def __len__(self) -> int:
        return 10


OPTIM_BASIC_HPARAMS = OmegaConf.create({
    'optimization': [
        {
            'optimizer': {
                'name': 'Adam',
                'params': {
                    'lr': 0.0001
                }
            },
            'scheduler': {
                'name': 'OneCycleLR',
                'params': {
                    'max_lr': 0.001,
                    'total_steps': 100000
                },
                'pl_params': {
                    'interval': 'step'
                }
            }
        }
    ]
})


DATASETS_BASIC_HPARAMS = OmegaConf.create({
    'data': {
        'train': [
            {
                'dataset': {
                    'name': 'TestDataset',
                    'transform': None,
                    'augment': None,
                    'params': {}
                },
                'dataloader': {
                    'batch_size': 8,
                    'num_workers': 4,
                    'drop_last': True,
                    'shuffle': True
                }
            }
        ]
    }
})


class TestConstructor(unittest.TestCase):
    def test_optimizers_configuration_when_module_is_passed(self):
        constructor = Constructor(OPTIM_BASIC_HPARAMS)
        module = Conv2d(16, 32, (3, 3))
        opt_sched_list = constructor.configure_optimizers(module)

        optimizer_params = opt_sched_list[0]['optimizer']
        self.assertEqual(optimizer_params.defaults, {
            'lr': 0.0001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False
        })

        self.assertEqual(len(optimizer_params.param_groups), 2)

    def test_optimizers_configuration_when_iterable_is_passed(self):
        constructor = Constructor(OPTIM_BASIC_HPARAMS)
        modules_list = [Conv2d(16, 32, (3, 3)), Conv2d(16, 64, (3, 3))]
        opt_sched_list = constructor.configure_optimizers(modules_list)

        optimizer_params = opt_sched_list[0]['optimizer']
        self.assertEqual(optimizer_params.defaults, {
            'lr': 0.0001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False
        })

        self.assertEqual(len(optimizer_params.param_groups), 4)

    def test_lr_scheduler_configuration_when_scheduler_specified(self):
        constructor = Constructor(OPTIM_BASIC_HPARAMS)
        module = Conv2d(16, 32, (3, 3))
        opt_sched_list = constructor.configure_optimizers(module)

        base_lrs = opt_sched_list[0]['lr_scheduler']['scheduler'].base_lrs
        np.testing.assert_allclose(base_lrs, [4e-5, 4e-5])

    def test_weight_decay_groups_in_optimizers_when_module_contains_bias(self):
        hparams = OPTIM_BASIC_HPARAMS.copy()
        hparams['optimization'][0]['optimizer']['params']['weight_decay'] = 1e-4
        constructor = Constructor(hparams)
        module = Conv2d(16, 32, (3, 3))
        opt_sched_list = constructor.configure_optimizers(module)

        param_groups = opt_sched_list[0]['optimizer'].param_groups
        weight_decays = [param_groups[i]['weight_decay'] for i in range(2)]
        np.testing.assert_allclose(weight_decays, [0.0, 1e-4])

    def test_weight_decay_groups_in_optimizers_when_module_contains_1d_tensor(self):
        hparams = OPTIM_BASIC_HPARAMS.copy()
        hparams['optimization'][0]['optimizer']['params']['weight_decay'] = 1e-4
        constructor = Constructor(hparams)

        class Module1dTensor(Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv2d(16, 32, (3, 3))
                self.param = Parameter(torch.ones(123))

        module = Module1dTensor()
        opt_sched_list = constructor.configure_optimizers(module)

        param_groups = opt_sched_list[0]['optimizer'].param_groups
        weight_decays = [param_groups[i]['weight_decay'] for i in range(2)]
        np.testing.assert_allclose(weight_decays, [0.0, 1e-4])

    def test_weight_decay_groups_in_optimizers_when_module_implements_no_weight_decay(self):
        hparams = OPTIM_BASIC_HPARAMS.copy()
        hparams['optimization'][0]['optimizer']['params']['weight_decay'] = 1e-4
        constructor = Constructor(hparams)

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
        self.assertEqual(len(param_groups[0]['params']), 3)
        self.assertEqual(len(param_groups[1]['params']), 0)
        np.testing.assert_allclose(weight_decays, [0.0, 1e-4])

    def test_dataloader_creation_when_dataloader_params_specified(self):
        constructor = Constructor(DATASETS_BASIC_HPARAMS)
        dataloaders = constructor.create_dataloaders('train')

        actual_params = {
            'batch_size': dataloaders[0].batch_size,
            'num_workers': dataloaders[0].num_workers,
            'drop_last': dataloaders[0].drop_last,
            'shuffle': dataloaders[0].sampler is not None
        }

        self.assertEqual(actual_params, DATASETS_BASIC_HPARAMS['data']['train'][0]['dataloader'])

    def test_transforms_when_transforms_specified(self):
        self.__test_transforms_augmentations('transform')

    def test_augmentations_when_augmentations_specified(self):
        self.__test_transforms_augmentations('augment')

    def __test_transforms_augmentations(self, kind):
        hparams = DATASETS_BASIC_HPARAMS.copy()
        hparams['data']['train'][0]['dataset'][kind] = OmegaConf.create([
            {
                'name': 'OneOf',
                'params': {
                    'transforms': [
                        {
                            'name': 'Blur',
                            'params': {
                                'blur_limit': 11
                            }
                        }, {
                            'name': 'RandomCrop',
                            'params': {
                                'width': 224,
                                'height': 224
                            }
                        }
                    ]
                }
            }
        ])

        constructor = Constructor(hparams)
        dataloaders = constructor.create_dataloaders('train')

        if kind == 'transform':
            actual_transforms = dataloaders[0].dataset.transform
        else:
            actual_transforms = dataloaders[0].dataset.augment

        self.assertEqual(actual_transforms.__class__.__name__, 'Compose')
        self.assertEqual(actual_transforms.transforms[0].__class__.__name__, 'OneOf')
        self.assertEqual(actual_transforms.transforms[0].transforms[0].__class__.__name__, 'Blur')
        self.assertEqual(actual_transforms.transforms[0].transforms[1].__class__.__name__, 'RandomCrop')

    def test_losses_creation_when_multiple_losses_specified(self):
        hparams = OmegaConf.create({
            'losses':{
                    'loss_params': [
                    {
                        'name': 'SmoothL1Loss',
                        'params': {
                            'beta': 0.5
                        },
                        'tag': 'representation',
                        'mapping': {
                            'input': 'emb_student',
                            'target': 'emb_teacher'
                        },
                        'weight': 0.3
                    }, {
                        'name': 'CrossEntropyLoss',
                        'params': {},
                        'tag': 'classification',
                        'mapping': {
                            'input': 'logits',
                            'target': 'target'
                        },
                        'weight': 0.7
                    }
                ],
                'normalize_weights': True
            }
        })

        constructor = Constructor(hparams)
        losses = constructor.configure_losses()

        self.assertEqual(losses._JointLoss__losses[0].__class__.__name__, 'SmoothL1Loss')
        self.assertEqual(losses._JointLoss__losses[1].__class__.__name__, 'CrossEntropyLoss')
        self.assertEqual(losses._JointLoss__mappings, [
            {
                'input': 'emb_student',
                'target': 'emb_teacher'
            }, {
                'input': 'logits',
                'target': 'target'
            }
        ])
        self.assertEqual(losses._JointLoss__tags, ['representation', 'classification'])
        self.assertEqual(losses._JointLoss__weights, [0.3, 0.7])
        self.assertEqual(losses['representation'].beta, 0.5)

    def test_metrics_manager_creation_when_multiple_metrics_specified(self):
        hparams = OmegaConf.create({
            'metrics': [
                {
                    'name': 'Accuracy',
                    'mapping': {
                        'input': 'logits',
                        'target': 'target'
                    },
                    'params':  {},
                    'prefix': 'Cls',
                    'phases': [Phase.TRAIN]
                }
            ]
        })

        constructor = Constructor(hparams)
        metric_manager = constructor.configure_metrics_manager()
        phase2metrics = metric_manager.phase2metrics
        mapping = dict(input='logits', target='target')

        self.assertEqual(list(phase2metrics.keys())[0], Phase.TRAIN.name)
        self.assertEqual(phase2metrics[Phase.TRAIN.name][0].log_name, 'Cls_Accuracy')
        self.assertEqual(phase2metrics[Phase.TRAIN.name][0].mapping, mapping)
