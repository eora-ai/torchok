from torchok.constructor.registry import Registry


DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
TASKS = Registry('tasks')
BACKBONES = Registry('backbones')
POOLINGS = Registry('poolings')
HEADS = Registry('heads')
NECKS = Registry('necks')


__all__ = [
    'DATASETS',
    'TRANSFORMS',
    'OPTIMIZERS',
    'SCHEDULERS',
    'LOSSES',
    'METRICS',
    'TASKS',
    'BACKBONES',
    'POOLINGS',
    'HEADS',
    'NECKS',
]
