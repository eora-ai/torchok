from .registry import Registry


DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
BACKBONES = Registry('backbones')
