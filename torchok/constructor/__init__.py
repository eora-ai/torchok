from torchok.constructor.registry import Registry


DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
LOSSES = Registry('losses')
METRICS = Registry('metrics')
CALLBACKS = Registry('callbacks')
TASKS = Registry('tasks')
BACKBONES = Registry('backbones')
POOLINGS = Registry('poolings')
HEADS = Registry('heads')
NECKS = Registry('necks')
DETECTION_NECKS = Registry('detection_necks')
