from .registry import Registry

OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
HEADS = Registry('head')
CLASSIFICATION_HEADS = Registry('classification_head')