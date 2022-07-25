from torch.optim.lr_scheduler import (
    LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR,
    CosineAnnealingWarmRestarts
)

from torchok.constructor import SCHEDULERS
from . import knee_lr_scheduler

SCHEDULERS.register_class(LambdaLR)
SCHEDULERS.register_class(MultiplicativeLR)
SCHEDULERS.register_class(StepLR)
SCHEDULERS.register_class(MultiStepLR)
SCHEDULERS.register_class(ExponentialLR)
SCHEDULERS.register_class(CosineAnnealingLR)
SCHEDULERS.register_class(ReduceLROnPlateau)
SCHEDULERS.register_class(CyclicLR)
SCHEDULERS.register_class(OneCycleLR)
SCHEDULERS.register_class(CosineAnnealingWarmRestarts)
