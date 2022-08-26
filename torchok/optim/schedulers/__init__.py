from torch.optim.lr_scheduler import (
    LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR,
    CosineAnnealingWarmRestarts
)

from timm.scheduler import (
    CosineLRScheduler, MultiStepLRScheduler, PlateauLRScheduler,
    PolyLRScheduler, StepLRScheduler, TanhLRScheduler
)

from torchok.constructor import SCHEDULERS

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

SCHEDULERS.register_class(CosineLRScheduler)
SCHEDULERS.register_class(MultiStepLRScheduler)
SCHEDULERS.register_class(PlateauLRScheduler)
SCHEDULERS.register_class(PolyLRScheduler)
SCHEDULERS.register_class(StepLRScheduler)
SCHEDULERS.register_class(TanhLRScheduler)
