from torchok.models.base import (
    BaseModel,
    FeatureHooks,
    FeatureInfo,
    HookType
)
from torchok.models import backbones
from torchok.models import identity
from torchok.models import heads
from torchok.models import necks
from torchok.models import poolings

from torchok.constructor import HEADS, POOLINGS


HEADS.register_class(identity.Identity)
POOLINGS.register_class(identity.Identity)


__all__ = [
    'BaseModel',
    'FeatureHooks',
    'FeatureInfo',
    'HookType',
]
