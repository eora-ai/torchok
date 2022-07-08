from src.models.base import (
    BaseModel,
    FeatureHooks,
    FeatureInfo,
    HookType
)
from src.models import backbones
from src.models import identity
from src.models import heads
from src.models import necks
from src.models import poolings

from src.constructor import HEADS, POOLINGS


HEADS.register_class(identity.Identity)
POOLINGS.register_class(identity.Identity)


__all__ = [
    'BaseModel',
    'FeatureHooks',
    'FeatureInfo',
    'HookType',
]
