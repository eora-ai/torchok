from albumentations import *
from albumentations.pytorch import *
from albumentations import BasicTransform
from albumentations.core.composition import Compose

from .pixelwise import *
from .spatial import *

from torchok.constructor import TRANSFORMS


TRANSFORMS.register_class(Compose)
TRANSFORMS.register_class(OneOf)
TRANSFORMS.register_class(ToTensorV2)
TRANSFORMS.register_class(Normalize)
TRANSFORMS.register_class(Resize)
