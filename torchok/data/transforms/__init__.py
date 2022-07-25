from albumentations import *
from albumentations.pytorch import *
from albumentations import BasicTransform
from albumentations.core.composition import Compose

from torchok.data.transforms.pixelwise import *
from torchok.data.transforms.spatial import *

from torchok.constructor import TRANSFORMS


TRANSFORMS.register_class(Compose)
TRANSFORMS.register_class(OneOf)
TRANSFORMS.register_class(ToTensorV2)
TRANSFORMS.register_class(Normalize)
TRANSFORMS.register_class(Resize)
