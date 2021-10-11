from albumentations import *
from albumentations.pytorch import *
from albumentations import BasicTransform

# after adding extra custom classes import them like this and update local_transformations creation
from .pixelwise import *
from .spatial import *
