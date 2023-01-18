import importlib
import warnings

from torchok.constructor import (BACKBONES, DATASETS, HEADS, LOSSES, METRICS, NECKS,
                                 OPTIMIZERS, POOLINGS, SCHEDULERS, TASKS, TRANSFORMS)
from torchok import callbacks
from torchok import constructor
from torchok import data
from torchok import losses
from torchok import metrics
from torchok import models
from torchok import optim
from torchok import tasks

has_mmcv = importlib.util.find_spec("mmcv")
if has_mmcv is None:
    warnings.warn("MMCV is not installed therefore blocks based on MMDet code won't be added in the registry. "
                  "Install it with openmim and command `mim install mmcv-full`.")
