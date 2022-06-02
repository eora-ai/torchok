from .classification import *
from .segmentation import *
from .representation import *
from .metrics_manager import *

from torchmetrics import (
    Accuracy, AveragePrecision, BinnedAveragePrecision,
    CohenKappa, Precision, Recall
)

from src.constructor import METRICS


METRICS.register_class(Accuracy)
METRICS.register_class(AveragePrecision)
METRICS.register_class(BinnedAveragePrecision)
METRICS.register_class(CohenKappa)
METRICS.register_class(Precision)
METRICS.register_class(Recall)
