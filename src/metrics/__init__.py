# from .classification import *
# from .segmentation import *
# from .representation import *
# from .metrics_manager import *

from src.constructor import METRICS
from torchmetrics import Accuracy
METRICS.register_class(Accuracy)