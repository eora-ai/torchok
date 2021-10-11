from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS,
    RMSprop, Rprop, SGD, SparseAdam
)

from src.registry import OPTIMIZERS
from . import adafactor
from . import adahessian
from . import adamp
from . import lamb
from . import lars
from . import lookahead
from . import madgrad
from . import nadam
from . import novograd
from . import nvnovograd
from . import radam
from . import sgdp

OPTIMIZERS.register_class(Adadelta)
OPTIMIZERS.register_class(Adagrad)
OPTIMIZERS.register_class(Adam)
OPTIMIZERS.register_class(Adamax)
OPTIMIZERS.register_class(AdamW)
OPTIMIZERS.register_class(ASGD)
OPTIMIZERS.register_class(LBFGS)
OPTIMIZERS.register_class(RMSprop)
OPTIMIZERS.register_class(Rprop)
OPTIMIZERS.register_class(SGD)
OPTIMIZERS.register_class(SparseAdam)
