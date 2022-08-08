from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS,
    RMSprop, Rprop, SGD, SparseAdam
)

from torchok.constructor import OPTIMIZERS

# TODO: add other fresh optimizers from PyTorch
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
