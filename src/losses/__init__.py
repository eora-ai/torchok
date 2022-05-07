from torch.nn import (
    L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, CosineEmbeddingLoss, SmoothL1Loss,
    CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss, MultiMarginLoss, PoissonNLLLoss, SoftMarginLoss,
    TripletMarginLoss, TripletMarginWithDistanceLoss, AdaptiveLogSoftmaxWithLoss
)

from src.constructor import LOSSES
from .segmentation import dice, lovasz
from .representation import pairwise
from .common import cross_entropy, focal

LOSSES.register_class(L1Loss)
LOSSES.register_class(BCELoss)
LOSSES.register_class(CTCLoss)
LOSSES.register_class(MSELoss)
LOSSES.register_class(NLLLoss)
LOSSES.register_class(KLDivLoss)
LOSSES.register_class(SmoothL1Loss)
LOSSES.register_class(PoissonNLLLoss)
LOSSES.register_class(SoftMarginLoss)
LOSSES.register_class(MultiMarginLoss)
LOSSES.register_class(TripletMarginLoss)
LOSSES.register_class(MarginRankingLoss)
LOSSES.register_class(HingeEmbeddingLoss)
LOSSES.register_class(CosineEmbeddingLoss)
LOSSES.register_class(MultiLabelMarginLoss)
LOSSES.register_class(MultiLabelSoftMarginLoss)
LOSSES.register_class(AdaptiveLogSoftmaxWithLoss)
LOSSES.register_class(TripletMarginWithDistanceLoss)
