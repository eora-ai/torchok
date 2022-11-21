from torch.nn import (BCELoss, BCEWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, CTCLoss, GaussianNLLLoss,
                      HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss, MarginRankingLoss, MSELoss,
                      MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, NLLLoss, NLLLoss2d,
                      PoissonNLLLoss, SmoothL1Loss, SoftMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss)
from torch.nn import Identity

import torchok.losses.detection
import torchok.losses.segmentation
import torchok.losses.representation
import torchok.losses.classification
from torchok.constructor import LOSSES

LOSSES.register_class(L1Loss)
LOSSES.register_class(NLLLoss)
LOSSES.register_class(NLLLoss2d)
LOSSES.register_class(PoissonNLLLoss)
LOSSES.register_class(GaussianNLLLoss)
LOSSES.register_class(KLDivLoss)
LOSSES.register_class(MSELoss)
LOSSES.register_class(BCELoss)
LOSSES.register_class(HingeEmbeddingLoss)
LOSSES.register_class(MultiLabelMarginLoss)
LOSSES.register_class(SmoothL1Loss)
LOSSES.register_class(HuberLoss)
LOSSES.register_class(SoftMarginLoss)
LOSSES.register_class(CrossEntropyLoss)
LOSSES.register_class(MultiLabelSoftMarginLoss)
LOSSES.register_class(CosineEmbeddingLoss)
LOSSES.register_class(MarginRankingLoss)
LOSSES.register_class(MultiMarginLoss)
LOSSES.register_class(TripletMarginLoss)
LOSSES.register_class(TripletMarginWithDistanceLoss)
LOSSES.register_class(CTCLoss)
LOSSES.register_class(Identity)
