from torch.nn import (
    L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, CosineEmbeddingLoss, SmoothL1Loss,
    CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss, MultiMarginLoss, PoissonNLLLoss, SoftMarginLoss,
    TripletMarginLoss, TripletMarginWithDistanceLoss, AdaptiveLogSoftmaxWithLoss
)

from pytorch_metric_learning.losses import (
    AngularLoss, ArcFaceLoss, CentroidTripletLoss, CircleLoss, ContrastiveLoss,
    CosFaceLoss, CrossBatchMemory, FastAPLoss, GenericPairLoss, 
    GeneralizedLiftedStructureLoss, IntraPairVarianceLoss, LargeMarginSoftmaxLoss,
    LiftedStructureLoss, MarginLoss, MultiSimilarityLoss, NCALoss, NormalizedSoftmaxLoss,
    NPairsLoss, NTXentLoss, ProxyAnchorLoss, ProxyNCALoss, SignalToNoiseRatioContrastiveLoss,
    SoftTripleLoss, SphereFaceLoss, SubCenterArcFaceLoss, SupConLoss, VICRegLoss,
    WeightRegularizerMixin

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
# pytorch_metric_learning losses BACKEND IS  
# def forward(
#   self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None
#   ):
#   Args:
#       embeddings: tensor of size (batch_size, embedding_size)
#       labels: tensor of size (batch_size)
#       indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
#           or size 4 for pairs (anchor1, postives, anchor2, negatives)
#           Can also be left as None
#       Returns: the loss
LOSSES.register_class(AngularLoss)
LOSSES.register_class(ArcFaceLoss)
LOSSES.register_class(CentroidTripletLoss)
LOSSES.register_class(CircleLoss)
LOSSES.register_class(ContrastiveLoss)
LOSSES.register_class(CosFaceLoss)
LOSSES.register_class(CrossBatchMemory)
LOSSES.register_class(FastAPLoss)
LOSSES.register_class(GenericPairLoss)
LOSSES.register_class(GeneralizedLiftedStructureLoss)
LOSSES.register_class(IntraPairVarianceLoss)
LOSSES.register_class(LargeMarginSoftmaxLoss)
LOSSES.register_class(LiftedStructureLoss)
LOSSES.register_class(MarginLoss)
LOSSES.register_class(MultiSimilarityLoss)
LOSSES.register_class(NCALoss)
LOSSES.register_class(NormalizedSoftmaxLoss)
LOSSES.register_class(NPairsLoss)
LOSSES.register_class(NTXentLoss)
LOSSES.register_class(ProxyAnchorLoss)
LOSSES.register_class(ProxyNCALoss)
LOSSES.register_class(SignalToNoiseRatioContrastiveLoss)
LOSSES.register_class(SoftTripleLoss)
LOSSES.register_class(SphereFaceLoss)
LOSSES.register_class(SubCenterArcFaceLoss)
LOSSES.register_class(SupConLoss)
LOSSES.register_class(VICRegLoss)
LOSSES.register_class(WeightRegularizerMixin)
