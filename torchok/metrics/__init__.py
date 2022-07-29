from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric  # noqa: E402
from torchmetrics.classification import (  # noqa: E402
    AUC,
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    BinnedAveragePrecision,
    BinnedPrecisionRecallCurve,
    BinnedRecallAtFixedPrecision,
    CalibrationError,
    CohenKappa,
    ConfusionMatrix,
    CoverageError,
    F1Score,
    FBetaScore,
    HammingDistance,
    HingeLoss,
    JaccardIndex,
    KLDivergence,
    LabelRankingAveragePrecision,
    LabelRankingLoss,
    MatthewsCorrCoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
    StatScores,
)
from torchmetrics.image import (  # noqa: E402
    ErrorRelativeGlobalDimensionlessSynthesis,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    SpectralAngleMapper,
    SpectralDistortionIndex,
    StructuralSimilarityIndexMeasure,
    UniversalImageQualityIndex,
)
from torchmetrics.regression import (  # noqa: E402
    CosineSimilarity,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
    WeightedMeanAbsolutePercentageError,
)

from torchok.metrics.classification import *
from torchok.metrics.segmentation import *
from torchok.metrics.representation import (
    PrecisionAtKMeter,
    RecallAtKMeter,
    MeanAveragePrecisionAtKMeter,
    NDCGAtKMeter,
)
from torchok.metrics.metrics_manager import *

from torchok.constructor import METRICS
from torchok.metrics.metrics_manager import MetricsManager, MetricWithUtils

METRICS.register_class(AUC)
METRICS.register_class(AUROC)
METRICS.register_class(ROC)
METRICS.register_class(Accuracy)
METRICS.register_class(AveragePrecision)
METRICS.register_class(BinnedAveragePrecision)
METRICS.register_class(BinnedPrecisionRecallCurve)
METRICS.register_class(BinnedRecallAtFixedPrecision)
METRICS.register_class(CalibrationError)
METRICS.register_class(CohenKappa)
METRICS.register_class(ConfusionMatrix)
METRICS.register_class(CoverageError)
METRICS.register_class(F1Score)
METRICS.register_class(FBetaScore)
METRICS.register_class(HammingDistance)
METRICS.register_class(HingeLoss)
METRICS.register_class(JaccardIndex)
METRICS.register_class(KLDivergence)
METRICS.register_class(LabelRankingAveragePrecision)
METRICS.register_class(LabelRankingLoss)
METRICS.register_class(MatthewsCorrCoef)
METRICS.register_class(Precision)
METRICS.register_class(PrecisionRecallCurve)
METRICS.register_class(Recall)
METRICS.register_class(Specificity)
METRICS.register_class(StatScores)

METRICS.register_class(ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register_class(MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register_class(PeakSignalNoiseRatio)
METRICS.register_class(SpectralAngleMapper)
METRICS.register_class(SpectralDistortionIndex)
METRICS.register_class(StructuralSimilarityIndexMeasure)
METRICS.register_class(UniversalImageQualityIndex)

METRICS.register_class(CosineSimilarity)
METRICS.register_class(ExplainedVariance)
METRICS.register_class(MeanAbsoluteError)
METRICS.register_class(MeanAbsolutePercentageError)
METRICS.register_class(MeanSquaredError)
METRICS.register_class(MeanSquaredLogError)
METRICS.register_class(PearsonCorrCoef)
METRICS.register_class(R2Score)
METRICS.register_class(SpearmanCorrCoef)
METRICS.register_class(SymmetricMeanAbsolutePercentageError)
METRICS.register_class(TweedieDevianceScore)
METRICS.register_class(WeightedMeanAbsolutePercentageError)


__all__ = [
    'MetricsManager',
    'MetricWithUtils',
    'PrecisionAtKMeter',
    'RecallAtKMeter',
    'MeanAveragePrecisionAtKMeter',
    'NDCGAtKMeter',
]
