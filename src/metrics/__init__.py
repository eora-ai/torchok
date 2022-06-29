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
    HammingDistance,
    KLDivergence,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
    StatScores,
)

from torchmetrics.regression import (  # noqa: E402
    CosineSimilarity,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
)

from src.metrics.classification import *
from src.metrics.segmentation import *
from src.metrics.representation import (
    PrecisionAtKMeter,
    RecallAtKMeter,
    MeanAveragePrecisionAtKMeter,
    NDCGAtKMeter,
)
from src.metrics.metrics_manager import *

from src.constructor import METRICS
from src.metrics.metrics_manager import MetricsManager, MetricWithUtils

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
METRICS.register_class(HammingDistance)
METRICS.register_class(KLDivergence)
METRICS.register_class(Precision)
METRICS.register_class(PrecisionRecallCurve)
METRICS.register_class(Recall)
METRICS.register_class(Specificity)
METRICS.register_class(StatScores)

METRICS.register_class(CosineSimilarity)
METRICS.register_class(ExplainedVariance)
METRICS.register_class(MeanAbsoluteError)
METRICS.register_class(MeanAbsolutePercentageError)
METRICS.register_class(MeanSquaredError)
METRICS.register_class(MeanSquaredLogError)
METRICS.register_class(R2Score)
METRICS.register_class(SymmetricMeanAbsolutePercentageError)
METRICS.register_class(TweedieDevianceScore)


__all__ = [
    'MetricsManager',
    'MetricWithUtils',
    'HitAtKMeter',
    'PrecisionAtKMeter',
    'RecallAtKMeter',
    'MeanAveragePrecisionAtKMeter',
    'NDCGAtKMeter',
]
