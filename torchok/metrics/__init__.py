import importlib

from torchmetrics.classification import (Accuracy, AUROC, AveragePrecision, CalibrationError,
                                         CohenKappa, ConfusionMatrix, F1Score, FBetaScore,
                                         HammingDistance, HingeLoss, JaccardIndex, MatthewsCorrCoef, Precision,
                                         PrecisionRecallCurve, Recall, ROC, Specificity, StatScores)  # noqa: E402
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

from torchmetrics.detection import MeanAveragePrecision


from torchok.constructor import METRICS
from torchok.metrics.metrics_manager import MetricsManager, MetricWithUtils

import torchok.metrics.index_base_metric
import torchok.metrics.representation_ranx
import torchok.metrics.representation_torchmetrics
import torchok.metrics.torchmetric_060


has_mmcv = importlib.util.find_spec("mmcv")
if has_mmcv is not None:
    import torchok.metrics.detection

METRICS.register_class(AUROC)
METRICS.register_class(ROC)
METRICS.register_class(Accuracy)
METRICS.register_class(AveragePrecision)
METRICS.register_class(CalibrationError)
METRICS.register_class(CohenKappa)
METRICS.register_class(ConfusionMatrix)
METRICS.register_class(F1Score)
METRICS.register_class(FBetaScore)
METRICS.register_class(HammingDistance)
METRICS.register_class(HingeLoss)
METRICS.register_class(JaccardIndex)
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

METRICS.register_class(MeanAveragePrecision)
