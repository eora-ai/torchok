from torchok.tasks.base import BaseTask
from torchok.tasks.classification import ClassificationTask
from torchok.tasks.onnx import ONNXTask
from torchok.tasks.segmentation import SegmentationTask
from torchok.tasks.pairwise import PairwiseLearnTask
from torchok.tasks.representation import RepresentationLearnTask


__all__ = [
    'BaseTask',
    'ClassificationTask',
    'SegmentationTask',
    'ONNXTask',
    'PairwiseLearnTask',
    'RepresentationLearnTask'
]
