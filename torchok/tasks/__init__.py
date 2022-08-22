from torchok.tasks.base import BaseTask
from torchok.tasks.classification import ClassificationTask
from torchok.tasks.onnx import ONNXTask
from torchok.tasks.segmentation import SegmentationTask


__all__ = [
    'BaseTask',
    'ClassificationTask',
    'SegmentationTask',
    'ONNXTask',
]
