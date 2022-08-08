from torchok.tasks.base import BaseTask
from torchok.tasks.classification import ClassificationTask
from torchok.tasks.onnx import ONNXTask
from torchok.tasks.segmentation import SegmentationTask

# from src.tasks.moby import
# from src.tasks.segmentation import 
# from src.tasks.representation import 

__all__ = [
    'BaseTask',
    'ClassificationTask',
    'SegmentationTask',
    'ONNXTask',
]
