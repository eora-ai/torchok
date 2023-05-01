import importlib

has_mmcv = importlib.util.find_spec("mmcv")
# ToDo: fix mmdet heads
# if has_mmcv is not None:
#     import torchok.models.heads.detection.fcos
#     import torchok.models.heads.detection.detr
