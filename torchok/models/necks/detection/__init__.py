import importlib

has_mmcv = importlib.util.find_spec("mmcv")
if has_mmcv is not None:
    from torchok.models.necks.detection import fpn
    from torchok.models.necks.detection import mmdet_necks
