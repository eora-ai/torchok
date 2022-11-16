import importlib

has_mmcv = importlib.util.find_spec("mmcv")
if has_mmcv is not None:
    import torchok.models.heads.detection.fcos
