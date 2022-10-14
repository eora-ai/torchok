import importlib

import torchok.models.backbones.beit
import torchok.models.backbones.davit
import torchok.models.backbones.efficientnet
import torchok.models.backbones.hrnet
import torchok.models.backbones.mobilenetv3
import torchok.models.backbones.resnet
import torchok.models.backbones.swin
from torchok.models.backbones.base_backbone import BackboneWrapper, BaseBackbone

has_mmcv = importlib.util.find_spec("mmcv")
if has_mmcv is not None:
    import torchok.models.backbones.mmdet_backbones
