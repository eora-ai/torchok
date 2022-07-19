import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.constructor import HEADS, NECKS, BACKBONES


class TestUnetSegmentation(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False)
        self.neck = NECKS.get('UnetNeck')(self.backbone._out_feature_channels[0])
        neck_features = self.neck.out_channels
        self.head = HEADS.get('UnetHead')(neck_features, 10, True, (224, 224))


class TestUnetHRNetBackbone(TestUnetSegmentation):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('resnet18', methodName)

    def test_outputs_equals(self):
        backbone_features = self.backbone.forward_features(self._input)
        x = self.neck(backbone_features)
        x = self.head(x)
        self.assertTupleEqual(x.shape, (1, 10, 224, 224))
