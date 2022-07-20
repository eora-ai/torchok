import unittest

import timm
import torch

from src.constructor import BACKBONES


class TestMobileNetV3(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(2, 3, 224, 224)
        self._model = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        self._output = {}


class TestMobileNetV3Small(TestMobileNetV3):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('mobilenet_v3_small', methodName)

    def test_shape(self):
        x = self._model(self._input)
        self.assertTupleEqual(x.shape, (2, 576, 7, 7))


class TestMobileNetV3Large(TestMobileNetV3):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('mobilenet_v3_large', methodName)

    def test_shape(self):
        x = self._model(self._input)
        self.assertTupleEqual(x.shape, (2, 960, 7, 7))
