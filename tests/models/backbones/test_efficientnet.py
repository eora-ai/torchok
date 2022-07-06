import unittest

import torch

from src.constructor import BACKBONES


class TestEfficientNet(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self._model = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)



class TestEfficientNetB1(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b1', methodName)

    def test_outputs_equals(self):
        self._model.load_state_dict(torch.load('/workdir/vpatrushev/small2/efficientnet-b1_torchok.pth'))
        self.assertTupleEqual(self._model(self._input).shape, (1, 1280, 7, 7))


class TestEfficientNetB4(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b4', methodName)

    def test_outputs_equals(self):
        self._model.load_state_dict(torch.load('/workdir/vpatrushev/small2/efficientnet-b4_torchok.pth'))
        self.assertTupleEqual(self._model(self._input).shape, (1, 1792, 7, 7))


class TestEfficientNetB7(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b7', methodName)

    def test_outputs_equals(self):
        self._model.load_state_dict(torch.load('/workdir/vpatrushev/small2/efficientnet-b7_torchok.pth'))
        self.assertTupleEqual(self._model(self._input).shape, (1, 2560, 7, 7))
