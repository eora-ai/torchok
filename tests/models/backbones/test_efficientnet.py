import unittest

import timm
import torch

from src.constructor import BACKBONES


class TestEfficientNet(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self._timm_model = timm.create_model(backbone_name, pretrained=False, in_chans=3)
        self._model = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
        self._input_hook = {}

    def get_output(self, name):
        def hook(model, input, output):
            self._input_hook[name] = output.shape
        return hook


class TestEfficientNetB1(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b1', methodName)

    def test_outputs_equals(self):
        self._timm_model.conv_head.register_forward_hook(self.get_output('conv_head'))
        self._timm_model(self._input)
        self.assertTupleEqual(self._model(self._input).shape, self._input_hook['conv_head'])


class TestEfficientNetB4(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b4', methodName)

    def test_outputs_equals(self):
        self._timm_model.conv_head.register_forward_hook(self.get_output('conv_head'))
        self._timm_model(self._input)
        self.assertTupleEqual(self._model(self._input).shape, self._input_hook['conv_head'])


class TestEfficientNetB7(TestEfficientNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('efficientnet_b7', methodName)

    def test_outputs_equals(self):
        self._timm_model.conv_head.register_forward_hook(self.get_output('conv_head'))
        self._timm_model(self._input)
        self.assertTupleEqual(self._model(self._input).shape, self._input_hook['conv_head'])
