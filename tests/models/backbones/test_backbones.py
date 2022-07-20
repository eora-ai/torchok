import unittest

import timm
import torch

from src.constructor import BACKBONES


class TestResNet(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self._timm_model = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        self._model = BACKBONES.get(backbone_name)(pretrained=True, in_channels=3)
        self._output = {}

    def get_output(self, name):
        def hook(model, input, output):
            self._output[name] = output.detach()
        return hook


class TestResNet18(TestResNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('resnet18', methodName)

    def test_outputs_equals(self):
        self._timm_model.layer4[-1].act2.register_forward_hook(self.get_output('output_last_layer'))
        self._timm_model(self._input)
        self.assertTrue(self._model(self._input).equal(self._output['output_last_layer']))

class TestResNet50(TestResNet):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('resnet50', methodName)

    def test_outputs_equals(self):
        self._timm_model.layer4[-1].act3.register_forward_hook(self.get_output('output_last_layer'))
        self._timm_model(self._input)
        self.assertTrue(self._model(self._input).equal(self._output['output_last_layer']))
