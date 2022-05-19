import unittest

import timm
import torch

from src.constructor import BACKBONES


class TestResNet18(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.__input = torch.ones(1, 3, 224, 224)
        self.__timm_model = timm.create_model('resnet18', pretrained=True, in_chans=3)
        self.__output = {}

    def test_init(self):
        self.__timm_model.layer4[-1].act2.register_forward_hook(self.get_output('output_last_layer'))
        self.__timm_model(self.__input)
        self.__output['output_last_layer']
        self.__model = BACKBONES.get('resnet18')(pretrained=True, in_chans=3)
        self.assertTrue(self.__model(self.__input).equal(self.__output['output_last_layer']))

    def get_output(self, name):
        def hook(model, input, output):
            self.__output[name] = output.detach()
        return hook


class TestResNet50(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.__input = torch.ones(1, 3, 224, 224)
        self.__timm_model = timm.create_model('resnet50', pretrained=True)
        self.__output = {}

    def test_init(self):
        self.__timm_model.layer4[-1].act3.register_forward_hook(self.get_output('output_last_layer'))
        self.__timm_model(self.__input)
        self.__output['output_last_layer']
        self.__model = BACKBONES.get('resnet50')(pretrained=True, in_chans=3)
        self.assertTrue(self.__model(self.__input).equal(self.__output['output_last_layer']))

    def get_output(self, name):
        def hook(model, input, output):
            self.__output[name] = output.detach()
        return hook
