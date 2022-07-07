import unittest

import torch

from src.constructor import BACKBONES


class TestDaViT(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
        self.backbone.load_state_dict(torch.load('/workdir/vpatrushev/small2/torchOK2/torchok/model_best.pth.tar')['state_dict'])


class TestDaViT(TestDaViT):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('davit_t', methodName)

    def test_outputs_equals(self):
        x = self.backbone(self._input)
        self.assertTupleEqual(x[3].shape, (1, 768, 7, 7))
