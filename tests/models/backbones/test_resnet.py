import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


def inp(bsize, in_ch, w, h):
    return torch.ones(bsize, in_ch, w, h)


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    @parameterized.expand(BACKBONES.list_models(module='resnet'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()
