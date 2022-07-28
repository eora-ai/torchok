import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    def test_load_pretrained(self):
        model = BACKBONES.get('hrnet_w18')(pretrained=True).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()

    @parameterized.expand(BACKBONES.list_models(module='hrnet'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()


