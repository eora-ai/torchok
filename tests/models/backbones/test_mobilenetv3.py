import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    def test_load_pretrained(self):
        model = BACKBONES.get('mobilenetv3_small_050')(pretrained=True).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()

    def test_forward_feature_output_shape(self):
        model = BACKBONES.get('mobilenetv3_small_050')(pretrained=False, in_channels=3).to(device=self.device).eval()
        features = model.forward_features(self.input)
        feature_shapes = [f.shape for f in features]
        answer = [(2, 3, 64, 64), (2, 8, 16, 16), (2, 16, 8, 8),
                  (2, 24, 4, 4), (2, 288, 2, 2)]
        self.assertListEqual(feature_shapes, answer)
        torch.cuda.empty_cache()

    @parameterized.expand(BACKBONES.list_models(module='mobilenetv3'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()
