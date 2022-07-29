import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    def test_load_pretrained(self):
        model = BACKBONES.get('resnet18')(pretrained=True).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()

    def test_forward_feature_output_shape(self):
        model = BACKBONES.get('resnet18')(pretrained=False, in_channels=3).to(device=self.device).eval()
        features = model.forward_features(self.input)
        feature_shapes = [f.shape for f in features]
        answer = [(2, 3, 64, 64), (2, 64, 32, 32), (2, 64, 16, 16),
                  (2, 128, 8, 8), (2, 256, 4, 4), (2, 512, 2, 2)]
        self.assertListEqual(feature_shapes, answer)

    @parameterized.expand(BACKBONES.list_models(module='resnet'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()
