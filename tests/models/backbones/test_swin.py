import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


class TestSwin(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    def test_forward_output_shape(self):
        backbone_name = 'swinv2_tiny_window8_256'
        swin = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3).to(device=self.device).eval()
        output = swin(self.input)
        answer = (2, 768, 8, 8)
        self.assertTupleEqual(output.shape, answer)
        torch.cuda.empty_cache()

    def test_load_pretrained(self):
        backbone_name = 'swinv2_tiny_window8_256'
        BACKBONES.get(backbone_name)(pretrained=True, in_channels=3)

    def test_forward_feature_output_shape(self):
        backbone_name = 'swinv2_tiny_window16_256'
        swin = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3).to(device=self.device).eval()
        features = swin.forward_features(self.input)
        feature_shapes = [f.shape for f in features]
        answer = [(2, 3, 256, 256), (2, 96, 64, 64),
                  (2, 192, 32, 32), (2, 384, 16, 16), (2, 768, 8, 8)]
        self.assertListEqual(feature_shapes, answer)
        torch.cuda.empty_cache()

    @parameterized.expand(BACKBONES.list_models(module='swin'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        x = torch.rand(2, 3, *model.img_size, device=self.device)
        with torch.no_grad():
            torch.jit.trace(model, x)
        torch.cuda.empty_cache()
