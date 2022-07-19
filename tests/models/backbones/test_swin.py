import unittest

import torch

from src.constructor import BACKBONES


class TestSwin(unittest.TestCase):

    x = torch.rand(4, 3, 256, 256)

    def test_forward_output_shape(self):
        backbone_name = 'swinv2_tiny_window8_256'
        swin = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        output = swin(self.x)
        answer = (4, 768, 8, 8)
        self.assertTupleEqual(output.shape, answer)

    def test_load_pretrained(self):
        backbone_name = 'swinv2_tiny_window8_256'
        BACKBONES.get(backbone_name)(pretrained=True, in_channels=3)

    def test_forward_feature_output_shape(self):
        backbone_name = 'swinv2_tiny_window16_256'
        swin = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        features = swin.forward_features(self.x)
        feature_shapes = [f.shape for f in features]
        answer = [(4, 3, 256, 256), (4, 96, 64, 64), (4, 96, 64, 64),
                  (4, 192, 32, 32), (4, 384, 16, 16), (4, 768, 8, 8)]
        self.assertListEqual(feature_shapes, answer)
