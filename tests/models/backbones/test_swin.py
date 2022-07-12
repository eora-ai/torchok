import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.constructor import HEADS, NECKS, BACKBONES


class TestSwin(unittest.TestCase):

    x = torch.rand(4, 3, 256, 256)

    def test_forward_output_shape(self):
        backbone_name = 'swinv2_tiny_window8_256'
        swin = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
        output = swin(self.x)
        answer = (4, 768, 8, 8)
        self.assertTupleEqual(output.shape, answer)

    # def test_forward_feature_output_shape(self):
    #     backbone_name = 'swinv2_tiny_window16_256'
    #     swin = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
    #     features = swin.forward_features(self.x)
    #     feature_shapes = [f.shape for f in features]
    #     answer = [(4, 96, 64, 64), (4, 192, 32, 32), (4, 384, 16, 16), (4, 768, 8, 8)]
    #     self.assertListEqual(feature_shapes, answer)
    