import unittest

import torch

from torchok import HEADS


class TestArcFaceHead(unittest.TestCase):
    def test_shape(self):
        in_features = 128
        num_classes = 10
        arcface = HEADS.get('ArcFaceHead')(in_features, num_classes)
        input = torch.rand((2, in_features))
        target = torch.tensor([[4], [8]])
        output = arcface(input, target)
        self.assertEqual(output.shape, (2, num_classes))

    def test_weight_shape(self):
        in_features = 128
        num_classes = 10
        arcface = HEADS.get('ArcFaceHead')(in_features, num_classes)
        self.assertEqual(arcface.weight.shape, (num_classes, in_features))
