import unittest

import torch

from torchok import BACKBONES, NECKS


class TestHRNetNeck(unittest.TestCase):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 224, 224, device=self.device)
        self.backbone = BACKBONES.get('hrnet_w18')(pretrained=False).to(self.device)

    def test_classification_output_shape(self):
        neck = NECKS.get('HRNetClassificationNeck')(self.backbone.out_channels).to(self.device)
        x = self.backbone(self.input)
        x = neck(x)
        self.assertTupleEqual(x.shape, (2, 2048, 7, 7))

    def test_segmentation_output_shape(self):
        neck = NECKS.get('HRNetSegmentationNeck')(self.backbone.out_encoder_channels).to(self.device)
        x = self.backbone.forward_features(self.input)
        input_image, features = neck(x)
        self.assertTupleEqual(features.shape, (2, 270, 56, 56))
        self.assertTupleEqual(input_image.shape, (2, 3, 224, 224))
