import unittest

import torch

from torchok import BACKBONES, HEADS, NECKS


class TestHRNetSegmentationHead(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 224, 224, device=self.device)
        self.backbone = BACKBONES.get('hrnet_w18')(pretrained=False).to(self.device)
        self.neck = NECKS.get('HRNetSegmentationNeck')(self.backbone.out_encoder_channels).to(self.device)
        self.head = HEADS.get('HRNetSegmentationHead')(self.neck.out_channels, 10).to(self.device)

    def test_output_shape(self):
        backbone_features = self.backbone.forward_features(self.input)
        x = self.neck(backbone_features)
        x = self.head(x)
        self.assertTupleEqual(x.shape, (2, 10, 224, 224))
