import unittest

import torch

from torchok import BACKBONES, NECKS


class TestHRNetClassificationNeck(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)
        self.backbone = BACKBONES.get('hrnet_w18')(pretrained=False).to(self.device)
        self.neck = NECKS.get('HRNetClassificationNeck')(self.backbone.out_channels).to(self.device)

    def test_outputs_equals(self):
        x = self.backbone(self.input)
        x = self.neck(x)
        self.assertTupleEqual(x.shape, (2, 2048, 2, 2))
