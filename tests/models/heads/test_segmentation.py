import unittest

import torch
from parameterized import parameterized
from torch.nn import Sequential

from torchok import BACKBONES, HEADS, NECKS
from torchok.models.backbones.base_backbone import BackboneWrapper

tested_heads = [
    'SegmentationHead',
    'OCRSegmentationHead'
]


class AbstractTestSegmentationPair:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 10

    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    def create_model(self, head_name):
        raise NotImplementedError()

    def test_forward_output_shape(self, head_name):
        model = self.create_model(head_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertTupleEqual(output.shape, (2, self.num_classes, 256, 256))
        torch.cuda.empty_cache()

    def test_torchscript_conversion(self, head_name):
        model = self.create_model(head_name)
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()


class TestUnet(AbstractTestSegmentationPair, unittest.TestCase):

    def create_model(self, head_name):
        backbone = BACKBONES.get('resnet18')(pretrained=False, in_channels=3)
        backbone = BackboneWrapper(backbone)
        encoder_channels = backbone.out_encoder_channels
        decoder_channels = (512, 256, 128, 64, 64)
        if len(encoder_channels) < len(decoder_channels):
            decoder_channels = decoder_channels[:len(encoder_channels)]
        neck = NECKS.get('UnetNeck')(in_channels=encoder_channels, decoder_channels=decoder_channels)
        head = HEADS.get(head_name)(neck.out_channels, self.num_classes)

        return Sequential(backbone, neck, head).to(device=self.device).eval()

    @parameterized.expand(tested_heads)
    def test_forward_output_shape(self, head_name):
        super(TestUnet, self).test_forward_output_shape(head_name)

    @parameterized.expand(tested_heads)
    def test_torchscript_conversion(self, head_name):
        super(TestUnet, self).test_torchscript_conversion(head_name)


class TestHRNet(AbstractTestSegmentationPair, unittest.TestCase):

    def create_model(self, head_name):
        backbone = BACKBONES.get('hrnet_w18_small')(pretrained=False, in_channels=3)
        backbone = BackboneWrapper(backbone)
        neck = NECKS.get('HRNetSegmentationNeck')(in_channels=backbone.out_encoder_channels)
        head = HEADS.get(head_name)(neck.out_channels, self.num_classes)

        return Sequential(backbone, neck, head).to(device=self.device).eval()

    @parameterized.expand(tested_heads)
    def test_forward_output_shape(self, head_name):
        super(TestHRNet, self).test_forward_output_shape(head_name)

    @parameterized.expand(tested_heads)
    def test_torchscript_conversion(self, head_name):
        super(TestHRNet, self).test_torchscript_conversion(head_name)
