import unittest

import torch
from parameterized import parameterized
from torch.nn import Module, Sequential

from torchok import BACKBONES, HEADS, NECKS

tested_heads = [
    'SegmentationHead',
    'OCRSegmentationHead'
]


class Backbone(Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)

    def forward(self, x):
        return self.backbone.forward_features(x)

    @property
    def out_encoder_channels(self):
        return self.backbone.out_encoder_channels


class AbstractTestSegmentationPair:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 10

    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    def create_model(self, head_name):
        raise NotImplemented()

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
        backbone = Backbone('resnet18').to(device=self.device).eval()
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
        backbone = Backbone("hrnet_w18").to(device=self.device).eval()
        neck = NECKS.get('HRNetSegmentationNeck')(in_channels=backbone.out_encoder_channels)
        head = HEADS.get(head_name)(neck.out_channels, self.num_classes)

        return Sequential(backbone, neck, head).to(device=self.device).eval()

    @parameterized.expand(tested_heads)
    def test_forward_output_shape(self, head_name):
        super(TestHRNet, self).test_forward_output_shape(head_name)

    @parameterized.expand(tested_heads)
    def test_torchscript_conversion(self, head_name):
        super(TestHRNet, self).test_torchscript_conversion(head_name)
