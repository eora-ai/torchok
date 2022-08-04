import unittest

import torch
from parameterized import parameterized
from torch.nn import Module

from torchok import BACKBONES, HEADS, NECKS

example_backbones = [
    'resnet18',
    'efficientnet_b0',
    'mobilenetv3_small_100',
    'swinv2_tiny_window8_256',
    'davit_t'
]


class AbstractTestSegmentationNeck:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 10

    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    def test_forward_output_shape(self, backbone_name):
        model = self.create_model(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertTupleEqual(output.shape, (2, self.num_classes, 256, 256))
        torch.cuda.empty_cache()

    def test_torchscript_conversion(self, backbone_name):
        model = self.create_model(backbone_name)
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()


class UnetModel(Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        encoder_channels = self.backbone.out_encoder_channels
        decoder_channels = (512, 256, 128, 64, 64)
        if len(encoder_channels) < len(decoder_channels):
            decoder_channels = decoder_channels[:len(encoder_channels)]
        self.neck = NECKS.get("UnetNeck")(encoder_channels=encoder_channels,
                                          decoder_channels=decoder_channels)
        self.head = HEADS.get("SegmentationHead")(self.neck.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.neck(features)
        output = self.head(features)
        return output


class TestUnet(AbstractTestSegmentationNeck, unittest.TestCase):

    def create_model(self, backbone_name):
        return UnetModel(backbone_name, self.num_classes).to(device=self.device).eval()

    @parameterized.expand(example_backbones)
    def test_forward_output_shape(self, backbone_name):
        super(TestUnet, self).test_forward_output_shape(backbone_name)

    @parameterized.expand(example_backbones)
    def test_torchscript_conversion(self, backbone_name):
        super(TestUnet, self).test_torchscript_conversion(backbone_name)
