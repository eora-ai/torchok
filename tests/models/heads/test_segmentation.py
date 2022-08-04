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
    def __init__(self):
        super().__init__()
        self.backbone = BACKBONES.get('resnet18')(pretrained=False, in_channels=3)

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
        self.backbone = Backbone().to(device=self.device).eval()

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


class UnetModel(Module):
    def __init__(self, backbone_name, neck_name, num_classes):
        super().__init__()
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        encoder_channels = self.backbone.out_encoder_channels
        decoder_channels = (512, 256, 128, 64, 64)
        if len(encoder_channels) < len(decoder_channels):
            decoder_channels = decoder_channels[:len(encoder_channels)]
        self.neck = NECKS.get(neck_name)(encoder_channels=encoder_channels,
                                         decoder_channels=decoder_channels)
        self.head = HEADS.get('SegmentationHead')(self.neck.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.neck(features)
        output = self.head(features)
        return output


class TestUnet(AbstractTestSegmentationPair, unittest.TestCase):

    def create_model(self, head_name):
        encoder_channels = self.backbone.out_encoder_channels
        decoder_channels = (512, 256, 128, 64, 64)
        if len(encoder_channels) < len(decoder_channels):
            decoder_channels = decoder_channels[:len(encoder_channels)]
        self.neck = NECKS.get('UnetNeck')(encoder_channels=encoder_channels,
                                          decoder_channels=decoder_channels)
        self.head = HEADS.get(head_name)(self.neck.out_channels, self.num_classes)

        return Sequential(self.backbone, self.neck, self.head).to(device=self.device).eval()

    @parameterized.expand(tested_heads)
    def test_forward_output_shape(self, head_name):
        super(TestUnet, self).test_forward_output_shape(head_name)

    @parameterized.expand(tested_heads)
    def test_torchscript_conversion(self, head_name):
        super(TestUnet, self).test_torchscript_conversion(head_name)
