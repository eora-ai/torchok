import unittest

import torch
from parameterized import parameterized
from torch.nn import Sequential

from torchok import BACKBONES, HEADS, NECKS
from torchok.models.backbones.base_backbone import BackboneWrapper

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
        self.img_size = (256, 256)
        self.input = torch.rand(2, 3, *self.img_size, device=self.device)

    def create_model(self, head_name):
        raise NotImplemented()

    def test_forward_output_shape(self, backbone_name):
        model = self.create_model(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertTupleEqual(output.shape, (2, self.num_classes, *self.img_size))
        torch.cuda.empty_cache()

    def test_torchscript_conversion(self, backbone_name):
        model = self.create_model(backbone_name)
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()


class TestUnet(AbstractTestSegmentationNeck, unittest.TestCase):

    def create_model(self, backbone_name):
        backbone = BACKBONES.get(backbone_name)(pretrained=False, in_channels=3)
        backbone = BackboneWrapper(backbone)
        encoder_channels = backbone.out_encoder_channels
        decoder_channels = (512, 256, 128, 64, 64)
        if len(encoder_channels) < len(decoder_channels):
            decoder_channels = decoder_channels[:len(encoder_channels)]
        neck = NECKS.get("UnetNeck")(in_channels=encoder_channels, decoder_channels=decoder_channels)
        head = HEADS.get("SegmentationHead")(neck.out_channels, self.num_classes)

        return Sequential(backbone, neck, head).to(device=self.device).eval()

    @parameterized.expand(example_backbones)
    def test_forward_output_shape(self, backbone_name):
        super(TestUnet, self).test_forward_output_shape(backbone_name)

    @parameterized.expand(example_backbones)
    def test_torchscript_conversion(self, backbone_name):
        super(TestUnet, self).test_torchscript_conversion(backbone_name)
