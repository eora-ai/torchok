import unittest

import torch
from parameterized import parameterized

from torchok.constructor import BACKBONES


class AbstractTestBackboneCorrectness:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def create_backbone(self, backbone_name):
        return BACKBONES.get(backbone_name)(pretrained=False, in_channels=3).to(device=self.device).eval()

    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    def test_load_pretrained(self, backbone_name):
        BACKBONES.get(backbone_name)(pretrained=True).to(self.device).eval()
        torch.cuda.empty_cache()

    def test_forward_output_shape(self, backbone_name, expected_shape):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertTupleEqual(output.shape, expected_shape)
        torch.cuda.empty_cache()

    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            features = model.forward_features(self.input)
        feature_shapes = [f.shape for f in features]
        self.assertListEqual(feature_shapes, expected_shapes)
        torch.cuda.empty_cache()

    def test_torchscript_conversion(self, backbone_name):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()


class TestEfficientnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand([['efficientnet_b0']])
    def test_load_pretrained(self, model_name):
        super().test_load_pretrained(model_name)

    @parameterized.expand([['efficientnet_b0', (2, 1280, 2, 2)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand([
        ['efficientnet_b0', [(2, 3, 64, 64), (2, 16, 32, 32), (2, 24, 16, 16),
                             (2, 40, 8, 8), (2, 112, 4, 4), (2, 320, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='efficientnet'))
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestHrnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand([['hrnet_w18']])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([
        ['hrnet_w18', [(2, 18, 16, 16), (2, 36, 8, 8), (2, 72, 4, 4), (2, 144, 2, 2)]]
    ])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertListEqual([f.shape for f in output], expected_shape)
        torch.cuda.empty_cache()

    @parameterized.expand([
        ['hrnet_w18', [(2, 3, 64, 64), (2, 18, 16, 16), (2, 36, 8, 8), (2, 72, 4, 4), (2, 144, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='hrnet'))
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestDaViT(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 224, 224, device=self.device)

    @parameterized.expand([['davit_t']])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['davit_t', (2, 768, 7, 7)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand([
        ['davit_t', [(2, 3, 224, 224), (2, 96, 56, 56), (2, 192, 28, 28), (2, 384, 14, 14), (2, 768, 7, 7)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='davit'))
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestMobilenetv3(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand([['mobilenetv3_small_050']])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['mobilenetv3_small_050', (2, 288, 2, 2)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand([
        ['mobilenetv3_small_050', [(2, 3, 64, 64), (2, 8, 16, 16), (2, 16, 8, 8), (2, 24, 4, 4), (2, 288, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='mobilenetv3'))
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestResnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand([['resnet18']])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['resnet18', (2, 512, 2, 2)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand([
        ['resnet18', [(2, 3, 64, 64), (2, 64, 32, 32), (2, 64, 16, 16),
                      (2, 128, 8, 8), (2, 256, 4, 4), (2, 512, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='resnet'))
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestSwin(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    @parameterized.expand([['swinv2_tiny_window16_256']])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['swinv2_tiny_window16_256', (2, 768, 8, 8)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand(
        [['swinv2_tiny_window16_256', [(2, 3, 256, 256), (2, 96, 64, 64),
                                       (2, 192, 32, 32), (2, 384, 16, 16), (2, 768, 8, 8)]]]
    )
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(BACKBONES.list_models(module='swin'))
    def test_torchscript_conversion(self, backbone_name):
        model = BACKBONES.get(backbone_name)(pretrained=False).to(self.device).eval()
        x = torch.rand(2, 3, *model.img_size, device=self.device)
        with torch.no_grad():
            torch.jit.trace(model, x)
        torch.cuda.empty_cache()
