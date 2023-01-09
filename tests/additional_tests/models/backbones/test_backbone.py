import gc
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
        gc.collect()

    def test_forward_output_shape(self, backbone_name, expected_shape):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertTupleEqual(output.shape, expected_shape)
        torch.cuda.empty_cache()
        gc.collect()

    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            features = model.forward_features(self.input)
        feature_shapes = [f.shape for f in features]
        self.assertListEqual(feature_shapes, expected_shapes)
        torch.cuda.empty_cache()
        gc.collect()

    def test_torchscript_conversion(self, backbone_name):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            torch.jit.trace(model.forward, self.input)
        torch.cuda.empty_cache()
        gc.collect()


class TestEfficientnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand(['efficientnet_b0'])
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

    @parameterized.expand(['efficientnet_b0'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestHrnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand(['hrnet_w18_small'])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([
        ['hrnet_w18_small', [(2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4), (2, 128, 2, 2)]]
    ])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        model = self.create_backbone(backbone_name)
        with torch.no_grad():
            output = model(self.input)
        self.assertListEqual([f.shape for f in output], expected_shape)
        torch.cuda.empty_cache()

    @parameterized.expand([
        ['hrnet_w18_small', [(2, 3, 64, 64), (2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4), (2, 128, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(['hrnet_w18_small'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestDaViT(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 224, 224, device=self.device)

    @parameterized.expand(['davit_t'])
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

    @parameterized.expand(['davit_t'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestMobilenetv3(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand(['mobilenetv3_small_050'])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['mobilenetv3_small_050', (2, 288, 2, 2)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand([
        ['mobilenetv3_small_050', [(2, 3, 64, 64), (2, 16, 32, 32), (2, 8, 16, 16),
                                   (2, 16, 8, 8), (2, 24, 4, 4), (2, 288, 2, 2)]]
    ])
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(['mobilenetv3_small_050'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestResnet(AbstractTestBackboneCorrectness, unittest.TestCase):
    @parameterized.expand(['resnet18'])
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

    @parameterized.expand(['resnet18'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestSwin(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 256, 256, device=self.device)

    @parameterized.expand(['swinv2_tiny_window16_256'])
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

    @parameterized.expand(['swinv2_tiny_window16_256'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestGlobalContextVit(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 224, 224, device=self.device)

    @parameterized.expand(['gcvit_xxtiny'])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['gcvit_xxtiny', (2, 512, 7, 7)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand(
        [['gcvit_xxtiny', [(2, 3, 224, 224), (2, 64, 56, 56),
                           (2, 128, 28, 28), (2, 256, 14, 14), (2, 512, 7, 7)]]]
    )
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(['gcvit_xxtiny'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)


class TestVit(AbstractTestBackboneCorrectness, unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.rand(2, 3, 224, 224, device=self.device)

    @parameterized.expand(['vit_tiny_patch16_224'])
    def test_load_pretrained(self, backbone_name):
        super().test_load_pretrained(backbone_name)

    @parameterized.expand([['vit_tiny_patch16_224', (2, 192)]])
    def test_forward_output_shape(self, backbone_name, expected_shape):
        super().test_forward_output_shape(backbone_name, expected_shape)

    @parameterized.expand(
        [['vit_tiny_patch16_224', [(2, 3, 224, 224), (2, 192, 14, 14),
                                   (2, 192, 14, 14), (2, 192, 14, 14), (2, 192, 14, 14)]]]
    )
    def test_forward_feature_output_shape(self, backbone_name, expected_shapes):
        super().test_forward_feature_output_shape(backbone_name, expected_shapes)

    @parameterized.expand(['vit_tiny_patch16_224'])
    def test_torchscript_conversion(self, backbone_name):
        super().test_torchscript_conversion(backbone_name)
