import unittest

import torch
from parameterized import parameterized

from src.constructor import create_backbone
from src.models.backbones.utils import list_models
from .test_segmentation import example_backbones


def inp(bsize, in_ch, w, h):
    return torch.ones(bsize, in_ch, w, h)


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @parameterized.expand(list_models(module='vision_transformer', exclude_filters=''))
    def test_vit_torchscript_conversion(self, backbone_name):
        model = create_backbone(backbone_name, img_size=self.input.shape[2]).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()

    @parameterized.expand(list_models(module='coat', exclude_filters=''))
    def test_coat_torchscript_conversion(self, backbone_name):
        model = create_backbone(backbone_name, img_size=self.input.shape[2]).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()

    @parameterized.expand(list_models(module='swin_transformer', exclude_filters=''))
    def test_swin_torchscript_conversion(self, backbone_name):
        model = create_backbone(backbone_name).to(self.device).eval()
        input = torch.rand(2, 3, *model.img_size, device=self.device)
        with torch.no_grad():
            torch.jit.trace(model, input)
        torch.cuda.empty_cache()
