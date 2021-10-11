import unittest

import torch
from parameterized import parameterized

from src.constructor import create_backbone
from src.models.backbones.utils import list_models, FeatureHooks


def inp(bsize, in_ch, w, h):
    return torch.ones(bsize, in_ch, w, h)


class TestBackboneCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 64, 64, device=self.device)

    @parameterized.expand(list_models(module='', exclude_filters=['vit_*', 'coat', 'swin_transformer']))
    def test_backbone_forward(self, backbone_name):
        model = create_backbone(backbone_name, set_neck=True).to(self.device).eval()
        hooks = [dict(module=name, type='forward') for name in model.stage_names]
        with torch.no_grad():
            feature_hooks = FeatureHooks(hooks, model.named_modules())
            model(self.input)
            result = feature_hooks.get_output(self.device)
            self.assertTrue([model.forward_neck(i).shape for i in result.values()])
        torch.cuda.empty_cache()

    @parameterized.expand(list_models(module='', exclude_filters=['vit_*', 'coat', 'swin_transformer']))
    def test_torchscript_conversion(self, backbone_name):
        model = create_backbone(backbone_name).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input)
        torch.cuda.empty_cache()
