import unittest

import torch
from src.constructor import create_backbone
from parameterized import parameterized

from src.constructor.config_structure import StructureParams


def inp(bsize, in_ch, w, h):
    return torch.ones(bsize, in_ch, w, h)


class TestBackboneLoader(unittest.TestCase):
    @parameterized.expand([
        (1, 3, 224, 224),
        (3, 3, 224, 224),
    ])
    def test_resnet_with_pretrained_correct_in_chans(self, batch_size, in_chans, w, h):
        resnet_params = StructureParams(name='resnet18', params={'pretrained': True, 'in_chans': in_chans})
        resnet_model = create_backbone(model_name=resnet_params.name, **resnet_params.params)

        out = resnet_model.forward(inp(batch_size, in_chans, w, h))
        self.assertEqual(out.shape, (batch_size, 512, 7, 7))

    def test_resnet_without_pretrained(self):
        resnet_params = StructureParams(name='resnetv2_50x1_bitm', params={'pretrained': True})
        resnet_model = create_backbone(model_name=resnet_params.name, **resnet_params.params)

        out = resnet_model.forward(inp(1, 3, 224, 224))

        self.assertEqual(out.shape, (1, 2048, 7, 7))

    @parameterized.expand([
        (1, 4),
        (3, 3)
    ])
    def test_resnet_with_hparams(self, batch_size, in_chans):
        resnet_params = StructureParams(name='resnet18', params={'in_chans': in_chans})
        resnet_model = create_backbone(model_name=resnet_params.name, **resnet_params.params)

        out = resnet_model.forward(inp(batch_size, in_chans, 224, 224))

        self.assertEqual(out.shape, (batch_size, 512, 7, 7))

    @parameterized.expand([
        (1, 3), (3, 3)
    ])
    def test_pruned_with_hparams(self, batch_size, in_chans):
        params = StructureParams(name='efficientnet_b1_pruned', params={'in_chans': in_chans})
        model = create_backbone(model_name=params.name, **params.params)

        out = model.forward(inp(batch_size, in_chans, 224, 224))

        self.assertEqual(out.shape, (batch_size, 1280, 7, 7))


if __name__ == '__main__':
    unittest.main()
