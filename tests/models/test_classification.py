import unittest

import torch
import yaml
from parameterized import parameterized

from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS


example_backbones = [
    'hrnet_w18_small_v2',
    'resnest14d',
    'gluon_resnet18_v1b',
    'res2net50_26w_4s',
    'efficientnet_b0',
]


def _import_config(path) -> TrainConfigParams:
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return TrainConfigParams(**data_loaded)


class TestClassificationTask(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input = torch.rand(2, 3, 224, 224, device=self.device)

    @parameterized.expand(example_backbones)
    def test_forward(self, backbone_name):
        config = _import_config('tests/models/configs/classification_test.yml')
        config.task.params['backbone_name'] = backbone_name
        model = TASKS.get(config.task.name)(config).to(self.device).eval()
        with torch.no_grad():
            output = model(self.input)
            self.assertEqual(output.shape[1], model.params.head_params['num_classes'])
        torch.cuda.empty_cache()

    @parameterized.expand(example_backbones)
    def test_torchscript_conversion(self, backbone_name):
        config = _import_config('tests/models/configs/classification_test.yml')
        config.task.params['backbone_name'] = backbone_name
        model = TASKS.get(config.task.name)(config).to(self.device).eval()
        with torch.no_grad():
            torch.jit.trace(model, self.input, check_tolerance=1e-3)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
