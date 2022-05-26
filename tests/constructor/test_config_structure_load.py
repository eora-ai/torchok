import unittest

import omegaconf
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from typing import *

from src.constructor.config_structure import ConfigParams, Phase


SCHEMA = OmegaConf.structured(ConfigParams)

def load_structured_config(path):
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)
    schema = OmegaConf.structured(ConfigParams)
    config = OmegaConf.merge(schema, config)
    return config


class TestConfigSrucure(unittest.TestCase):
    def test_load_config_when_full_config_was_defined(self):
        load_structured_config('tests/constructor/configs/config.yaml')

    def test_enum_load_when_full_config_was_defined(self):
        config = load_structured_config('tests/constructor/configs/config.yaml')

        data_phase_type = type(list(config.data.keys())[0])
        self.assertEqual(data_phase_type, Phase)

        metric_phase_type = type(config.metrics[0].phases[0])
        self.assertEqual(metric_phase_type, Phase)

    def test_env_variable_when_full_config_was_defined(self):
        config = load_structured_config('tests/constructor/configs/config_with_env_variable.yaml')
        self.assertEqual(config.job_link, 'root')

    def test_optional_type_of_metrics_when_config_does_not_have_metrics(self):
        config = load_structured_config('tests/constructor/configs/config_without_metrics.yaml')
        self.assertEqual(len(config.metrics), 0)

    def test_structure_schema_when_add_not_registrated_parameter(self):
        self.assertRaises(KeyError, load_structured_config, 'tests/constructor/configs/config_with_bag.yaml')
    