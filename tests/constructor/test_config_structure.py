import sys
sys.path.append('../../')

import hydra
from dataclasses import dataclass, field
from enum import Enum
from typing import *

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.constructor.config_structure import ConfigParams, Phase


@hydra.main(config_path="configs/", config_name="config.yaml")
def test_config_load_when_full_config_was_define(config):
    config_params = ConfigParams(**config)
    # test if metric phase is enum
    if isinstance(type(config_params.metrics[0].phases[0]), Phase):
        print('Failed. Phase is not Enum.')
    else:
        print('Phase enum OK.')


@hydra.main(config_path="configs/", config_name="config_without_metrics.yaml")
def test_config_load_when_metric_is_not_define(config):
    config_params = ConfigParams(**config)
    # test if metric phase is enum
    if isinstance(config_params.metrics, list) and len(config_params.metrics) == 0:
        print('Load conf without metric OK.')
    else:
        print('Failed. Load conf without metric.')


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ConfigParams)
    test_config_load_when_full_config_was_define()
    test_config_load_when_metric_is_not_define()
    