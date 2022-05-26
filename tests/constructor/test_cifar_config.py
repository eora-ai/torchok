import sys
sys.path.append('../../')

import hydra
import omegaconf
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from enum import Enum
from typing import *

from src.constructor.config_structure import ConfigParams, Phase


@hydra.main(config_path='/workdir/rbayazitov/torchok/examples/configs/', config_name='classification_cifar10')
def main(config: DictConfig):
    OmegaConf.resolve(config)
    schema = OmegaConf.structured(ConfigParams)
    config = OmegaConf.merge(schema, config)
    params = ConfigParams(**config)
    print(params.checkpoint)
 

if __name__ == '__main__':
    main()
