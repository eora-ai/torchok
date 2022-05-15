import sys
sys.path.append('../../')

import hydra
from dataclasses import dataclass, field
from enum import Enum
from typing import *

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.constructor.config_structure import ConfigParams, Phase