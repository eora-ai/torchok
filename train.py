import argparse
import re
import yaml
import os
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pathlib import Path


# Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
# (more info: https://github.com/pytorch/pytorch/issues/1355)
import cv2
cv2.setNumThreads(0)

from src.constructor.config_structure import ConfigParams
from src.constructor.runner import create_trainer
from src.constructor.registry import TASKS


@hydra.main(config_path=None)
def load_config(config: DictConfig):
    # Need to add --config-path (-cp) and --config_name (-cn) in run command
    # Example config in configs/classification_cifar10.yaml
    # TODO: add config into the project
    # Then the command line will be:
    # python --config-path configs --config_name classification_cifar10 
    config_params = ConfigParams(**config)
    return config_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: May be need set default config path
    parser.add_argument('-cp', '--config-path', type=str,
                        help="Path to folder with yaml file with configuration parameters.")
    parser.add_argument('-cn', '--config-name', type=str,
                        help="Yaml file name with configuration parameters.")
    parser.add_argument('-jl', '--job-link', type=str,
                        help="sagemaker job name, if running localy set to 'local'", default='local')
    args = parser.parse_args()

    # registr ConfigStructure
    cs = ConfigStore.instance()
    cs.store(name='base_configparams', node=ConfigParams)

    config = load_config()

    model = TASKS.get(config.task.name)(config)
    trainer = create_trainer(config, str(args.job_link))
    trainer.fit(model)
