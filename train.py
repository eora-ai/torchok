import hydra
import omegaconf
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

# Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
# (more info: https://github.com/pytorch/pytorch/issues/1355)
import cv2
cv2.setNumThreads(0)

from src.constructor.config_structure import ConfigParams
from src.constructor.runner import create_trainer
from src.constructor.registry import TASKS


@hydra.main(config_path='configs')
def main(config: DictConfig):
    # Need to add --config-path (-cp) and --config_name (-cn) in run command
    # Example config in configs/classification_cifar10.yaml
    # TODO: add config into the project
    # Then the command line will be:
    # python --config-path configs --config_name classification_cifar10

    # Resolve -> change evn variable to values for example ${oc.env:USER} -> 'root'
    OmegaConf.resolve(config)
    # Register structure
    schema = OmegaConf.structured(ConfigParams)
    # Merge sturcture with config
    config = OmegaConf.merge(schema, config)
    # Create task
    model = TASKS.get(config.task.name)(config)
    trainer = create_trainer(config)
    trainer.fit(model)

if __name__ == '__main__':
    main()
    