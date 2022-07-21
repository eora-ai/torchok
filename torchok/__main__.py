import hydra
from omegaconf import OmegaConf, DictConfig

# Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
# (more info: https://github.com/pytorch/pytorch/issues/1355)
import cv2
cv2.setNumThreads(0)

from torchok.constructor.config_structure import ConfigParams
from torchok.constructor.runner import create_trainer
from torchok.constructor import TASKS


@hydra.main(version_base=None, config_path=None, config_name=None)
def entrypoint(config: DictConfig):
    # Need to add --config-path (-cp) and --config_name (-cn) in run command
    # Example config: examples/configs/classification_cifar10.yaml
    # Then the command will be:
    # python --config-path configs --config_name classification_cifar10

    # Resolve -> change evn variable to values for example ${oc.env:USER} -> 'root'
    OmegaConf.resolve(config)
    # Register structure
    schema = OmegaConf.structured(ConfigParams)
    # Merge structure with config
    config = OmegaConf.merge(schema, config)
    # Create task
    model = TASKS.get(config.task.name)(config)
    trainer = create_trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    entrypoint()
