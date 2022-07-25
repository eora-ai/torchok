import cv2
import hydra
from omegaconf import OmegaConf, DictConfig

from torchok.constructor.config_structure import ConfigParams
from torchok.constructor.runner import create_trainer
from torchok.constructor import TASKS


# Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
# (more info: https://github.com/pytorch/pytorch/issues/1355)
cv2.setNumThreads(0)


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
    trainer.fit(model, ckpt_path=config.resume_path)


if __name__ == '__main__':
    entrypoint()
