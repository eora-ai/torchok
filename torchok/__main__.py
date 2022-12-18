import cv2
import hydra
from omegaconf import DictConfig, OmegaConf

import torchok
from pytorch_lightning import seed_everything
from torchok.constructor.config_structure import ConfigParams
from torchok.constructor.runner import create_trainer
from torchok.constructor.auto_lr_find import find_lr


@hydra.main(version_base=None, config_path=None, config_name=None)
def entrypoint(config: DictConfig):
    # Need to add --config-path (-cp) and --config_name (-cn) in run command
    # Example config: examples/configs/classification_cifar10.yaml
    # Then the command will be:
    # python --config-path configs --config_name classification_cifar10

    # Resolve -> change evn variable to values for example ${oc.env:USER} -> 'root'
    OmegaConf.resolve(config)
    # Get mode name - default is train
    mode = config.get('mode', 'train')
    # Remove mode key, because it isn't in ConfigParams
    if 'mode' in config:
        config = dict(config)
        config.pop('mode')
    # Register structure
    schema = OmegaConf.structured(ConfigParams)
    # Merge structure with config
    config = OmegaConf.merge(schema, config)
    # Seed everything
    if config.seed_params is not None:
        seed_everything(**config.seed_params)
    # Create task
    model = torchok.TASKS.get(config.task.name)(config, **config.task.params)
    trainer = create_trainer(config)
    if mode == 'train':
        trainer.fit(model, ckpt_path=config.resume_path)
    elif mode == 'test':
        trainer.test(model, ckpt_path=config.resume_path)
    elif mode == 'predict':
        trainer.predict(model, ckpt_path=config.resume_path)
    elif mode == 'find_lr':
        find_lr(model, trainer)
    else:
        raise ValueError(f'Main function error. Entrypoint with name <{mode}> does not support, please use '
                         f'the following entrypoints - [train, test, predict].')


if __name__ == '__main__':
    # Hack to fix multiprocessing deadlock when PyTorch's DataLoader is used
    # (more info: https://github.com/pytorch/pytorch/issues/1355)
    cv2.setNumThreads(0)
    entrypoint()
