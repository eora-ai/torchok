from typing import Tuple
from pytorch_lightning import Trainer, LightningModule
from omegaconf import DictConfig

from torchok.constructor import TASKS
from torchok.constructor.runner import create_trainer


def find_lr(config: DictConfig, model: LightningModule, trainer: Trainer) -> float:
    """Run pytorch lightning `tuner` for auto learning rate find, if `config.trainer.auto_lr_find` flag is True and
    config have only one optimizer.

    Args:
        config: Yaml config.
        model: Task for learning rate find.
        trainer: Task trainer.

    Returns:
        suggested_lr: Found learning rate.

    Raises:
        ValueError: When `config.trainer.auto_lr_find` is True and task has more than one optimizer.
    """
    if config.trainer.auto_lr_find is None or not config.trainer.auto_lr_find:
        return None

    if len(config.optimization) > 1:
        raise ValueError(f'Pytorch Lightning support only one optimizer auto learning rate find. Current optimizer'
                         f' count is {len(config.optimization)}. Set the trainer.auto_lr_find flag to False to avoid'
                         f' this problem.')

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)
    suggested_lr = lr_finder.suggestion()

    return suggested_lr


def auto_lr_find(config: DictConfig, model: LightningModule, trainer: Trainer) -> Tuple[LightningModule, Trainer]:
    """Recreate model and trainer after auto learning rate find. Model and trainer need to be recreated because
    for auto learning rate found pytorch lightning use `training_step` so it change loaded weights. If
    `config.trainer.auto_lr_find` is True - return input model and trainer.

    Args:
        config: Yaml config.
        model: Task for learning rate find.
        trainer: Task trainer.

    Returns:
        model: Recreated task with auto found learning rate.
        trainer: Recreated trainer.
    """
    suggested_lr = find_lr(config, model, trainer)

    if suggested_lr is None:
        return model, trainer

    config.optimization[0].optimizer.params.lr = suggested_lr
    model = TASKS.get(config.task.name)(config)
    trainer = create_trainer(config)
    return model, trainer