import logging
from pytorch_lightning import Trainer, LightningModule
from omegaconf import DictConfig


def find_lr(config: DictConfig, model: LightningModule, trainer: Trainer):
    """Run pytorch lightning `tuner` for auto learning rate find, if `config.trainer.auto_lr_find` flag is True and
    config have only one optimizer.

    Args:
        config: Yaml config.
        model: Task for learning rate find.
        trainer: Task trainer.

    Raises:
        ValueError: When `config.trainer.auto_lr_find` is True and task has more than one optimizer.
    """
    if len(config.optimization) > 1:
        raise ValueError(f'Pytorch Lightning support only one optimizer auto learning rate find. Current optimizer'
                         f' count is {len(config.optimization)}. Set the trainer.auto_lr_find flag to False to avoid'
                         f' this problem.')

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)
    suggested_lr = lr_finder.suggestion()

    logging.info(f'Suggested learning rate = {suggested_lr}')
