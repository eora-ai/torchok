import logging
from pytorch_lightning import Trainer, LightningModule


def find_lr(model: LightningModule, trainer: Trainer):
    """Run pytorch lightning `tuner` for auto learning-rate find, if `config.trainer.auto_lr_find` flag is True and
    config has only one optimizer.

    Note that finding the learning rate doesn't work with more than one optimizer.

    Args:
        config: Yaml config.
        model: Task for learning rate find.
        trainer: Task trainer.
    """
    lr_finder = trainer.tuner.lr_find(model)
    suggested_lr = lr_finder.suggestion()

    logging.info(f'Suggested learning rate = {suggested_lr}')
