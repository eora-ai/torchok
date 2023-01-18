from pytorch_lightning import Trainer

from torchok.constructor import CALLBACKS
from torchok.constructor.logger import create_logger


def create_trainer(train_config):
    callbacks = []
    logger = create_logger(train_config.logger)

    # Create callbacks
    callbacks_config = train_config.callbacks
    if callbacks_config is not None and len(callbacks_config) != 0:
        for callback_config in callbacks_config:
            callbacks.append(CALLBACKS.get(callback_config.name)(**callback_config.params))
    callbacks = callbacks or None

    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
