from pytorch_lightning.callbacks import Callback


class UploadWeightsAfterException(Callback):
    """Callback for """
    def on_exception(self, trainer, pl_module, outputs):
        # I think we need to save checkpoints for every exception not even isinstance(outputs, KeyboardInterrupt)
        trainer.logger.finalize(type(outputs).__name__)
