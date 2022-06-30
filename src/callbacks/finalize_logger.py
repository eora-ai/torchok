from pytorch_lightning.callbacks import Callback


class FinalizeLogger(Callback):
    """Callback for finalize logger if an error occurs"""
    def on_exception(self, trainer, pl_module, outputs):
        # I think we need to save checkpoints for every exception not even isinstance(outputs, KeyboardInterrupt)
        status = 'FINISHED' if type(outputs).__name__ == 'KeyboardInterrupt' else 'FAILED'
        trainer.logger.finalize(status)