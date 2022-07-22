from pytorch_lightning.callbacks import Callback


class FinalizeLogger(Callback):
    """Callback to finalize logger if an error occurs"""
    def on_exception(self, trainer, pl_module, outputs):
        # Need to save checkpoints for every exception not only isinstance(outputs, KeyboardInterrupt)
        status = 'KILLED' if type(outputs) == KeyboardInterrupt else 'FAILED'
        trainer.logger.finalize(status)
