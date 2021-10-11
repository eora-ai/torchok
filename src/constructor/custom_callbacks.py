from pathlib import Path

import torch
from torch import distributed as dist
from pytorch_lightning.callbacks import Callback

from src.registry import CALLBACKS


def gather_list_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


@CALLBACKS.register_class
class EmbeddingCollectorCallback(Callback):
    def __init__(self, path_to_save):
        self.__path_to_save = Path(path_to_save)
        self.__embeddings = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx: int, dataloader_idx: int) -> None:
        """Called when the train batch ends."""
        if dist.is_initialized():
            embeddings = gather_list_and_concat(outputs['embeddings'])
        else:
            embeddings = outputs['embeddings']
        self.__embeddings.append(embeddings.detach().cpu())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx: int, dataloader_idx: int) -> None:
        """Called when the train batch ends."""
        if dist.is_initialized():
            embeddings = gather_list_and_concat(outputs['embeddings'])
        else:
            embeddings = outputs['embeddings']
        self.__embeddings.append(embeddings.detach().cpu())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx: int, dataloader_idx: int) -> None:
        """Called when the train batch ends."""
        if dist.is_initialized():
            embeddings = gather_list_and_concat(outputs['embeddings'])
        else:
            embeddings = outputs['embeddings']
        self.__embeddings.append(embeddings.detach().cpu())

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        embeddings = torch.cat(self.__embeddings, 0)
        torch.save(embeddings, self.__path_to_save)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        embeddings = torch.cat(self.__embeddings, 0)
        torch.save(embeddings, self.__path_to_save)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        embeddings = torch.cat(self.__embeddings, 0)
        torch.save(embeddings, self.__path_to_save)

