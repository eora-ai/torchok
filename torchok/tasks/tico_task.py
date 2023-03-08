from typing import Dict, List, Any, Union
from omegaconf import DictConfig

import torch
import numpy as np
from torch import Tensor, nn
from torch.optim import Optimizer

from torchok.constructor.config_structure import Phase
from torchok.constructor import TASKS, BACKBONES
from torchok.tasks.base import BaseTask


@TASKS.register_class
class TiCoTask(BaseTask):
    """
    Task-agnostic part of the SimCLR v2 approach described in paper
    Big Self-Supervised Models are Strong Semi-Supervised Learners: https://arxiv.org/abs/2006.10029

    This task use `UnsupervisedContrastiveDataset` dataset.
    """

    def __init__(
        self,
        hparams: DictConfig,
        backbone_name: str,
        backbone_params: dict = None,
        momentum: float = 0.99,
        final_dim: int = 256,
        inputs: dict = None,
    ):
        """Init SimCLRTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            pooling_name: name of the backbone architecture in the POOLINGS registry.
            head_name: name of the neck architecture in the HEADS registry.
            neck_name: if present, name of the head architecture in the NECKS registry. Otherwise, model will be created
                without neck.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            pooling_params: parameters for neck constructor. `in_channels` will be set automatically based on neck or
                backbone if neck is absent.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, inputs=inputs)

        self.backbone = BACKBONES.get(backbone_name)(**backbone_params)

        self.backbone_momentum = BACKBONES.get(backbone_name)(**backbone_params)
        self.backbone_momentum.eval()

        self.projection_head = nn.Sequential(
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, final_dim, bias=True),
        )

        self.projection_head_momentum = nn.Sequential(
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, final_dim, bias=True),
        )
        self.projection_head_momentum.eval()

        self.max_epoch = self.hparams.trainer.max_epochs
        self.start_momentum = momentum
        self.momentum = 0

        # self.register_buffer("c_prev", torch.zeros(final_dim, final_dim, requires_grad=False))
        self.prev_cov_matrix = nn.Variable(torch.zeros(final_dim, final_dim), requires_grad=True)
        self.prev_cov_matrix = self.C_prev.detach()

    def configure_optimizers(self) -> List[Dict[str, Union[Optimizer, Dict[str, Any]]]]:
        """Configure optimizers."""
        modules = [self.backbone, self.projection_head]
        opt_sched_list = self._constructor.configure_optimizers(modules)
        return opt_sched_list

    def train(self, mode: bool = True):
        self.training = mode
        for block in [self.backbone, self.projection_head]:
            for module in block.children():
                module.train(mode)
        return self

    def on_train_epoch_start(self):
        self.momentum = self.start_momentum + (1 - self.start_momentum) * np.sin(
            (np.pi / 2) * self.current_epoch / (self.max_epoch - 1)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_momentum.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        for param_q, param_k in zip(self.projection_head.parameters(), self.projection_head_momentum.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def forward_with_gt(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image_0` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.
                - `image_1` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.

        Returns:
            Dictionary with the following keys and values

            - 'emb1': torch.Tensor of shape `(B, num_features)`, representing embeddings for batch['image_0'].
            - 'emb2': torch.Tensor of shape `(B, num_features)`, representing embeddings for batch['image_1'].
        """
        # compute key features

        x1, x2 = batch["image_0"], batch["image_1"]
        output = dict()
        output["emb1"] = self.forward(x1)
        output["emb2"] = self.forward_momentum(x2)
        return output

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete training loop."""
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
        output = self.forward_with_gt(batch)
        total_loss, tagged_loss_values = self.losses(**output)

        loss, prev_cov_matrix = total_loss
        self.prev_cov_matrix = prev_cov_matrix.detach()

        self.metrics_manager.update(Phase.TRAIN, **output)
        output_dict = {"loss": loss}
        output_dict.update(tagged_loss_values)
        return output_dict
