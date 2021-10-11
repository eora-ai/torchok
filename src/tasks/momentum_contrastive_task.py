import torch
from torch.nn import functional as F
from pydantic import BaseModel

from src.constructor import create_backbone, create_optimizer, create_scheduler
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, HEADS, POOLINGS
from .base_task import BaseTask
from src.models.backbones.utils import load_checkpoint


class MomentumContrastiveParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    pooling_name: str = 'Pooling'
    # in_features is taken from backbone.num_features
    # Be careful, default pooling parameters dict is not full!
    pooling_params: dict = {}
    head_name: str
    head_params: dict
    memory_size: int = 65536
    momentum: float = 0.999
    temperature: float = 0.07


@TASKS.register_class
class MomentumContrastiveTask(BaseTask):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    config_parser = MomentumContrastiveParams

    def __init__(self, hparams: TrainConfigParams):
        """
        num_features: feature dimension
        memory_size: number of negative keys (default: 65536)
        momentum: moco momentum of updating key encoder (default: 0.999)
        temperature: softmax temperature (default: 0.07)
        """
        super().__init__(hparams)

        self.memory_size = self.params.memory_size
        self.momentum = self.params.momentum
        self.temperature = self.params.temperature

        self.backbone_q = create_backbone(model_name=self.params.backbone_name,
                                          **self.params.backbone_params)
        self.backbone_k = create_backbone(model_name=self.params.backbone_name,
                                          **self.params.backbone_params)
        self.num_features = self.backbone_q.num_features
        self.params.pooling_params['in_features'] = self.num_features
        self.pooling_q = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)
        self.pooling_k = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)

        self.params.head_params['in_features'] = self.num_features
        self.head_q = HEADS.get(self.params.head_name)(**self.params.head_params)
        self.head_k = HEADS.get(self.params.head_name)(**self.params.head_params)

        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.register_buffer("queue", torch.randn(self.num_features, self.params.queue_size))
        self.queue = torch.zeros(self.head_q.out_features, self.memory_size)
        self.queue = F.normalize(self.queue, dim=0)
        self.pointer = 0

    def configure_optimizers(self):
        optimizer = create_optimizer(list(self.backbone_q.parameters()) + list(self.head_q.parameters()),
                                     self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward(self, x):
        x = self.backbone_q(x)
        x = self.pooling_q(x)
        x = self.head_q(x)

        return x

    def forward_with_gt(self, batch):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        im_q = batch['input_0']
        im_k = batch['input_1']

        # compute query features
        q = self.forward(im_q)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.backbone_k(im_k)  # keys: NxC
            k = self.pooling_k(k)
            k = self.head_k(k)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK

        if q.device != self.queue.device:
            self.queue = self.queue.type(q.dtype).to(q.device)

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.params.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # dequeue and enqueue
        self.enqueue(k)

        output = {'embeddings': q, 'prediction': logits, 'target': labels}
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('train', **output)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        val_loss = self.criterion(**output)
        self.metric_manager.update('valid', **output)

        return val_loss

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        output = self.forward_with_gt(batch)
        val_loss = self.criterion(**output)

        return val_loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        m = self.params.momentum
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        for param_q, param_k in zip(self.pooling_q.parameters(), self.pooling_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def enqueue(self, features):
        # features = concat_all_gather(features)
        features = features.detach().requires_grad_(False)
        self.queue = torch.cat([features.t(), self.queue[:, :-features.shape[0]]], dim=1)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
