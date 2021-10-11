import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from src.constructor import create_backbone
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, POOLINGS
from .base_task import BaseTask


class MoBYParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    pooling_name: str = 'Pooling'
    # in_features is taken from backbone.num_features
    # Be careful, default pooling parameters dict is not full!
    pooling_params: dict = {}
    momentum: float = 0.99
    temperature: float = 0.2
    memory_size: int = 4096
    proj_num_layers: int = 2
    pred_num_layers: int = 2
    distributed_mode: bool = False


@TASKS.register_class
class MoBYTask(BaseTask):
    config_parser = MoBYParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)

        self.momentum = self.params.momentum
        self.temperature = self.params.temperature
        self.memory_size = self.params.memory_size

        proj_num_layers = self.params.proj_num_layers
        pred_num_layers = self.params.pred_num_layers

        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)
        self.backbone_k = create_backbone(model_name=self.params.backbone_name,
                                          **self.params.backbone_params)
        self.num_features = self.backbone.num_features
        self.params.pooling_params['in_features'] = self.num_features
        self.pooling = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)
        self.pooling_k = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)

        self.projector = MoBYMLP(in_dim=self.pooling.out_features, num_layers=proj_num_layers)
        self.projector_k = MoBYMLP(in_dim=self.pooling.out_features, num_layers=proj_num_layers)

        self.predictor = MoBYMLP(num_layers=pred_num_layers)

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.backbone_k.eval()

        for param_q, param_k in zip(self.pooling.parameters(), self.pooling_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.pooling_k.eval()

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.projector_k.eval()

        # create the queue
        self.register_buffer("queue1", torch.randn(256, self.memory_size, requires_grad=False))
        self.register_buffer("queue2", torch.randn(256, self.memory_size, requires_grad=False))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def train(self, mode: bool = True):
        self.training = mode
        for block in [self.backbone, self.pooling, self.projector, self.predictor]:
            for module in block.children():
                module.train(mode)
        return self

    def on_train_start(self) -> None:
        super(MoBYTask, self).on_train_start()
        max_epochs = self.hparams.trainer.max_epochs
        self.K = len(self.train_dataloader()) * max_epochs

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = (1 - self.momentum) * (np.cos(np.pi * self.global_step / self.K) + 1) / 2

        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * (1. - momentum) + param_q.data * momentum

        for param_q, param_k in zip(self.pooling.parameters(), self.pooling_k.parameters()):
            param_k.data = param_k.data * (1. - momentum) + param_q.data * momentum

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * (1. - momentum) + param_q.data * momentum

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        if self.params.distributed_mode:
            keys1 = dist_collect(keys1)
            keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.memory_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.memory_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pooling(feat)  # queries: NxC
        proj = self.projector(feat)
        pred = self.predictor(proj)
        return pred

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        im_1 = batch['input_0']
        im_2 = batch['input_1']

        feat_1 = self.pooling(self.backbone(im_1))  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.pooling(self.backbone(im_2))
        proj_2 = self.projector(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.pooling_k(self.backbone_k(im_1))  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.pooling_k(self.backbone_k(im_2))
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        if pred_1.device != self.queue2.device or pred_1.dtype != self.queue2.dtype:
            self.queue2 = self.queue2.to(pred_2)

        if pred_2.device != self.queue1.device or pred_2.dtype != self.queue1.dtype:
            self.queue1 = self.queue1.to(pred_2)

        # compute loss
        l1 = self.contrastive_loss(pred_1, proj_2_ng, self.queue2)
        l2 = self.contrastive_loss(pred_2, proj_1_ng, self.queue1)
        loss = l1 + l2

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

        return loss

    def validation_step(self, batch, batch_idx):
        batch['embeddings'] = self.pooling(self.backbone(batch['input']))
        self.metric_manager.update('valid', **batch)
        return torch.zeros(1).to(batch['input'])

    def test_step(self, batch, batch_idx):
        batch['embeddings'] = self.pooling(self.backbone(batch['input']))
        self.metric_manager.update('test', **batch)
        return torch.zeros(1).to(batch['input'])


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        if num_layers == 0:
            self.layers = nn.Identity()
        else:
            linear_hidden = []
            for i in range(num_layers - 1):
                linear_hidden.append(nn.Linear(in_dim, inner_dim))
                linear_hidden.append(nn.BatchNorm1d(inner_dim))
                linear_hidden.append(nn.ReLU(inplace=True))
                in_dim = inner_dim
            self.layers = nn.Sequential(*linear_hidden, nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.layers(x)


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    from diffdist import functional
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in
                range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()
