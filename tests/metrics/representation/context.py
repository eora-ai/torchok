import torch

from typing import Dict, Optional
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import VECTORS, TARGETS, SCORES, QUERIES_IDX


MAX_K = 6
BATCH_SIZE = 1
EPOCH = 1


class RepresentationData(Dataset):
    def __init__(self, vectors: Tensor = VECTORS, scores: Tensor = SCORES, queries_idxs: Tensor = QUERIES_IDX):
        self.vectors = vectors
        self.scores = scores
        self.queries_idxs = queries_idxs

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        output = {
            'vectors': self.vectors[item],
            'scores': self.scores[item],
            'queries_idxs': self.queries_idxs[item]
        }
        return output


class ClassificationData(Dataset):
    def __init__(self, vectors: Tensor = VECTORS, targets: Tensor = TARGETS):
        self.vectors = vectors
        self.targets = targets

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        output = {
            'vectors': self.vectors[item],
            'targets': self.targets[item],
        }
        return output


class Model(LightningModule):
    def __init__(self, metric_class: type, metric_params: Dict, max_k: int = MAX_K):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.dataset = metric_params['dataset_type']
        # reset to 1
        self.metrics = [metric_class(**metric_params, k=k) for k in range(1, max_k + 1)]

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        vectors = batch['vectors']
        predict = self(vectors.float())
        loss = F.cross_entropy(predict, torch.zeros(predict.shape[0], dtype=torch.long))
        # set fake data to output, to check metrics
        for metric in self.metrics:
            if self.dataset == 'classification':
                # classification
                metric.update(vectors=batch['vectors'], targets=batch['targets'])
            else:
                # representation
                metric.update(vectors=batch['vectors'], scores=batch['scores'], query_numbers=batch['queries_idxs'])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run_model(metric_class: type, metric_params: Dict, trainer_params: Optional[Dict] = None, max_k: int = MAX_K):
    if metric_params['dataset_type'] == 'classification':
        train_ds = ClassificationData()
    else:
        train_ds = RepresentationData()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    model = Model(metric_class, metric_params, max_k)

    # Initialize a trainer
    if trainer_params is None:
        trainer_params = {}

    trainer_params['num_sanity_val_steps'] = 0
        
    trainer = Trainer(**trainer_params, max_epochs=EPOCH)

    # Train the model ⚡
    trainer.fit(model, train_loader)

    metric_dict = {}
    for k, metric in enumerate(model.metrics):
        metric_dict[k + 1] = metric.compute()

    return metric_dict
    