import unittest
import os

from pytorch_lightning import LightningModule, Trainer
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Metric, Accuracy

from torchvision import transforms
from torchvision.datasets import MNIST
from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter
from src.metrics.representation import DatasetType, MetricDistance
from typing import *
import math


BATCH_SIZE = 3
EPOCH = 1

vectors = {
    DatasetType.CLASSIFICATION: torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),

    DatasetType.REPRESENTATION: torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),
}

targets = torch.tensor([0, 1, 1, 2, 2, 1, 0, 0, 3], dtype=torch.int32)

queries_idxs = torch.tensor([0, 1, -1, 2, -1, -1, -1, -1, 3], dtype=torch.int32)

scores = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 2.5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

# 1-4 is top-k for retrieval
classification_dataset_answers = {
    'recall': {
        1: 0.5, 
        2: 2/3, 
        3: 2.5/3,
        4: 1 
    },
    'precision': {
        1: 2/3,
        2: 0.5,
        3: 4/9,
        4: 5/12
    },
    'average_precision': {
        1: 0.5,
        2: 0.5,
        3: 0.7222222222222223,
        4: 0.8055555555555555
    },
    'ndcg': {
        1: 0.6666666666666666,
        2: 0.5436432511904858,
        3: 0.7688578654609097,
        4: 0.8568805729851068
    }
}

representation_dataset_answers = {
    'recall': {
        1: 0.5, 
        2: 2/3, 
        3: 2.5/3,
        4: 1 
    },
    'precision': {
        1: 2/3,
        2: 0.5,
        3: 4/9,
        4: 5/12
    },
    'average_precision': {
        1: 0.5,
        2: 0.5,
        3: 0.5555555555555555,
        4: 0.6388888888888888
    },
    'ndcg': {
        1: 0.6666666666666666,
        2: 0.6152427457535189,
        3: 0.7325624272592081,
        4: 0.7229100454445524
    }
}

name2class = {
    'recall': RecallAtKMeter,
    'precision': PrecisionAtKMeter,
    'average_precision': MeanAveragePrecisionAtKMeter,
    'ndcg': NDCGAtKMeter
}


class FakeData(Dataset):
    def __init__(self, dataset_type=DatasetType.CLASSIFICATION):
        self.vectors = vectors[dataset_type]
        self.targets = targets
        self.scores = scores
        self.queries_idxs = queries_idxs

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        output = {
            'vectors': self.vectors[item],
            'targets': self.targets[item],
            'scores': self.scores[item],
            'queries_idxs': self.queries_idxs[item]
        }
        return output


class TestCase:
    def __init__(self, test_name, dataset_type: DatasetType = DatasetType.CLASSIFICATION, \
            search_batch_size: bool = None, exact_index: bool = True, normalize_vectors: bool = True, \
            metric_distance: MetricDistance = MetricDistance.IP):
        self.params = dict(
            dataset_type = dataset_type,
            search_batch_size = search_batch_size,
            exact_index = exact_index,
            normalize_vectors = normalize_vectors,
            metric_distance = metric_distance,
        )
        self.class_name = test_name
        if dataset_type == DatasetType.CLASSIFICATION:
            self.expected = classification_dataset_answers[test_name]
        else:
            self.expected = representation_dataset_answers[test_name]


class Model(LightningModule):
    def __init__(self, test_case: TestCase):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.dataset = test_case.params['dataset_type']
        metric_class = name2class[test_case.class_name]
        self.metrics = [metric_class(**test_case.params, k=k) for k in range(1, 5)]

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        vectors = batch['vectors']
        predict = self(vectors.float())
        loss = F.cross_entropy(predict, torch.zeros(predict.shape[0], dtype=torch.long))
        # set fake data to output, to check metrics
        for metric in self.metrics:
            if self.dataset == DatasetType.CLASSIFICATION:
                metric(vectors=batch['vectors'], targets=batch['targets'])
            else:
                metric(vectors=batch['vectors'], scores=batch['scores'], queries_idxs=batch['queries_idxs'])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run_model(test_case: TestCase):
    train_ds = FakeData(test_case.params['dataset_type'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    model = Model(test_case)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="cpu",
        strategy="ddp", 
        num_processes=3,
        max_epochs=EPOCH,
    )

    # Train the model ⚡
    trainer.fit(model, train_loader)

    metric_dict = {}
    for k, metric in enumerate(model.metrics):
        metric_dict[k + 1] = metric.compute()

    return metric_dict


class TestDDPRepresentationMetrics(unittest.TestCase):
    def test_recall_when_dataset_is_representation(self):
        case = TestCase(test_name='recall', dataset_type=DatasetType.REPRESENTATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_precision_when_dataset_is_representation(self):
        case = TestCase(test_name='precision', dataset_type=DatasetType.REPRESENTATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_average_precision_when_dataset_is_representation(self):
        case = TestCase(test_name='average_precision', dataset_type=DatasetType.REPRESENTATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_ndcg_when_dataset_is_representation(self):
        case = TestCase(test_name='ndcg', dataset_type=DatasetType.REPRESENTATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    # all metrics with classification dataset failed the tests
    # TODO: overwrite Classification Dataset.
    def test_recall_when_dataset_is_classification(self):
        case = TestCase(test_name='recall', dataset_type=DatasetType.CLASSIFICATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_precision_when_dataset_is_classification(self):
        case = TestCase(test_name='precision', dataset_type=DatasetType.CLASSIFICATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_average_precision_when_dataset_is_classification(self):
        case = TestCase(test_name='average_precision', dataset_type=DatasetType.CLASSIFICATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])

    def test_ndcg_when_dataset_is_classification(self):
        case = TestCase(test_name='ndcg', dataset_type=DatasetType.CLASSIFICATION)
        for k in range(1, 5):
            actual = run_model(case)
            assert math.isclose(case.expected[k], actual[k])


if __name__ == '__main__':
    unittest.main()
