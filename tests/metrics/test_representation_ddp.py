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
from typing import *
import math



BATCH_SIZE = 3
EPOCH = 1

vectors = {
    'classification': torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),

    'representation': torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], # queries
        [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], # database
    ])
}

targets = {
    'classification': torch.tensor([0, 1, 1, 2, 2, 1, 0, 0, 3]),
    'representation': torch.tensor([0, 1, 2, 3, 1, 2, 1, 0, 0])
}

is_queries = torch.tensor([True, True, True, True, False, False, False, False, False])

scores = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 2.5, 0, 0],
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0]
    ]
)

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
        3: 0.7222222222222223,
        4: 0.8055555555555555
    },
    'ndcg': {
        1: 0.6666666666666666,
        2: 0.4847038505208375,
        3: 0.6933556704671239,
        4: 0.8099261657231794
    }
}

name2class = {
    'recall': RecallAtKMeter,
    'precision': PrecisionAtKMeter,
    'average_precision': MeanAveragePrecisionAtKMeter,
    'ndcg': NDCGAtKMeter
}


class FakeData(Dataset):
    def __init__(self, dataset='classification'):
        self.vectors = vectors[dataset]
        self.targets = targets[dataset]
        self.scores = scores
        self.is_queries = is_queries

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        output = {
            'vectors': self.vectors[item],
            'targets': self.targets[item],
            'scores': self.scores[item],
            'is_queries': self.is_queries[item]
        }
        return output


class TestCase:
    def __init__(self, test_name, dataset: str = 'classification', search_batch_size: bool = None, \
                 exact_index: bool = True, normalize_input: bool = True, metric: str = 'IP'):
        self.params = dict(
            dataset = dataset,
            search_batch_size = search_batch_size,
            exact_index = exact_index,
            normalize_input = normalize_input,
            metric = metric,
        )
        self.class_name = test_name
        if dataset == 'classification':
            self.expected = classification_dataset_answers[test_name]
        else:
            self.expected = representation_dataset_answers[test_name]


class Model(LightningModule):
    def __init__(self, test_case: TestCase):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.dataset = test_case.params['dataset']
        metric_class = name2class[test_case.class_name]
        self.metrics = [metric_class(**test_case.params, k=k) for k in range(1, 5)]

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        vectors = batch['vectors']
        targets = batch['targets']
        predict = self(vectors.float())
        loss = F.cross_entropy(predict, targets)
        print(batch)
        # set fake data to output, to check metrics
        for metric in self.metrics:
            if self.dataset == 'classification':
                metric(vectors=batch['vectors'], targets=batch['targets'])
            else:
                metric(vectors=batch['vectors'], scores=batch['scores'], is_queries=batch['is_queries'])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run_model(test_case: TestCase):
    train_ds = FakeData(test_case.params['dataset'])
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
        # print(f'metric vectors shape = {torch.cat(metric.vectors).shape}')
        metric_dict[k + 1] = metric.compute()

    return metric_dict


class DDPTestRepresentationMetrics(unittest.TestCase):
    def test_classification_dataset(self):
        test_cases = [
            TestCase(test_name='recall', dataset='representation'),
            # TestCase(test_name='precision'),
            # TestCase(test_name='average_precision'),
            # TestCase(test_name='ndcg'),
        ]

        for case in test_cases:
            actual = run_model(case)
            print(actual)
            print(case.expected)
            name = case.class_name
            for k in range(1, 5):
                assert math.isclose(case.expected[k], actual[k])
            # print(f'Answer = {actual}')
            # self.assertDictEqual(
            #     case.expected,
            #     actual,
            #     "failed test {} expected {}, actual {}".format(
            #         case.class_name, case.expected, actual
            #     ),
            # )


if __name__ == '__main__':
    unittest.main()
