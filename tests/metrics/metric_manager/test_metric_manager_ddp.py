import unittest
import os
from src.constructor import METRICS
from src.metrics.metric_manager import MetricParams, MetricManager, Phase

from pytorch_lightning import LightningModule, Trainer
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Metric, Accuracy

from torchvision import transforms
from torchvision.datasets import MNIST

from typing import *

INPUT_DATA_SHAPE = 16
BATCH_SIZE = 4
EPOCH = 5

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ]

predicts = [0, 0, 1, 3, 3, 4, 5, 6, 7, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 
     8, 8, 9, 9, 7, 7, 8, 8, 8, 8, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 9
]

uniq_label_count = 10
accuracy_answer = 0.18


class FakeData(Dataset):
    def __init__(self, labels=labels, predicts=predicts):
        self.labels = labels
        self.predicts = predicts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return torch.rand(INPUT_DATA_SHAPE), torch.tensor(self.predicts[item]), torch.tensor(self.labels[item])


@METRICS.register_class
class MetricMemoryBlock(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("memory_list", default=[], dist_reduce_fx=None)

    def update(self, state: torch.Tensor):
        self.memory_list.append(state)

    def compute(self):
        return torch.tensor((torch.cat(self.memory_list)).shape[0])


class Model(LightningModule):
    def __init__(self, metric_params: List[MetricParams]):
        super().__init__()
        self.l1 = torch.nn.Linear(INPUT_DATA_SHAPE, uniq_label_count)
        self.metric_manager = MetricManager(metric_params)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        input, fake_predict, fake_target = batch
        predict = self(input)
        loss = F.cross_entropy(predict, fake_target)
        # set fake data to output, to check metrics
        output = dict(predict=fake_predict, target=fake_target)
        self.metric_manager(Phase.TRAIN, **output)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def run_model(metric_params: List[MetricParams]):
    train_ds = FakeData()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    mnist_model = Model(metric_params)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="cpu",
        strategy="ddp", 
        num_processes=2,
        max_epochs=EPOCH,
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)

    metric_manager_answer = mnist_model.metric_manager.on_epoch_end(Phase.TRAIN)
    return metric_manager_answer
    

class TestCase:
    def __init__(self, test_name: str, metric_params: List[MetricParams], expected):
        self.test_name = test_name
        self.metric_params = metric_params
        self.expected = expected


class DDPMetricManagerTest(unittest.TestCase):
    METRICS.register_class(Accuracy)
    accuracy_mapping = dict(preds='predict', target='target')
    accuracy_params = MetricParams(
        name='Accuracy', mapping=accuracy_mapping,  
        )
    accuracy_answer = {'train/Accuracy': accuracy_answer}

    memory_bank_mapping = dict(state='predict')
    memory_block_params = MetricParams(
        name='MetricMemoryBlock', mapping=memory_bank_mapping,
    )
    memory_block_answer = {'train/MetricMemoryBlock': len(labels) * EPOCH}
    
    def test(self):
        testcases = [
            TestCase(
                test_name='Accuracy', metric_params=[self.accuracy_params], expected=self.accuracy_answer
            ),
            TestCase(
                test_name='MetricMemoryBlock', metric_params=[self.memory_block_params], expected=self.memory_block_answer
            ),
        ]
        for case in testcases:
            actual = run_model(
                metric_params=case.metric_params
            )
            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.test_name, case.expected, actual
                ),
            )


if __name__ == '__main__':
    unittest.main()
    