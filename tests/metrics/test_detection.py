import unittest

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchok.metrics.detection import MeanAveragePrecisionX


class DetectionMetricTest(unittest.TestCase):
    def test_mean_average_precision_x(self):
        torchmetrics_map = MeanAveragePrecision()
        preds = [
            dict(
                boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=torch.tensor([0.536]),
                labels=torch.tensor([0]),
            )
        ]
        target = [
            dict(
                boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                labels=torch.tensor([0]),
            )
        ]

        torchmetrics_map.update(preds, target)
        torchmetrics_answer = torchmetrics_map.compute()

        map_x = MeanAveragePrecisionX()
        map_x_preds = [
            dict(
                bboxes=torch.tensor([[258.0, 41.0, 606.0, 285.0, 0.536]]),
                label=torch.tensor([0]),
            )
        ]
        map_x_target = [
            dict(
                bboxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                label=torch.tensor([0]),
            )
        ]

        map_x.update(map_x_preds, map_x_target)
        map_x_answer = map_x.compute()

        self.assertDictEqual(
            map_x_answer,
            torchmetrics_answer,
            "failed test detection metric mapx expected {}, actual {}".format(
                torchmetrics_answer, map_x_answer
            ),
        )


if __name__ == '__main__':
    unittest.main()
