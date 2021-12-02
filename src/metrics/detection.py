from src.registry import METRICS
from .common import Metric
import torch
from torchmetrics import MAP
from src.models.losses.iou import bbox_overlaps

@METRICS.register_class
class MeanAveragePrecision(Metric):
    def __init__(self, name, target_fields = None, metric_name='map_50'):
        #metric names from https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/detection/map.py
        super().__init__(name=name, target_fields=target_fields)
        self.metric_name = metric_name
        self.map = MAP()
        self.use_gpu = True
        self.use_torch = True

    def calculate(self, target, prediction):
        # Update metric with predictions and respective ground truth
        self.map.update(prediction, target)
        result = self.map.compute()
        if result is None:
            return 0
        return result[self.metric_name]

    def update(self, target, prediction, *args, **kwargs):
        """Updates metric buffer"""
        batch_size = prediction['boxes'].shape[0]
        value = self.calculate([target], [prediction]) * batch_size
        self.mean = (self.n * self.mean + float(value)) / (self.n + batch_size)
        self.n += batch_size


@METRICS.register_class
class BinaryBetaMeanAveragePrecision(Metric):
    """
    F Betta score for binary detection task
    """
    def __init__(self, \
        name, target_fields=None, \
            beta=2, min_iou_th=0.3, max_iou_th=0.85, step=0.05):
        super().__init__(name=name, target_fields=target_fields)

        self.beta = beta
        self.min_iou_th = min_iou_th
        self.max_iou_th = max_iou_th
        self.step = step

    def f_beta(self, tp, fp, fn):
        numerator = (1 + self.beta**2) * tp
        denominator = ((1 + self.beta**2) * tp + self.beta**2 * fn + fp)
        score = numerator / denominator
        return score

    def score_from_iou(self, iou_th, ious):
        

    def calculate(self, target, prediction):
        """
        target: (N, 4) torch.tensor in [xmin, ymin, xmax, ymax] format
        prediction: (N, 5) torch.tensor in [confidance, xmin, ymin, xmax, ymax] format
        """
        # sort by confidance
        indexes = prediction[:,0].argsort()[::-1]
        target = target[indexes]
        prediction = prediction[indexes]

        b_s = target.shape[0]
        ious = [bbox_overlaps(target[i], prediction[i][1:] ,is_aligned=True) for i in range(b_s)]
        # for iou_th in torch.arange(self.min_iou_th, self.max_iou_th, self.step):
            
    