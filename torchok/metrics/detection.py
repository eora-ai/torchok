import torch
import numpy as np

from torch import Tensor
from typing import List, Dict
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mmdet.core.evaluation.mean_ap import eval_map

from torchok.constructor import METRICS


@METRICS.register_class
class MeanAveragePrecisionX(MeanAveragePrecision):
    """Mean Average Precision metric for detection, which prediction scores inside bbox tensor."""

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):
                - ``boxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 5]`` containing ``num_boxes`` detection boxes
                  of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax, score]`` in absolute image coordinates.
                - ``labels``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed detection classes
                  for the boxes.
                - ``masks``: ``torch.bool`` of shape ``[num_boxes, image_height, image_width]`` containing boolean
                  masks. Only required when `iou_type="segm"`.
            target: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):
                - ``bboxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 4]`` containing ``num_boxes``
                  ground truth boxes of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax]`` in absolute image coordinates.
                - ``label``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed ground truth
                   classes for the boxes.
                - ``masks``: ``torch.bool`` of shape ``[num_boxes, image_height, image_width]`` containing boolean
                  masks. Only required when `iou_type="segm"`.
        Raises:
            ValueError:
                If ``preds`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``target`` is not of type ``List[Dict[str, Tensor]]``
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        for i in range(len(preds)):
            bboxes_with_scores = preds[i].pop('boxes')
            bboxes, scores = torch.split(bboxes_with_scores, [4, 1], -1)
            scores = scores.squeeze(-1)
            preds[i]['boxes'] = bboxes
            preds[i]['scores'] = scores

        super().update(preds, target)


@METRICS.register_class
class MMDetectionMAP(Metric):
    """Mean Average Precision metric for detection, which use mmdetection function.

    This class compute mAP with numpy so, it convert torch tensors to numpy before compute.
    """
    def __init__(self, num_classes: int, iou_thr: float = 0.5, nproc: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thr = iou_thr
        self.nproc = nproc
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("map", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]):
        """Update function for mAP. It compute metric for batch and save result to compute 

        Args:
            preds: Model prediction, each dict should contain `bboxes` with value Tensor (m, 5) where the 5-th value is
                score, and `labels` with value Tensor (m).
            target: Ground truth, each dict should contain `bboxes` with value Tensor (m, 5), and `labels` with value
                Tensor (m).
        """
        mmdet_format_pred = []
        for pred in preds:
            curr_img_pred = []
            curr_labels_pred = pred['labels'].detach().cpu().numpy()
            curr_boxes_pred = pred['boxes'].detach().cpu().numpy()
            for cls_id in range(self.num_classes):
                label_indexes = np.where(curr_labels_pred==cls_id)[0]
                if len(label_indexes) != 0:
                    boxes = curr_boxes_pred[label_indexes]
                else:
                    boxes = np.empty((0, 5), dtype=np.float32)
                curr_img_pred.append(boxes)
            mmdet_format_pred.append(curr_img_pred)

        for i in range(len(target)):
            target[i]['bboxes'] = target[i]['bboxes'].detach().cpu().numpy()
            target[i]['labels'] = target[i]['labels'].detach().cpu().numpy()

        self.map += eval_map(mmdet_format_pred, target, iou_thr=self.iou_thr, nproc=self.nproc)[0]
        self.total += 1

    def compute(self):
        return self.map.float() / self.total
