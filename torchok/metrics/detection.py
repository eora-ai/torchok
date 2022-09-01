import torch
from torch import Tensor
from typing import List, Dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchok.constructor import METRICS


@METRICS.register_class
class MeanAveragePrecisionX(MeanAveragePrecision):
    """Mean Average Precision metric for detection, which prediction scores inside bbox tensor."""

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):
                - ``bboxes``: ``torch.FloatTensor`` of shape ``[num_boxes, 5]`` containing ``num_boxes`` detection boxes
                  of the format specified in the constructor. By default, this method expects
                  ``[xmin, ymin, xmax, ymax, score]`` in absolute image coordinates.
                - ``label``: ``torch.IntTensor`` of shape ``[num_boxes]`` containing 0-indexed detection classes
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
            bboxes_with_scores = preds[i].pop('bboxes')
            bboxes, scores = torch.split(bboxes_with_scores, [4, 1], -1)
            preds[i]['boxes'] = bboxes
            preds[i]['scores'] = scores
            preds[i]['labels'] = preds[i].pop('label')

        for i in range(len(target)):
            target[i]['boxes'] = target[i].pop('bboxes')
            target[i]['labels'] = target[i].pop('label')

        super().update(preds, target)
