from typing import Dict, List

import numpy as np
from mmdet.core.evaluation.mean_ap import eval_map
from torch import Tensor
from torchmetrics import Metric

from torchok.constructor import METRICS


@METRICS.register_class
class MMDetectionMAP(Metric):
    """Mean Average Precision metric for detection, which use mmdetection function.

    This class compute mAP with numpy so, it converts torch tensors to numpy before compute.
    """

    def __init__(self, num_classes: int, iou_thr: float = 0.5, nproc: int = 4, scale_ranges=None):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thr = iou_thr
        self.nproc = nproc
        self.scale_ranges = scale_ranges
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]):
        """Update function for mAP. It computes metric for batch and save result to compute

        Args:
            preds: Model prediction, each dict should contain `bboxes` with value Tensor (m, 5) where the 5-th value is
                score, and `labels` with value Tensor (m).
            target: Ground truth, each dict should contain `bboxes` with value Tensor (m, 5), and `labels` with value
                Tensor (m).
        """
        mmdet_format_pred = []
        for pred in preds:
            curr_img_pred = []
            curr_labels_pred = pred['labels'].detach().cpu().float().numpy()
            curr_boxes_pred = pred['bboxes'].detach().cpu().float().numpy()
            for cls_id in range(self.num_classes):
                label_indexes = np.where(curr_labels_pred == cls_id)[0]
                if len(label_indexes) != 0:
                    boxes = curr_boxes_pred[label_indexes]
                else:
                    boxes = np.empty((0, 5), dtype=np.float32)
                curr_img_pred.append(boxes)
            mmdet_format_pred.append(curr_img_pred)

        self.preds += mmdet_format_pred

        np_target = []
        for targ in target:
            np_target.append(dict(
                bboxes=targ['bboxes'].detach().cpu().float().numpy(),
                labels=targ['labels'].detach().cpu().float().numpy()
            ))

        self.targets += np_target

    def compute(self):
        return eval_map(self.preds, self.targets, iou_thr=self.iou_thr, scale_ranges=self.scale_ranges,
                        nproc=self.nproc, logger='silent')[0]
