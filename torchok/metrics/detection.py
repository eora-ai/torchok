from typing import Dict, List

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP

from torchok.constructor import METRICS


def calculate_recall_precision_scores(
        recall: Tensor,
        precision: Tensor,
        scores: Tensor,
        idx_cls: int,
        idx_bbox_area: int,
        idx_max_det_thrs: int,
        eval_imgs: list,
        rec_thresholds: Tensor,
        max_det: int,
        nb_imgs: int,
        nb_bbox_areas: int,
):
    nb_rec_thrs = len(rec_thresholds)
    idx_cls_pointer = idx_cls * nb_bbox_areas * nb_imgs
    idx_bbox_area_pointer = idx_bbox_area * nb_imgs
    # Load all image evals for current class_id and area_range
    img_eval_cls_bbox = [eval_imgs[idx_cls_pointer + idx_bbox_area_pointer + i] for i in range(nb_imgs)]
    img_eval_cls_bbox = [e for e in img_eval_cls_bbox if e is not None]
    if not img_eval_cls_bbox:
        return recall, precision, scores

    det_scores = torch.cat([e["dtScores"][:max_det].bool() for e in img_eval_cls_bbox])

    # different sorting method generates slightly different results.
    # mergesort is used to be consistent as Matlab implementation.
    # Sort in PyTorch does not support bool types on CUDA (yet, 1.11.0)
    dtype = torch.uint8 if det_scores.is_cuda and det_scores.dtype is torch.bool else det_scores.dtype
    # Explicitly cast to uint8 to avoid error for bool inputs on CUDA to argsort
    inds = torch.argsort(det_scores.to(dtype), descending=True)
    det_scores_sorted = det_scores[inds]

    det_matches = torch.cat([e["dtMatches"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
    det_ignore = torch.cat([e["dtIgnore"][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
    gt_ignore = torch.cat([e["gtIgnore"] for e in img_eval_cls_bbox])
    npig = torch.count_nonzero(gt_ignore == False)  # noqa: E712
    if npig == 0:
        return recall, precision, scores
    tps = torch.logical_and(det_matches, torch.logical_not(det_ignore))
    fps = torch.logical_and(torch.logical_not(det_matches), torch.logical_not(det_ignore))

    tp_sum = torch.cumsum(tps, axis=1, dtype=torch.float)
    fp_sum = torch.cumsum(fps, axis=1, dtype=torch.float)
    for idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
        prec = torch.zeros((nb_rec_thrs,))
        score = torch.zeros((nb_rec_thrs,))

        recall[idx, idx_cls, idx_bbox_area, idx_max_det_thrs] = rc[-1] if nd else 0

        # Remove zigzags for AUC
        diff_zero = torch.zeros((1,), device=pr.device)
        diff = torch.ones((1,), device=pr.device)
        while not torch.all(diff == 0):
            diff = torch.clamp(torch.cat(((pr[1:] - pr[:-1]), diff_zero), 0), min=0)
            pr += diff

        inds = torch.searchsorted(rc, rec_thresholds.to(rc.device), right=False)
        num_inds = inds.argmax() if inds.max() >= nd else nb_rec_thrs
        inds = inds[:num_inds]
        prec[:num_inds] = pr[inds]
        score[:num_inds] = det_scores_sorted[inds]
        precision[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = prec
        scores[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = score

    return recall, precision, scores


@METRICS.register_class
class MeanAveragePrecision(MAP):
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

    def _calculate(self, class_ids: List):
        """Calculate the precision and recall for all supplied classes to calculate mAP/mAR.

        Args:
            class_ids:
                List of label class Ids.
        """
        img_ids = range(len(self.groundtruths))
        max_detections = self.max_detection_thresholds[-1]
        area_ranges = self.bbox_area_ranges.values()

        ious = {
            (idx, class_id): self._compute_iou(idx, class_id, max_detections)
            for idx in img_ids
            for class_id in class_ids
        }

        eval_imgs = [
            self._evaluate_image(img_id, class_id, area, max_detections, ious)
            for class_id in class_ids
            for area in area_ranges
            for img_id in img_ids
        ]

        nb_iou_thrs = len(self.iou_thresholds)
        nb_rec_thrs = len(self.rec_thresholds)
        nb_classes = len(class_ids)
        nb_bbox_areas = len(self.bbox_area_ranges)
        nb_max_det_thrs = len(self.max_detection_thresholds)
        nb_imgs = len(img_ids)
        precision = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        recall = -torch.ones((nb_iou_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))
        scores = -torch.ones((nb_iou_thrs, nb_rec_thrs, nb_classes, nb_bbox_areas, nb_max_det_thrs))

        # move tensors if necessary
        rec_thresholds_tensor = torch.tensor(self.rec_thresholds)

        # retrieve E at each category, area range, and max number of detections
        for idx_cls, _ in enumerate(class_ids):
            for idx_bbox_area, _ in enumerate(self.bbox_area_ranges):
                for idx_max_det_thrs, max_det in enumerate(self.max_detection_thresholds):
                    recall, precision, scores = calculate_recall_precision_scores(
                        recall,
                        precision,
                        scores,
                        idx_cls=idx_cls,
                        idx_bbox_area=idx_bbox_area,
                        idx_max_det_thrs=idx_max_det_thrs,
                        eval_imgs=eval_imgs,
                        rec_thresholds=rec_thresholds_tensor,
                        max_det=max_det,
                        nb_imgs=nb_imgs,
                        nb_bbox_areas=nb_bbox_areas,
                    )

        return precision, recall
