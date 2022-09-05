import torch
import numpy as np

from torch import Tensor
from typing import List, Dict
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool
from mmdet.core.evaluation.mean_ap import tpfp_imagenet, average_precision, get_cls_group_ofs,\
    tpfp_openimages, tpfp_default, get_cls_results

from torchok.constructor import METRICS


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             ioa_thr=None,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4,
             use_legacy_coordinate=False,
             use_group_of=False):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Default: None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Default: False.
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # There is no need to use multi processes to process
    # when num_imgs = 1 .
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            elif dataset in ['oid_challenge', 'oid_v6'] \
                    or use_group_of is True:
                tpfp_fn = tpfp_openimages
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        if num_imgs > 1:
            # compute tp and fp for each image with multiple processes
            args = []
            if use_group_of:
                # used in Open Images Dataset evaluation
                gt_group_ofs = get_cls_group_ofs(annotations, i)
                args.append(gt_group_ofs)
                args.append([use_group_of for _ in range(num_imgs)])
            if ioa_thr is not None:
                args.append([ioa_thr for _ in range(num_imgs)])

            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)],
                    [use_legacy_coordinate for _ in range(num_imgs)], *args))
        else:
            tpfp = tpfp_fn(
                cls_dets[0],
                cls_gts[0],
                cls_gts_ignore[0],
                iou_thr,
                area_ranges,
                use_legacy_coordinate,
                gt_bboxes_group_of=(get_cls_group_ofs(annotations, i)[0]
                                    if use_group_of else None),
                use_group_of=use_group_of,
                ioa_thr=ioa_thr)
            tpfp = [tpfp]

        if use_group_of:
            tp, fp, cls_dets = tuple(zip(*tpfp))
        else:
            tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
                    bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    if num_imgs > 1:
        pool.close()

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    return mean_ap, eval_results


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
            curr_boxes_pred = pred['bboxes'].detach().cpu().numpy()
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
