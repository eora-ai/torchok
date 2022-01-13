# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool
from src.registry import METRICS
from .common import Metric
import torch
from torchmetrics import MAP


# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float16)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float16)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None,
                 use_legacy_coordinate=True):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Default: None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """

    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float16)
    fp = np.zeros((num_scales, num_dets), dtype=np.float16)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (
                det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
                    det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps(
        det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (
                    bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_class_bboxes(bboxes_with_labels, class_id, size=4):
    class_bboxes = []
    for i in range(len(bboxes_with_labels)):
        if bboxes_with_labels[i][0].shape[0] == 0:
            class_bboxes.append(np.empty((0, size)))
        else:
            indexes = np.where(bboxes_with_labels[i][1] == class_id)[0]
            if len(indexes) != 0:
                bboxes = bboxes_with_labels[i][0][indexes]
            else:
                bboxes = np.empty((0, size))
            class_bboxes.append(bboxes)
    class_bboxes = np.array(class_bboxes)

    return class_bboxes


def get_cls_results(pred_bboxes, target_bboxes, class_id):
    """Get det results and gt information of a certain class.
    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.
    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = get_class_bboxes(pred_bboxes, class_id, size=4)

    cls_gts = get_class_bboxes(target_bboxes, class_id)

    return cls_dets, cls_gts


def eval_map(pred_bboxes, target_bboxes, num_classes, nproc=4, iou_thr=0.5):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
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
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """

    for i in range(len(pred_bboxes)):
        pred_bboxes[i][0] = pred_bboxes[i][0].detach().cpu().numpy()
        pred_bboxes[i][1] = pred_bboxes[i][1].detach().cpu().numpy()
    

    clear_target_bboxes = []
    for i in range(len(target_bboxes)):
        t_bboxes = target_bboxes[i][0].detach().cpu().numpy()
        t_labels = target_bboxes[i][1].detach().cpu().numpy()
        clear_bboxes = []
        clear_target = []
        for j in range(len(t_labels)):
            if t_labels[j] != -1:
                clear_target.append(t_labels[j])
                clear_bboxes.append(t_bboxes[j])

        clear_bboxes = np.array(clear_bboxes)
        clear_target = np.array(clear_target)
        clear_target_bboxes.append([clear_bboxes, clear_target])

    num_imgs = len(pred_bboxes)
    
    # pool = Pool(nproc)
    eval_results = []
    for i in range(1, num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts = get_cls_results(pred_bboxes, clear_target_bboxes, i)
        cls_gts_ignore = [np.empty((0, 4), dtype=np.float16) for _ in range(len(cls_gts))]

        # tpfp = pool.starmap(
        #     tpfp_default,
        #     zip(cls_dets, cls_gts, cls_gts_ignore, [iou_thr for _ in range(num_imgs)]))
        # tp, fp = tuple(zip(*tpfp))
        tpfp = []
        for img_num in range(num_imgs):
            tp, fp = tpfp_default(cls_dets[img_num], cls_gts[img_num], cls_gts_ignore[img_num], iou_thr)
            tpfp.append([tp, fp])
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(1, dtype=int)
        for j, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]
            
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float16).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # calculate AP
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        num_gts = num_gts.item()
      
        ap = average_precision(recalls, precisions)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])

    mean_ap = np.array(aps).mean().item() if aps else 0.0

    del (clear_target_bboxes[:], eval_results[:], aps[:], pred_bboxes[:])
    del (clear_target_bboxes, eval_results, aps, pred_bboxes, tp, fp)

    return mean_ap

@METRICS.register_class
class MeanAveragePrecision(Metric):
    def __init__(self, num_classes, name=None, target_fields=None, nproc=4, iou_thr=0.5):
        if name is None:
            name = 'mAP' + str(int(100*iou_thr))
        super().__init__(name=name, target_fields=target_fields)
        self.nproc = nproc
        self.iou_thr = iou_thr
        self.num_classes = num_classes
        self.use_gpu = True
        self.use_torch = True

    def calculate(self, target, prediction):
        mean_ap = eval_map(pred_bboxes=prediction, target_bboxes=target, \
                                            num_classes=self.num_classes, nproc=self.nproc, iou_thr=self.iou_thr)
        # print('mAP = ' + str(mean_ap))
        # print('target = ' + str(len(target)))
        # print('prediction = ' + str(len(prediction)))
        return mean_ap

    def update(self, target, prediction, *args, **kwargs):
        """Updates metric buffer"""
        batch_size = 1
        # print(prediction[0])
        # print(target)
        value = self.calculate(target, prediction) * batch_size
        self.mean = (self.n * self.mean + value) / (self.n + batch_size)
        self.n += batch_size



# @METRICS.register_class
# class BinaryBetaMeanAveragePrecision(Metric):
#     """
#     F Betta score for binary detection task
#     """
#     def __init__(self, \
#         name, target_fields=None, \
#             beta=2, min_iou_th=0.3, max_iou_th=0.85, step=0.05):
#         super().__init__(name=name, target_fields=target_fields)

#         self.beta = beta
#         self.min_iou_th = min_iou_th
#         self.max_iou_th = max_iou_th
#         self.step = step

#     def f_beta(self, tp, fp, fn):
#         numerator = (1 + self.beta**2) * tp
#         denominator = ((1 + self.beta**2) * tp + self.beta**2 * fn + fp)
#         score = numerator / denominator
#         return score

#     # def score_from_iou(self, iou_th, ious):
        

#     def calculate(self, target, prediction):
#         """
#         target: (N, 4) torch.tensor in [xmin, ymin, xmax, ymax] format
#         prediction: (N, 5) torch.tensor in [confidance, xmin, ymin, xmax, ymax] format
#         """
#         # sort by confidance
#         indexes = prediction[:,0].argsort()[::-1]
#         target = target[indexes]
#         prediction = prediction[indexes]

#         b_s = target.shape[0]
#         ious = [bbox_overlaps(target[i], prediction[i][1:] ,is_aligned=True) for i in range(b_s)]
        # for iou_th in torch.arange(self.min_iou_th, self.max_iou_th, self.step):
            
    