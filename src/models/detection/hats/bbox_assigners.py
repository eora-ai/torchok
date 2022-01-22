# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import torch


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class AssignResult:
    """Stores assignments between predicted and truth boxes.
    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.
        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])


class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        return self.bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'scale={self.scale})'

        return repr_str


    @staticmethod
    def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        """Calculate overlap between two set of bboxes.
        FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
        Note:
            Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
            there are some new generated variable when calculating IOU
            using bbox_overlaps function:
            1) is_aligned is False
                area1: M x 1
                area2: N x 1
                lt: M x N x 2
                rb: M x N x 2
                wh: M x N x 2
                overlap: M x N x 1
                union: M x N x 1
                ious: M x N x 1
                Total memory:
                    S = (9 x N x M + N + M) * 4 Byte,
                When using FP16, we can reduce:
                    R = (9 x N x M + N + M) * 4 / 2 Byte
                    R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                    Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                               N + 1 < 3 * N, when N or M is 1.
                Given M = 40 (ground truth), N = 400000 (three anchor boxes
                in per grid, FPN, R-CNNs),
                    R = 275 MB (one times)
                A special case (dense detection), M = 512 (ground truth),
                    R = 3516 MB = 3.43 GB
                When the batch size is B, reduce:
                    B x R
                Therefore, CUDA memory runs out frequently.
                Experiments on GeForce RTX 2080Ti (11019 MiB):
                |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
                |:----:|:----:|:----:|:----:|:----:|:----:|
                |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
                |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
                |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
                |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |
            2) is_aligned is True
                area1: N x 1
                area2: N x 1
                lt: N x 2
                rb: N x 2
                wh: N x 2
                overlap: N x 1
                union: N x 1
                ious: N x 1
                Total memory:
                    S = 11 x N * 4 Byte
                When using FP16, we can reduce:
                    R = 11 x N * 4 / 2 Byte
            So do the 'giou' (large than 'iou').
            Time-wise, FP16 is generally faster than FP32.
            When gpu_assign_thr is not -1, it takes more time on cpu
            but not reduce memory.
            There, we can reduce half the memory and keep the speed.
        If ``is_aligned`` is ``False``, then calculate the overlaps between each
        bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
        pair of bboxes1 and bboxes2.
        Args:
            bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
            bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection over
                foreground) or "giou" (generalized intersection over union).
                Default "iou".
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Default 1e-6.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        Example:
            >>> bboxes1 = torch.FloatTensor([
            >>>     [0, 0, 10, 10],
            >>>     [10, 10, 20, 20],
            >>>     [32, 32, 38, 42],
            >>> ])
            >>> bboxes2 = torch.FloatTensor([
            >>>     [0, 0, 10, 20],
            >>>     [0, 10, 10, 19],
            >>>     [10, 10, 20, 20],
            >>> ])
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
            >>> assert overlaps.shape == (3, 3)
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
            >>> assert overlaps.shape == (3, )
        Example:
            >>> empty = torch.empty(0, 4)
            >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
            >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
            >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
            >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
        """

        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        # Either the boxes are empty or the length of boxes' last dimension is 4
        assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            if is_aligned:
                return bboxes1.new(batch_shape + (rows, ))
            else:
                return bboxes1.new(batch_shape + (rows, cols))

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

        if is_aligned:
            lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
            rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1 + area2 - overlap
            else:
                union = area1
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        else:
            lt = torch.max(bboxes1[..., :, None, :2],
                           bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
            rb = torch.min(bboxes1[..., :, None, 2:],
                           bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1[..., None] + area2[..., None, :] - overlap
            else:
                union = area1[..., None]
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                        bboxes2[..., None, :, :2])
                enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                        bboxes2[..., None, :, 2:])

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        if mode in ['iou', 'iof']:
            return ious
        # calculate gious
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

        return gious


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.
    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.
    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt
    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.match_low_quality = match_low_quality
        self.iou_calculator = BboxOverlaps2D()

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.
        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.
        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
