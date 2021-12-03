# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from src.registry import DETECTION_HAT

from .point_generator import MlvlPointGenerator
from .assigner import SimOTAAssigner


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@DETECTION_HAT.register_class
class YOLOXHat(nn.Module):
    """
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        if num_classes=1, then label in csv file must be = 0 
    """
    def __init__(self,
                num_classes,
                strides=[8, 16, 32],
                input_size = (640, 640),
                conf_thr=0.65,
                nms_cfg=dict(type='nms', iou_threshold=0.65)
                ):
        super(YOLOXHat, self).__init__()

        # get priors points
        featmap_sizes = [torch.Size([int(input_size[0]/stride), int(input_size[1]/stride)]) for stride in strides]
        prior_generator = MlvlPointGenerator(strides, featmap_sizes, offset=0, with_stride=True)
        self.flatten_priors = prior_generator.flatten_priors

        # create assigner
        self.assigner = SimOTAAssigner(center_radius=2.5)

        self.num_classes = num_classes
        # confidance for inference
        self.conf_thr = conf_thr
        # nms config params
        self.nms_cfg = nms_cfg

        
    def _get_flatten_output(self,
                            cls_scores,
                            bbox_preds,
                            objectnesses,
                            ):

        num_imgs = bbox_preds[0].shape[0]
        
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)

        flatten_bboxes = self._bbox_decode(self.flatten_priors, flatten_bbox_preds)

        return flatten_cls_preds, flatten_objectness, flatten_bboxes, flatten_bbox_preds
        
    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)

        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, conf_thr = 0.65):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= conf_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, dict(type='nms', iou_threshold=0.65))
            return dets, labels[keep]

    def forward_infer(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   ):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        flatten_cls_preds, flatten_objectness, flatten_bboxes, flatten_bbox_preds = self._get_flatten_output(
                                                                                            cls_scores=cls_scores,
                                                                                            bbox_preds=bbox_preds,
                                                                                            objectnesses=objectnesses
                                                                                            )
        result_list = []
        flatten_cls_preds = flatten_cls_preds.sigmoid()
        flatten_objectness = flatten_objectness.sigmoid()
     
        for img_id in range(len(flatten_cls_preds)):
            cls_scores = flatten_cls_preds[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]

            result_list.append(self._bboxes_nms(cls_scores, bboxes, score_factor, conf_thr=0.65))

        #change tuple to list        
        result_list = [list(elem) for elem in result_list]
        
        return result_list
 
    def forward_train(self,
                cls_scores,
                bbox_preds,
                objectnesses,
                gt_bboxes,
                gt_labels,
        ):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        flatten_cls_preds, flatten_objectness, flatten_bboxes, flatten_bbox_preds = self._get_flatten_output(
                                                                                            cls_scores=cls_scores,
                                                                                            bbox_preds=bbox_preds,
                                                                                            objectnesses=objectnesses
                                                                                            )
        num_imgs = bbox_preds[0].shape[0]
        (pos_masks, cls_targets, obj_targets, bbox_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             self.flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)
       
        num_total_samples = max(sum(num_fg_imgs), 1)
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
   
        bbox_pred = flatten_bboxes.view(-1, 4)[pos_masks]
        obj_pred = flatten_objectness.view(-1, 1)
        cls_pred = flatten_cls_preds.view(-1, self.num_classes)[pos_masks]

        return num_total_samples, bbox_pred, bbox_targets,\
                 obj_pred, obj_targets, cls_pred, cls_targets


    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)

        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, 0)

        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        #Sampler
        #pos_inds - assigner positive indexes
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        num_pos_per_img = pos_inds.size(0)
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        pos_gt_labels = assign_result.labels[pos_inds]
        pos_ious = assign_result.max_overlaps[pos_inds]
        pos_ious = pos_ious.unsqueeze(-1)

        # IOU aware classification score 
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * pos_ious
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = pos_gt_bboxes

        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target, num_pos_per_img)
 