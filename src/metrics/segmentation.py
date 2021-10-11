from typing import Union

import numpy as np
from numpy import ndarray

from src.registry import METRICS
from .common import Metric, ConfusionMatrix


@METRICS.register_class
class MeanIntersectionOverUnionMeter(Metric):
    def __init__(self, num_classes: int = None, target_classes: Union[list, int] = None, binary_mode: bool = False,
                 ignore_classes: Union[list, int] = None, weighted: bool = False, reduce: int = True, name: str = None,
                 target_fields: dict = None, use_gpu: bool = True):
        """Calculates mean intersection over union for a multi-class semantic
        segmentation problem. The meter makes calculations based on confusion matrix

        Keyword arguments:
        :param num_classes (int): number of classes
        :param target_classes (int, list): list of class indexes or class index, which are used for metric calculation.
            Values of a list should be within range [0, num_classes -1]. If set to None or empty list,
            metric is calculated for all classes. If binary_mod is set to True, metric is calculated for both classes.
        :param ignore_classes (int, list): Specifies a class or list of classes that will be ignored and
            not contribute to the total score. Suppress values in `target_classes`.
        :param binary_mode (bool): If True consider input as a [N, H, W] tensor and set target_classes to 1 otherwise
            consider input as a [N, C, H, W] tensor.
        """
        if binary_mode:
            if num_classes is not None:
                raise ValueError("Number of classes cannot be specified when `binary_mod` to True")
            num_classes = 2
            target_classes = [1]  # in case of binary mode calculate only for the class 1 by default
        elif num_classes is None:
            raise ValueError("You must specify number of classes or set `binary_mod` to True")

        if num_classes < 2:
            raise ValueError("Number of classes must be >= 2")
        elif not binary_mode:
            if not target_classes:
                target_classes = list(range(num_classes))
            elif isinstance(target_classes, int):
                target_classes = [target_classes]

            if max(target_classes) >= num_classes or min(target_classes) < 0:
                bad_vals = [idx for idx in target_classes if idx >= num_classes or idx < 0]
                raise ValueError(f"Values in `target_values` are out of range "
                                 f"[0, {num_classes - 1}], got {bad_vals}")

            if ignore_classes is not None:
                if isinstance(ignore_classes, int):
                    ignore_classes = [ignore_classes]
                for ignore in ignore_classes:
                    if ignore in target_classes:
                        target_classes.remove(ignore)

        if name is None:
            name = f'mIoU_weighted' if weighted else f'mIoU'
        super().__init__(name, target_fields=target_fields)
        self._conf_matrix = ConfusionMatrix(num_classes, False)
        self._num_classes = num_classes
        self._weighted = weighted
        self._reduce = reduce
        self._binary_mod = binary_mode
        self._target_classes = target_classes
        self.use_gpu = use_gpu
        self.use_torch = True

    def reset(self):
        super().reset()
        self._conf_matrix.reset()

    def _unify_shapes(self, target, prediction):
        if self._binary_mod:
            if prediction.shape != target.shape:
                raise ValueError('shapes of target and prediction do not match',
                                 target.shape, prediction.shape)
            prediction = prediction > 0
        else:
            # Dimensions check
            if prediction.shape[0] != target.shape[0]:
                raise ValueError('Batch size of target and prediction do not match',
                                 target.shape[0], prediction.shape[0])
            if prediction.ndim == target.ndim + 1:
                prediction = prediction.argmax(1)

            # Dimensions check
            if prediction.shape[1:] != target.shape[1:]:
                raise ValueError('Spatial shapes of target and prediction do not match',
                                 target.shape[1:], prediction.shape[1:])
        return target, prediction

    def _calculate_score(self, conf_matrix):
        tp = np.diagonal(conf_matrix)
        pos_pred = conf_matrix.sum(axis=0)
        pos_gt = conf_matrix.sum(axis=1)

        # Check which classes have elements
        valid_idxs = pos_gt > 0
        ious_valid = valid_idxs & (pos_gt + pos_pred - tp > 0)

        # Calculate intersections over union for each class
        ious = np.zeros((self._num_classes,))
        union = pos_gt[ious_valid] + pos_pred[ious_valid] - tp[ious_valid]
        ious[ious_valid] = tp[ious_valid] / union

        # Calculate mean intersection over union
        iou = self._averaging(ious, ious_valid, pos_gt, conf_matrix)

        return iou

    def _averaging(self, scores, valid_classes, pos_gt, conf_matrix):
        multihot = np.zeros((self._num_classes,), dtype=bool)
        multihot[self._target_classes] = True
        valid_classes = valid_classes & multihot

        if not self._weighted:
            score = np.mean(scores[valid_classes])
        else:
            weights = np.divide(pos_gt, conf_matrix.sum())
            score = scores[valid_classes] @ weights[valid_classes]

        return score

    def _calculate(self, target: ndarray, prediction: ndarray) -> Union[list, ndarray]:
        ious = []
        for true_mask, pred_mask in zip(target, prediction):
            pred_mask = pred_mask.reshape(-1)
            true_mask = true_mask.reshape(-1)

            conf_matrix = self._conf_matrix.calculate(true_mask, pred_mask)
            ious.append(self._calculate_score(conf_matrix))

        # ignore Nan elements (samples where no target classes were in GT)
        ious = np.array(ious)
        ious = ious[~np.isnan(ious)]

        return ious

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        """Calculate IoU metric based on the predicted and target pair.
        Keyword arguments:
        :param prediction: if `binary_mod` is False it can be a (N, *D) tensor of integer values
            between 0 and K-1 or (N, C, *D) tensor of floats values;
            if `binary_mod` is True ir can be a (N, *D) tensor of floats values.
        :param target: Can be a (N, *D) tensor of
            integer values between 0 and K-1.
        """
        target, prediction = self._unify_shapes(target, prediction)
        scores = self._calculate(target, prediction)

        if self._reduce:
            return np.mean(scores)
        else:
            return scores

    def update(self, target: ndarray, prediction: ndarray, *args, **kwargs):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        :param prediction: if `binary_mod` is False it can be a (N, *D) tensor of integer values
            between 0 and K-1 or (N, C, *D) tensor of floats values;
            if `binary_mod` is True ir can be a (N, *D) tensor of floats values.
        :param target: Can be a (N, *D) tensor of
            integer values between 0 and K-1.
        """
        target, prediction = self._unify_shapes(target, prediction)

        scores = self._calculate(target, prediction)

        batch_size = scores.shape[0]
        value = scores.mean() * batch_size if batch_size else 0  # handle case when there's no valid score

        self.mean = (self.n * self.mean + value) / (self.n + batch_size) if self.n + batch_size else 0
        self.n += batch_size


@METRICS.register_class
class MeanDiceMeter(MeanIntersectionOverUnionMeter):
    def __init__(self, num_classes: int = None, target_classes: Union[list, int] = None, binary_mode: bool = False,
                 ignore_classes: Union[list, int] = None, weighted: bool = False, reduce: int = True, name: str = None,
                 target_fields: dict = None, use_gpu: bool = True):
        """Calculates mean dice similarity coefficient for a multi-class semantic
        segmentation problem. The meter makes calculations based on confusion matrix

        Keyword arguments:
        :param num_classes (int): number of classes
        :param target_classes (int, list): list of class indexes or class index, which are used for metric calculation.
            Values of a list should be within range [0, num_classes -1]. If set to None or empty list,
            metric is calculated for all classes. If binary_mod is set to True, metric is calculated for both classes.
        :param ignore_classes (int, list): Specifies a class or list of classes that will be ignored and
            not contribute to the total score. Suppress values in `target_classes`.
        :param binary_mode (bool): If True consider input as a [N, H, W] tensor and set target_classes to 1 otherwise
            consider input as a [N, C, H, W] tensor.
        """
        if name is None:
            name = f'mDice_weighted' if weighted else f'mDice'

        super().__init__(num_classes=num_classes, target_classes=target_classes,
                         ignore_classes=ignore_classes, binary_mode=binary_mode, weighted=weighted,
                         reduce=reduce, name=name, target_fields=target_fields, use_gpu=use_gpu)

    def _calculate_score(self, conf_matrix):
        tp = np.diagonal(conf_matrix)
        pos_pred = conf_matrix.sum(axis=0)
        pos_gt = conf_matrix.sum(axis=1)

        # Check which classes have elements
        valid_classes = (pos_gt > 0) & (pos_gt + pos_pred - tp > 0)

        # Calculate intersections over union for each class
        dice_scores = np.zeros((self._num_classes,))
        dice_scores[valid_classes] = 2 * tp[valid_classes] / (pos_gt[valid_classes] + pos_pred[valid_classes])

        # Calculate mean intersection over union
        mean_dice = self._averaging(dice_scores, valid_classes, pos_gt, conf_matrix)

        return mean_dice
