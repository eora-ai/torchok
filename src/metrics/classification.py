from typing import Tuple

import numpy as np
from numpy import ndarray

from src.registry import METRICS
from .common import Metric


@METRICS.register_class
class AccuracyMeter(Metric):

    def __init__(self, binary_mod=False, name=None, ignore_index=-100, target_fields=None):
        super().__init__('accuracy' if name is None else name, target_fields=target_fields)
        self.binary_mod = binary_mod
        self.ignore_index = ignore_index

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        if self.binary_mod:
            tdim = target.ndim
            pdim = prediction.ndim
            if tdim == pdim:
                prediction = prediction > 0
            else:
                raise ValueError(f'Dimension sizes for target and prediction do not match {tdim} != {pdim}')
        else:
            if prediction.ndim == target.ndim + 1:
                prediction = prediction.argmax(1)
        prediction = prediction[target != self.ignore_index]
        target = target[target != self.ignore_index]
        return (target == prediction).mean()


@METRICS.register_class
class TPFPFNMeter(Metric):

    def __init__(self, num_classes=None, target_class=None, average='binary',
                 ignore_index=-100, name=None, target_fields=None):
        if average not in ['binary', 'macro', 'micro', 'weighted', 'none']:
            raise ValueError('Supported averaging modes are: "binary", "macro", "micro", "weighted", "none". '
                             'Got: {average}')

        if num_classes is None and average != 'binary':
            raise TypeError('You must specify "num_classes" for non-binary averaging')
        if num_classes is not None and target_class is None and average == 'binary':
            raise TypeError('You must leave "num_classes" not specified or specify "target_class" '
                            'for average mode "binary"')
        if target_class is not None and average != 'binary':
            raise ValueError('"target_class" is compatible with averaging mode "binary" only')
        if target_class is not None and (num_classes is None or target_class >= num_classes):
            raise ValueError('When "target_class" is specified, "num_classes" must also be given, '
                             'so that "target_class" < "num_classes". '
                             f'Got "target_class": {target_class}, "num_classes": {num_classes}')

        if name is None:
            if average == 'binary':
                if target_class is None:
                    name = f'TPFPFN'
                else:
                    name = f'TPFPFN[{target_class}]'
            else:
                name = f'TPFPFN_{average}'

        super().__init__(name=name, target_fields=target_fields)

        self.average = average
        self.num_classes = num_classes
        self.target_class = target_class
        if num_classes is not None and self.target_class is None:
            self.valid_mask = np.arange(self.num_classes) != ignore_index
        else:
            self.valid_mask = None
        self.true_pos = None
        self.false_pos = None
        self.false_neg = None
        self._classes_idx = np.arange(self.num_classes)[:, None] if num_classes is not None else None
        self.reset()

    def reset(self):
        if self.average == 'binary' or self.target_class is not None:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0
        else:
            self.true_pos = np.zeros(self.num_classes)
            self.false_pos = np.zeros(self.num_classes)
            self.false_neg = np.zeros(self.num_classes)

    def _unify_shapes(self, target, prediction):
        if prediction.shape[0] != target.shape[0]:
            raise ValueError('Batch size of target and prediction do not match',
                             target.shape[0], prediction.shape[0])

        if self.average == 'binary' and self.target_class is None:
            if prediction.shape != target.shape:
                raise ValueError('shapes of target and prediction do not match',
                                 target.shape, prediction.shape)
            # prediction and target will have shapes: (N,)
            prediction = prediction > 0     # for logits this is 0.5 probability after applying sigmoid
        elif self.average == 'binary' and self.target_class is not None:
            # Dimensions check
            if prediction.ndim != 2 or target.ndim != 1:
                raise ValueError('prediction and target must be 2d and 1d matrices respectively')

            # if target class is given prediction and target will have shapes: (N,), dtype=np.bool
            prediction = prediction.argmax(1) == self.target_class
            target = target == self.target_class
        else:
            # Dimensions check
            if prediction.ndim != 2 or target.ndim != 1:
                raise ValueError('prediction and target must be 2d and 1d matrices respectively')

            # if target class isn't given prediction and target will have shapes: (N,), dtype=np.int
            prediction = prediction.argmax(1)

        return target, prediction

    def _get_metric(self, tp, fp, fn):
        raise NotImplementedError()

    def _calc_with_reduce(self, tp: ndarray, fp: ndarray, fn: ndarray) -> np.ndarray:
        if self.valid_mask is not None:
            tp, fp, fn = tp[self.valid_mask], fp[self.valid_mask], fn[self.valid_mask]

        if self.average == 'macro':
            metrics = self._get_metric(tp, fp, fn)
            metrics = np.mean(metrics)
        elif self.average == 'micro':
            tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
            metrics = self._get_metric(tp, fp, fn)
        elif self.average == 'weighted':
            metrics = self._get_metric(tp, fp, fn)
            weights = tp + fn
            metrics = np.average(metrics, weights=weights)
        else:
            metrics = self._get_metric(tp, fp, fn)

        return metrics

    def _calc_tp_fp_fn(self, target: ndarray, prediction: ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target, prediction = self._unify_shapes(target, prediction)

        if self.average == 'binary' or self.target_class is not None:
            # true_n and pred_n will have shapes (N), dtype=np.bool
            pred_n = prediction
            true_n = target
        else:
            # true_n and pred_n will have shapes (C, N), dtype=np.bool
            true_n: np.ndarray = target == self._classes_idx
            pred_n: np.ndarray = prediction == self._classes_idx

        tp = (pred_n & true_n).sum(-1)
        fp = (pred_n & ~true_n).sum(-1)
        fn = (~pred_n & true_n).sum(-1)

        return tp, fp, fn

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        tp, fp, fn = self._calc_tp_fp_fn(target, prediction)
        value = self._calc_with_reduce(tp, fp, fn)

        return value

    def update(self, target, prediction, *args, **kwargs):
        tp, fp, fn = self._calc_tp_fp_fn(target, prediction)

        self.true_pos += tp
        self.false_pos += fp
        self.false_neg += fn

    def on_epoch_end(self, do_reset=True):
        value = self._calc_with_reduce(self.true_pos, self.false_pos, self.false_neg)

        if do_reset:
            self.reset()

        return value


@METRICS.register_class
class FbetaMeter(TPFPFNMeter):

    def __init__(self, beta, num_classes=None, target_class=None, average='binary',
                 ignore_index=-100, name=None, target_fields=None):
        if name is None:
            if average == 'binary':
                if target_class is None:
                    name = f'F{beta}'
                else:
                    name = f'F{beta}[{target_class}]'
            else:
                name = f'F{beta}_{average}'

        super().__init__(name=name, num_classes=num_classes, target_class=target_class, average=average,
                         ignore_index=ignore_index, target_fields=target_fields)
        self.beta2 = beta ** 2
        self.reset()

    def _get_metric(self, tp, fp, fn):
        tp_rate = (1 + self.beta2) * tp
        denum = tp_rate + self.beta2 * fn + fp + self.eps
        f1_scores = tp_rate / denum

        return f1_scores


@METRICS.register_class
class F1Meter(FbetaMeter):

    def __init__(self, num_classes=None, target_class=None, average='binary',
                 ignore_index=-100, name=None, target_fields=None):
        if name is None:
            if average == 'binary':
                if target_class is None:
                    name = f'F1'
                else:
                    name = f'F1[{target_class}]'
            else:
                name = f'F1_{average}'

        super().__init__(beta=1, num_classes=num_classes, target_class=target_class,
                         average=average, ignore_index=ignore_index, name=name, target_fields=target_fields)


@METRICS.register_class
class PrecisionMeter(TPFPFNMeter):

    def __init__(self, num_classes=None, target_class=None, average='binary',
                 ignore_index=-100, name=None, target_fields=None):
        if name is None:
            if average == 'binary':
                if target_class is None:
                    name = f'Precision'
                else:
                    name = f'Precision[{target_class}]'
            else:
                name = f'Precision_{average}'

        super().__init__(name=name, num_classes=num_classes, target_class=target_class, average=average,
                         ignore_index=ignore_index, target_fields=target_fields)

    def _get_metric(self, tp, fp, fn):
        precisions = tp / (tp + fp + self.eps)

        return precisions


@METRICS.register_class
class RecallMeter(TPFPFNMeter):

    def __init__(self, num_classes=None, target_class=None, average='binary',
                 ignore_index=-100, name=None, target_fields=None):
        if name is None:
            if average == 'binary':
                if target_class is None:
                    name = f'Recall'
                else:
                    name = f'Recall[{target_class}]'
            else:
                name = f'Recall_{average}'

        super().__init__(name=name, num_classes=num_classes, target_class=target_class, average=average,
                         ignore_index=ignore_index, target_fields=target_fields)

    def _get_metric(self, tp, fp, fn):
        recalls = tp / (tp + fn + self.eps)

        return recalls


@METRICS.register_class
class MultiLabelFbetaMeter(FbetaMeter):
    def __init__(self, beta, threshold=0.5, num_classes=None, target_class=None, average='macro',
                 ignore_index=-100, name=None, target_fields=None):
        if target_class is not None and average != 'binary':
            raise ValueError('averaging mode "binary" is available only when "target_class" is specified')

        if name is None:
            if average == 'binary':
                name = f'MultiLabel_F_beta@{beta}[{target_class}]'
            else:
                name = f'MultiLabel_F_beta@{beta}_{average}'

        super().__init__(beta=beta, num_classes=num_classes, target_class=target_class, average=average,
                         ignore_index=ignore_index, name=name, target_fields=target_fields)

        self.threshold = float(-np.log(1. / (threshold + self.eps) - 1.))  # reversed function of sigmoid

    def _calc_tp_fp_fn(self, target: ndarray, prediction: ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = prediction >= self.threshold
        target = target.astype(bool)

        if self.target_class is not None:
            prediction = prediction[:, self.target_class]
            target = target[:, self.target_class]

        tp = (prediction & target).sum(0)
        fp = (prediction & ~target).sum(0)
        fn = (target & ~prediction).sum(0)

        return tp, fp, fn


@METRICS.register_class
class MultiLabelF1Meter(MultiLabelFbetaMeter):
    def __init__(self, threshold=0.5, num_classes=None, target_class=None, average='macro',
                 ignore_index=-100, name=None, target_fields=None):
        if name is None:
            if average == 'binary':
                name = f'MultiLabel_F1[{target_class}]'
            else:
                name = f'MultiLabel_F1_{average}'

        super().__init__(beta=1, threshold=threshold, num_classes=num_classes, target_class=target_class,
                         average=average, ignore_index=ignore_index, name=name, target_fields=target_fields)


@METRICS.register_class
class MultiLabelRecallMeter(MultiLabelF1Meter):
    def __init__(self, threshold=0.5, num_classes=None, target_class=None, average='macro',
                 ignore_index=-100, name=None, target_fields=None):
        """
        param name: name of metric, if not stated threshold used as part of name
        param target_fields: fields that contain models predictions and targets
        param threshold: confidence threshold in terms of probabilities
        """
        if name is None:
            if average == 'binary':
                name = f'MultiLabelRecall@{threshold}[{target_class}]'
            else:
                name = f'MultiLabelRecall@{threshold}_{average}'

        super().__init__(threshold=threshold, num_classes=num_classes, target_class=target_class,
                         average=average, ignore_index=ignore_index, name=name, target_fields=target_fields)

    def _get_metric(self, tp: ndarray, fp: ndarray, fn: ndarray) -> np.ndarray:
        recall = tp / (tp + fn + self.eps)
        print(recall)

        return recall


@METRICS.register_class
class MultiLabelPrecisionMeter(MultiLabelF1Meter):

    def __init__(self, threshold=0.5, num_classes=None, target_class=None, average='macro',
                 ignore_index=-100, name=None, target_fields=None):
        """
        param name: name of metric, if not stated threshold used as part of name
        param target_fields: fields that contain models predictions and targets
        param threshold: confidence threshold in terms of probabilities
        """
        if name is None:
            if average == 'binary':
                name = f'MultiLabelPrecision@{threshold}[{target_class}]'
            else:
                name = f'MultiLabelPrecision@{threshold}_{average}'

        super().__init__(threshold=threshold, num_classes=num_classes, target_class=target_class,
                         average=average, ignore_index=ignore_index, name=name, target_fields=target_fields)

    def _get_metric(self, tp: ndarray, fp: ndarray, fn: ndarray) -> np.ndarray:
        precision = tp / (tp + fp + self.eps)

        return precision
