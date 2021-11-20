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
class FbetaMeter(Metric):

    def __init__(self, beta, num_classes=None, target_class=None, binary_mod=False,
                 weighted=False, ignore_index=-100, reduce=True, name=None, target_fields=None):
        if num_classes is None and target_class is None and not binary_mod:
            raise TypeError('You must specify either `num_classes` or `target_class` or `binary_mod`')
        if target_class is not None and binary_mod:
            raise ValueError('`target_class` is not compatible with `binary_mod`')
        if (target_class is not None or binary_mod) and weighted:
            raise ValueError('`weighted` is not compatible with `binary_mod` and `target_class`')
        if name is None:
            if target_class is None:
                name = f'F_beta={beta}'
            else:
                name = f'F_beta={beta}_class={target_class}'

        super().__init__(name=name, target_fields=target_fields)

        self.num_classes = num_classes
        self.target_class = target_class
        self.binary_mod = binary_mod
        self.weighted = weighted
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.beta_sq = beta ** 2
        if self.binary_mod or self.target_class is not None:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0
        else:
            self._classes_idx = np.arange(self.num_classes)[:, None]
            self.true_pos = np.zeros(self.num_classes)
            self.false_pos = np.zeros(self.num_classes)
            self.false_neg = np.zeros(self.num_classes)

    def reset(self):
        if self.binary_mod or self.target_class is not None:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0
        else:
            self.true_pos = np.zeros(self.num_classes)
            self.false_pos = np.zeros(self.num_classes)
            self.false_neg = np.zeros(self.num_classes)

    def _unify_shapes(self, target, prediction):
        if self.binary_mod:
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

            if self.target_class is not None:
                target = target == self.target_class
                prediction = prediction == self.target_class

        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        prediction = prediction[target != self.ignore_index]
        target = target[target != self.ignore_index]
        return prediction, target

    def _get_f1(self, tp, fn, fp):
        tp_rate = (1 + self.beta_sq) * tp
        denum = tp_rate + self.beta_sq * fn + fp
        np.seterr(divide='ignore')
        f1_scores = np.where(denum != 0.0, tp_rate / denum, 0)

        if self.reduce:
            if self.weighted:
                weights = (tp + fn) / (tp + fn).sum()
                f1_scores = weights @ f1_scores
            else:
                f1_scores = np.mean(f1_scores)
        return f1_scores

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        target, prediction = self._unify_shapes(target, prediction)

        if self.binary_mod or self.target_class is not None:
            pred_n = prediction
            true_n = target
        else:
            true_n: np.ndarray = target == self._classes_idx
            pred_n: np.ndarray = prediction == self._classes_idx
        tp = (pred_n & true_n).sum(-1)
        fp = (pred_n & ~true_n).sum(-1)
        fn = (~pred_n & true_n).sum(-1)

        return self._get_f1(tp, fn, fp)

    def update(self, target, prediction, *args, **kwargs):
        target, prediction = self._unify_shapes(target, prediction)

        if self.binary_mod or self.target_class is not None:
            pred_n = prediction
            true_n = target
        else:
            true_n: np.ndarray = target == self._classes_idx
            pred_n: np.ndarray = prediction == self._classes_idx
        self.true_pos += (pred_n & true_n).sum(-1)
        self.false_pos += (pred_n & ~true_n).sum(-1)
        self.false_neg += (~pred_n & true_n).sum(-1)

    def on_epoch_end(self, do_reset=True):
        tp = self.true_pos
        fp = self.false_pos
        fn = self.false_neg

        if do_reset:
            self.reset()
        return self._get_f1(tp, fn, fp)


@METRICS.register_class
class F1Meter(FbetaMeter):

    def __init__(self, num_classes=None, target_class=None, binary_mod=False, weighted=False,
                 ignore_index=-100, reduce=True, name=None, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'F1'
            else:
                name = f'F1_class={target_class}'

        super().__init__(beta=1, num_classes=num_classes, target_class=target_class,
                         binary_mod=binary_mod, weighted=weighted, ignore_index=ignore_index,
                         reduce=reduce, name=name, target_fields=target_fields)


@METRICS.register_class
class MultiLabelFbetaMeter(FbetaMeter):
    def __init__(self, beta, threshold=0.5, num_classes=None, target_class=None,
                 weighted=False, reduce=True, name=None, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'MultiLabel_F_beta={beta}'
            else:
                name = f'F_beta={beta}_class={target_class}'

        self.eps = 1e-9
        self.threshold = float(-np.log(1. / (threshold + self.eps) - 1.))  # reversed function of sigmoid

        super().__init__(beta=beta, num_classes=num_classes, target_class=target_class,
                         weighted=weighted, reduce=reduce, name=name, target_fields=target_fields)

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        prediction = prediction >= self.threshold
        target = target.astype(bool)
        prediction = prediction.astype(bool)

        if self.target_class is not None:
            prediction = prediction[:, self.target_class]
            target = target[:, self.target_class]
        tp = (prediction & target).sum(0)
        fp = (prediction & ~target).sum(0)
        fn = (target & ~prediction).sum(0)
        return self._get_f1(tp, fn, fp)

    def update(self, target, prediction, *args, **kwargs):
        prediction = prediction >= self.threshold
        target = target.astype(bool)
        prediction = prediction.astype(bool)

        if self.target_class is not None:
            prediction = prediction[:, self.target_class]
            target = target[:, self.target_class]
        self.true_pos += (prediction & target).sum(0)
        self.false_pos += (prediction & ~target).sum(0)
        self.false_neg += (target & ~prediction).sum(0)


@METRICS.register_class
class MultiLabelF1Meter(MultiLabelFbetaMeter):

    def __init__(self, threshold=0.5, num_classes=None, target_class=None, weighted=False,
                 reduce=True, name=None, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'MultiLabel_F1'
            else:
                name = f'F1_class={target_class}'

        super().__init__(beta=1, threshold=threshold, num_classes=num_classes, target_class=target_class,
                         weighted=weighted, reduce=reduce, name=name, target_fields=target_fields)


@METRICS.register_class
class MultilabelRecall(Metric):

    def __init__(self, name=None, target_fields=None, threshold=0.5):
        """
        param name: name of metric, if not stated threshold used as part of name
        param target_fields: fields that contain models predictions and targets
        param threshold: confidence threshold in terms of probabilities
        """
        if name is None:
            name = f'MultilabelRecall_{threshold}_macro'
        self.eps = 1e-9
        self.threshold = float(-np.log(1. / (threshold + self.eps) - 1.))   # reversed function of sigmoid
        super().__init__(name=name, target_fields=target_fields)

    def calculate(self, target, prediction):
        """
        param target: numpy array of shape (batch_size, n_classes), contains multihot vectors
        param prediction: numpy array of shape (batch_size, n_classes), contains class logits predicted by model
        """
        # replace probabilities that are bigger than threshold by 1 and that are smaller than threshold by 0
        prediction = prediction >= self.threshold
        k = (prediction & (target == 1)).sum(axis=1)
        n = target.sum(axis=1)
        result_for_each_image = k / (n + self.eps)
        # get mean of batch
        return result_for_each_image.mean()


@METRICS.register_class
class MultilabelNoise(Metric):

    def __init__(self, name=None, target_fields=None, threshold=0.5):
        """
        param name: name of metric, if not stated threshold used as part of name
        param target_fields: fields that contain models predictions and targets
        param threshold: confidence threshold in terms of probabilities
        """
        if name is None:
            name = f'MultilabelNoise_{threshold}_macro'
        self.eps = 1e-9
        self.threshold = float(-np.log(1. / (threshold + self.eps) - 1.))   # reversed function of sigmoid
        super().__init__(name=name, target_fields=target_fields)

    def calculate(self, target, prediction):
        """
        param target: numpy array of shape (batch_size, n_classes), contains multihot vectors
        param prediction: numpy array of shape (batch_size, n_classes), contains class logits predicted by model
        """
        # replace probabilities that are bigger than threshold by 1 and that are smaller than threshold by 0
        prediction = prediction >= self.threshold
        k = (prediction & (target == 1)).sum(axis=1)
        s = prediction.sum(axis=1)
        result_for_each_image = (s - k) / (s + self.eps)
        # get mean of batch
        return result_for_each_image.mean()


@METRICS.register_class
class MultilabelPrecision(Metric):

    def __init__(self, name=None, target_fields=None, threshold=0.5):
        """
        param name: name of metric, if not stated threshold used as part of name
        param target_fields: fields that contain models predictions and targets
        param threshold: confidence threshold in terms of probabilities
        """
        if name is None:
            name = f'MultilabelPrecision_{threshold}_macro'
        self.eps = 1e-9
        self.threshold = float(-np.log(1. / (threshold + self.eps) - 1.))   # reversed function of sigmoid
        super().__init__(name=name, target_fields=target_fields)

    def calculate(self, target, prediction):
        """
        param target: numpy array of shape (batch_size, n_classes), contains multihot vectors
        param prediction: numpy array of shape (batch_size, n_classes), contains class logits predicted by model
        """
        # replace probabilities that are bigger than threshold by 1 and that are smaller than threshold by 0
        prediction = prediction >= self.threshold
        k = (prediction & (target == 1)).sum(axis=1)
        s = prediction.sum(axis=1)
        result_for_each_image = k / (s + self.eps)
        # get mean of batch
        return result_for_each_image.mean()

@METRICS.register_class
class MSE(Metric):

    def calculate(self, target, prediction):
        """
        param target: numpy array of shape (batch_size, n_classes), contains multihot vectors
        param prediction: numpy array of shape (batch_size, n_classes), contains class logits predicted by model
        """
        return np.mean((np.array(target) - np.array(prediction))**2)