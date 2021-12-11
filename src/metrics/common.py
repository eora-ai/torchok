import numpy as np
import torch


def far(conf):
    """
    False acceptance rate. Frequency that the system makes False Accepts per amount of impostor attempts.
    :param conf: confusion matrix.
    :return: False accept rate.
    """
    fa = conf[0, 1]  # false acceptance
    n = np.sum(conf[:, 1])  # negative attempts (false acceptance + true rejected)
    return fa / n


def frr(conf):
    """
    False rejection rate. Frequency that the system makes False Reject per amount of legal attempts.
    :param conf: confusion matrix.
    :return: False reject rate.
    """
    fr = conf[1, 0]  # false rejected
    p = np.sum(conf[:, 0])  # positive attempts (true accepted + false rejected)
    return fr / p


class Metric:
    """Template class for model's metrics"""

    def __init__(self, name: str, target_fields: dict):
        """Initialize metric"""
        self.name = name
        self.target_fields = target_fields
        self.mean = np.zeros(1)
        self.n = 0
        self.eps = 1e-9
        self.use_gpu = False
        self.use_torch = False

    def calculate(self, target, prediction):
        """Returns the instant value of a metric given prediction and target"""
        raise NotImplementedError()

    def update(self, target, prediction, *args, **kwargs):
        """Updates metric buffer"""
        batch_size = prediction.shape[0]
        value = self.calculate(target, prediction) * batch_size
        self.mean = (self.n * self.mean + value) / (self.n + batch_size)
        self.n += batch_size

    def reset(self):
        """Resets the state of metric"""
        self.mean = np.zeros(1)
        self.n = 0

    def on_epoch_end(self, do_reset=True):
        """Returns summarized value of metric, clears buffer"""
        output = self.mean
        if do_reset:
            self.reset()
        return output


class AverageMeter(Metric):
    def calculate(self, value, *args, **kwargs):
        """Returns the instant value of a metric given prediction and target"""
        raise NotImplementedError()

    def update(self, value, *args, **kwargs):
        """Updates metric buffer"""
        self.mean = (self.n * self.mean + value) / (self.n + 1)
        self.n += 1


class ConfusionMatrix:
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        self._conf = None
        self.normalized = normalized
        self.num_classes = num_classes

    def reset(self):
        self._conf = None

    def calculate(self, target, prediction):
        if prediction.shape != target.shape:
            raise ValueError('number of targets and predicted outputs do not match',
                             prediction.shape, target.shape)
        max_v = prediction.max()
        if max_v >= self.num_classes:
            raise ValueError(f'predicted values are not between 0 and k-1 ({self.num_classes - 1}), '
                             f'got max={max_v}')
        max_v = target.max()
        if max_v >= self.num_classes:
            raise ValueError(f'target values are not between 0 and k-1 ({self.num_classes - 1}), '
                             f'got max={max_v}')

        prediction = prediction.reshape(-1)
        target = target.reshape(-1)

        # hack for bincounting 2 arrays together
        x = prediction + self.num_classes * target
        bincount_2d = torch.bincount(x, minlength=self.num_classes ** 2)
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        return conf.cpu().numpy()

    def update(self, target, prediction, *args, **kwargs):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        conf = self.calculate(target, prediction)
        if self._conf is None:
            self._conf = conf
        else:
            self._conf += conf

    @property
    def conf(self) -> np.ndarray:
        if self._conf is None:
            shape = (self.num_classes, self.num_classes)
            return torch.zeros(shape, dtype='int32')
        else:
            return self._conf

    def value(self):
        """
        Returns:
            Confusion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        conf = self.conf
        if self.normalized:
            conf / conf.sum(1, keepdims=True).clip(min=1e-12)
        return conf
