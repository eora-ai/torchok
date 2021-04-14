import numpy as np
import torch


def true_positive(conf, n):
    """
    True positive values of the confusion matrix (i.e. n-th diagonal element of the confusion matrix)
    :param conf: confusion matrix
    :param n: number of class
    :return: true positive value for n-th class
    """
    return conf[n, n]


def false_positive(conf, n):
    """
    False positive values of the confusion matrix (i.e. n-th row of the confusion matrix without diagonal element)
    :param conf: confusion matrix
    :param n: number of class
    :return: false positive value for n-th class
    """
    return np.sum([np.sum(conf[:n, n]), np.sum(conf[n + 1:, n])])


def true_negative(conf, n):
    """
    True negative values of the confusion matrix (i.e. all elements that do not belong to n-th row and n-th column)
    :param conf: confusion matrix
    :param n: number of class
    :return: True negative value for n-th class
    """
    return np.sum([np.sum(conf), - np.sum(conf[:, n]), - np.sum(conf[n, :]), conf[n, n]])


def false_negative(conf, n):
    """
    False negative values of the confusion matrix (i.e. n-th column of the confusion matrix without diagonal element)
    :param conf: confusion matrix
    :param n: number of class
    :return: false negative value for n-th class
    """
    return np.sum([np.sum(conf[n, :n]), np.sum(conf[n, n + 1:])])


def accuracy(conf):
    """
    Calculates accuracy for the given confusion matrix
    :param conf: confusion matrix
    :return: accuracy (float)
    """
    tp = np.sum(np.diag(conf))
    total = conf.sum()

    return tp / total


def precision_binary(conf, n=1):
    """
    Precision score for a given class of confusion matrix
    :param conf: confusion matrix
    :param n: number of class
    :return: precision score for a certain class
    """
    tp = true_positive(conf, n)
    fp = false_positive(conf, n)

    if tp + fp == 0:
        return 0.
    return float(tp) / (tp + fp)


def precision_micro(conf):
    """
    Precision score with a micro average of a confusion matrix
    :param conf: confusion matrix
    :return: precision score with a micro average
    """
    total_tp = 0
    total_fp = 0
    for i in range(len(conf)):
        total_tp += true_positive(conf, i)
        total_fp += false_positive(conf, i)

    return float(total_tp) / (total_tp + total_fp)


def precision_macro_weighted(conf, average='macro'):
    """
    Precision score with a macro or weighted average of a confusion matrix
    :param conf: confusion matrix
    :param average: type of average, macro or weighted
    :return: precision score with a macro or weighted average
    """
    total_precision = 0

    for i in range(len(conf)):
        coef = 1
        if average == 'weighted':
            coef = np.sum(conf[i, :])
        total_precision += precision_binary(conf, i) * coef

    if average == 'weighted':
        total_precision /= np.sum(conf)
    else:
        total_precision /= len(conf)

    return total_precision


def precision(conf, average=None):
    """
    Precision score for a confusion matrix with binary, micro, macro, weighted average
    or a list of precision scores for each class
    :param conf: confusion matrix
    :param average: type of average, None (returns list of f-scores), binary, micro, macro, weighted
    :return: precision score with binary, micro, macro, weighted average
    or a list of precision scores for each class
    """
    if not average:
        return [precision_binary(conf, i) for i in range(len(conf))]

    if average == 'binary':
        if len(conf) > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")
        return precision_binary(conf)

    if average == 'micro':
        return precision_micro(conf)

    if average in ['macro', 'weighted']:
        return precision_macro_weighted(conf, average)


def precision_k(labels, scores, k=5, average=False, largest=False):
    """
    Computes the precision@k for the specified values of k
    or average precision@k if attribute average is True
    :param labels: pytorch.Tensor of shape (n,) containing labels of each value
    :param scores: pytorch.Tensor of shape (n,) containing values
    :param k: number of items that will be taken
    :param average: if True calculate ap@k metric else calculate precision@k metric
    :param largest: if True take k items with the largest value
                    else take k items with the smallest value
    :return: precision@k or ap@k metric
    """

    topk, indices = scores.topk(k, largest=largest)
    topk_target = labels[indices]

    if average:
        cummulative_sum = topk_target.cumsum(dim=0).float()
        ks = torch.arange(1, k + 1, dtype=torch.float32)
        return (cummulative_sum / ks).mean().item()
    else:
        return topk_target.float().mean().item()


def map_k(labels, scores, k=5, largest=False):
    """
    Computes the mean average precision@k for the specified values of k
    :param labels: list of pytorch.Tensor of shape (n,) containing labels of each value
    :param scores: list of pytorch.Tensor of shape (n,) containing values of the same size as labels
    :param k: number of items that will be taken
    :param largest: if True take k items with the largest value
                    else take k items with the smallest value
    :return: map@k metric
    """

    return np.mean([precision_k(label, value, k, True, largest) for label, value in zip(labels, scores)])


def recall_binary(conf, n=1):
    """
    Recall score for a given class of confusion matrix
    :param conf: confusion matrix
    :param n: number of class
    :return: recall score for a certain class
    """
    tp = true_positive(conf, n)
    fn = false_negative(conf, n)

    if tp + fn == 0:
        return 0.
    return float(tp) / (tp + fn)


def recall_micro(conf):
    """
    Recall score with a micro average of a confusion matrix
    :param conf: confusion matrix
    :return: recall score with a micro average
    """
    total_tp = 0
    total_fn = 0

    for i in range(len(conf)):
        total_tp += true_positive(conf, i)
        total_fn += false_negative(conf, i)

    return float(total_tp) / (total_tp + total_fn)


def recall_macro_weighted(conf, average='macro'):
    """
    Recall score with a macro or weighted average of a confusion matrix
    :param conf: confusion matrix
    :param average: type of average, macro or weighted
    :return: recall score with a macro or weighted average
    """
    total_recall = 0

    for i in range(len(conf)):
        coef = 1
        if average == 'weighted':
            coef = np.sum(conf[i, :])
        total_recall += recall_binary(conf, i) * coef

    if average == 'weighted':
        total_recall /= np.sum(conf)
    else:
        total_recall /= len(conf)

    return total_recall


def recall(conf, average=None):
    """
    Recall score for a confusion matrix with binary, micro, macro, weighted average
    or a list of recall scores for each class
    :param conf: confusion matrix
    :param average: type of average, None (returns list of f-scores), binary, micro, macro, weighted
    :return: recall score with binary, micro, macro, weighted average
    or a list of recall scores for each class
    """
    if not average:
        return [recall_binary(conf, i) for i in range(len(conf))]

    if average == 'binary':
        if len(conf) > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")
        return recall_binary(conf)

    if average == 'micro':
        return recall_micro(conf)

    if average == 'macro':
        return recall_macro_weighted(conf, average)

    if average == 'weighted':
        return recall_macro_weighted(conf, average)


def fbeta_precision_recall(precision_val, recall_val, beta):
    """
    Fbeta score for a given precision and recall
    :param precision_val: precision score of a certain class
    :param recall_val: recall score of a certain class
    :param beta: beta coefficient
    :return: fbeta score
    """
    beta2 = beta ** 2

    if precision_val + recall_val == 0:
        return 0

    return (1 + beta2) * (precision_val * recall_val) / (beta2 * precision_val + recall_val)


def fbeta_binary_micro(conf, average='binary', beta=1, n=1):
    """
    Fbeta score for a certain class of confusion matrix with binary and micro average
    :param conf: confusion matrix
    :param average: type of average, binary or micro
    :param beta: beta coefficient
    :param n: number of class
    :return: fbeta score with binary and micro average
    """
    if average == 'binary':
        precision_value = precision_binary(conf, n)
        recall_value = recall_binary(conf, n)
    elif average == 'micro':
        precision_value = precision_micro(conf)
        recall_value = recall_micro(conf)
    else:
        raise ValueError('Unexpected averaging method')

    return fbeta_precision_recall(precision_value, recall_value, beta)


def fbeta_macro_weighted(conf, average='macro', beta=1):
    """
    Fbeta score for a confusion matrix with macro or weighted average
    :param conf: confusion matrix
    :param average: type of average, macro or weighted
    :param beta: beta coefficient
    :return: fbeta score with macro or weighted average
    """
    total_fbeta = 0

    for i in range(len(conf)):
        coef = 1

        if average == 'weighted':
            coef = np.sum(conf[i, :])

        total_fbeta += fbeta_binary_micro(conf, 'binary', beta, i) * coef

    if average == 'weighted':
        total_fbeta /= np.sum(conf)
    else:
        total_fbeta /= len(conf)

    return total_fbeta


def fbeta(conf, average=None, beta=1):
    """
    Fbeta score for a confusion matrix with binary, micro, macro, weighted average
    or a list of fbeta scores for each class
    :param conf: confusion matrix
    :param average: type of average, None (returns list of f-scores), binary, micro, macro, weighted
    :param beta: beta coefficient
    :return: fbeta score with binary, micro, macro, weighted average
    or a list of fbeta scores for each class
    """
    if not average:
        return [fbeta_binary_micro(conf, 'binary', beta, i) for i in range(len(conf))]

    if average == 'binary':
        if len(conf) > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")
        return fbeta_binary_micro(conf, average, beta)

    if average == 'micro':
        return fbeta_binary_micro(conf, average, beta)

    if average == 'macro':
        return fbeta_macro_weighted(conf, average, beta)

    if average == 'weighted':
        return fbeta_macro_weighted(conf, average, beta)


def f1(conf, average=None):
    """
    F1 score for a confusion matrix with binary, micro, macro, weighted average
    or a list of f1 scores for each class. Basically, a wrap up on fbeta funciton.
    :param conf: confusion matrix
    :param average: type of average, None (returns list of f1-scores), binary, micro, macro, weighted
    :return: f1 score with binary, micro, macro, weighted average
    or a list of f1 scores for each class
    """
    return fbeta(conf, average, 1)


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
