from itertools import combinations

import numpy as np
from sklearn.metrics import auc, roc_curve
import torch

from src.metrics import Metric, frr, far


class FVMeter(Metric):
    def __init__(self, dist='euclidean', override_name='fv_meter'):
        """
        abstract class of metrics for a given verification problem.
        :param dist: type of distance function. `euclidean` and `cosine` are available now.
        """
        super().__init__(name=dist + "_" + override_name)
        self.conf = np.zeros((2, 2), dtype=np.int32)
        self.distances = []
        self.labels = []
        if dist not in ['euclidean', 'cosine']:
            raise TypeError('distance function does not support or incorrect')
        self.dist_type = dist

    def reset(self):
        """
        Resets internal state of the metric to initial state
        :return: None
        """
        self.distances = []
        self.labels = []

    def dist(self, inp1, inp2):
        if self.dist_type == 'euclidean':
            return torch.norm(inp1 - inp2, dim=1) / 2
        elif self.dist_type == 'cosine':
            h1 = inp1 * inp2
            h2 = (torch.norm(inp1, dim=1) * torch.norm(inp2, dim=1)).unsqueeze(1)
            res = (1 - torch.sum(h1 / h2, dim=1)) / 2
            return res

    def add(self, input, target, **kwargs):
        """
        Computes the distances between given descriptors and saves it and label for inheritors' purposes
        :param input1: Tensor of face descriptors
        :param input2: Tensor of face descriptors
        :param target: Tensor of 0s and 1s. 1 if input1 and input2 belonged to the same person else 0.
        -----
        :param anchor: Tensor of control face descriptors.
        :param positive: Tensor of face descriptors belonging to the same person as the anchor.
        :param negative: Tensor of face descriptors belonging to a user other than the anchor person.
        ------
        :param descriptor: Tensor of face descriptors.
        :param target: Tensor of labels of descriptors' classes.
        """

        if all([(kwargs.get(param, None) is not None) for param in ['input1', 'input2']]):
            distances = self.dist(kwargs['input1'], kwargs['input2']).detach().cpu().numpy()
            labels = target.cpu().numpy()
        elif all([(kwargs.get(param, None) is not None) for param in ['anchor', 'positive', 'negative']]):
            anchor = kwargs['anchor']
            positive = kwargs['positive']
            negative = kwargs['negative']
            distance_ap = self.dist(anchor, positive).detach().cpu().numpy()
            distance_an = self.dist(anchor, negative).detach().cpu().numpy()

            label_p = np.ones_like(distance_ap)
            label_n = np.zeros_like(distance_an)

            distances = np.hstack([distance_ap, distance_an])
            labels = np.hstack([label_p, label_n])
        elif 'S' in kwargs and 'R' in kwargs:
            S = kwargs['S']
            R = kwargs['R']
            if self.dist_type == 'cosine':
                S = (1. - S) / 2.

            distances = S.view(-1).detach().cpu().numpy()
            labels = R.view(-1).cpu().numpy()
        elif 'descriptor' in kwargs:
            descriptors = kwargs['descriptor']
            col1, col2 = torch.tensor(list(combinations(range(target.shape[0]), r=2))).t()
            distances = self.dist(descriptors[col1], descriptors[col2]).detach().cpu().numpy()
            labels = (target[col1] == target[col2]).detach().cpu().numpy()
        else:
            raise TypeError("missing requirement set of arguments: ('input1', 'input2', 'target') "
                            "or ('anchor', 'positive', 'negative') or ('descriptor', 'target')")

        self.distances.append(distances)
        self.labels.append(labels)

    def _merge_data(self):
        if len(self.distances) > 1:
            self.distances = [np.hstack(self.distances)]
            self.labels = [np.hstack(self.labels)]
        elif len(self.distances) == 0:
            raise ValueError('Cannot merge empty list')
        return self.distances[0], self.labels[0]

    def calculate_conf(self, threshold):
        """
        Build a confusion matrix for given differences labels and threshold.
        :param threshold: Threshold of the acceptance of the tho photos as belonged to the same person.
        :return: False accept rate.
        """
        distances, labels = self._merge_data()

        not_labels = labels == 0
        recognized = (distances < threshold)
        not_recognized = recognized == 0

        ta = np.sum(recognized * labels)  # True accepted
        tr = np.sum(not_recognized * not_labels)  # True rejected
        fa = np.sum(recognized * not_labels)  # False accepted
        fr = np.sum(not_recognized * labels)  # False rejected
        return np.array([[ta, fa], [fr, tr]])


class EERMeter(FVMeter):
    def __init__(self, dist='euclidean', init_threshold=0.5, v=0.1, a=0.5, e=1e-7, de=1e-7, override_name='eer_score'):
        """
        Calculates Equal error rate metric with updates in the given confusion matrix
        :param dist: type of distance function. `euclidean` and `cosine` are available now.
        :param init_threshold: Threshold of the acceptance of the tho photos as belonged to the same person.
        :param v: speed of changing threshold.
        :param a: ratio of decreasing v.
        :param e: maximal difference between FAR and FRR to stop.
        :param de: maximal difference between differences of FAR and FRR at current and previous step
        """
        super().__init__(dist, override_name=override_name)
        self.eer = 0.
        self.init_threshold = init_threshold
        self.threshold = self.init_threshold
        self.v = v
        self.a = a
        self.e = e
        self.de = de

    def reset(self):
        """
        Resets internal state of the metric to initial state
        :return: None
        """
        super(EERMeter, self).reset()
        self.threshold = self.init_threshold
        self.eer = 0.

    def value(self) -> float:
        return self.calculate_eer()[0]

    def calculate_eer(self):
        """"
        Equal Error Rate. Used to predetermine the threshold values for its false acceptance rate and its false
        rejection rate. When the rates are equal, the common value is referred to as the equal error rate.
        :return: Equal Error Rate.
        """

        e, v, a = self.e, self.v, self.a
        cur_far = far(self.calculate_conf(self.threshold))
        cur_frr = frr(self.calculate_conf(self.threshold))
        prev_diff = abs(cur_frr - cur_far)
        while prev_diff > e:
            if (cur_frr - cur_far) * v / abs(v) > 0:
                self.threshold += v
            else:
                v *= -a
                self.threshold += v
            cur_far = far(self.calculate_conf(self.threshold))
            cur_frr = frr(self.calculate_conf(self.threshold))
            cur_diff = abs(cur_frr - cur_far)
            if abs(prev_diff - cur_diff) < self.de:
                break
            prev_diff = cur_diff
        self.eer = (cur_frr + cur_far) / 2
        return self.eer, self.threshold


class ROCMeter(FVMeter):
    def __init__(self, dist='euclidean', override_name='roc_score'):
        """
        Calculates Receiver Operating Characteristic
        :param dist: type of distance function. `euclidean` and `cosine` are available now.
        """
        super().__init__(dist, override_name=override_name)

    def value(self) -> float:
        return self.roc_auc()

    def roc_curve(self):
        distances, labels = self._merge_data()
        fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=0)
        thresholds = np.where(thresholds < 1, thresholds, np.ones_like(thresholds))
        return fpr, tpr, thresholds

    def roc_auc(self):
        fpr, tpr, thresholds = self.roc_curve()
        return auc(fpr, tpr)

    def calculate_mer(self):
        """
        Minimal error rate. Used to predetermine the threshold values for its false positive rate and its false
        negative rate. MER is calculated as a minimal sum of rates.
        :return: 3 floats: false positive rate, false negative rate, threshold for which sum of rates is minimal
        """
        #  point = [fpr, tpr, thresh], fnr = 1 - tpr
        fpr, tpr, threshold = min(np.array(self.roc_curve()).T, key=lambda point: 1 - point[1] + point[0])
        return fpr, 1 - tpr, threshold


class TPRFPRMeter(FVMeter):
    def __init__(self, fpr=0.01, dist='euclidean', override_name='tpr@fpr={}'):
        """
        Calculates Receiver Operating Characteristic
        :param dist: type of distance function. `euclidean` and `cosine` are available now.
        """
        super().__init__(dist, override_name=override_name.format(fpr))
        self.default_fpr = fpr

    def value(self) -> float:
        return self.tpr_fpr(self.default_fpr)[0]

    def tpr_fpr(self, fpr):
        """
        Calculate true positive rate and threshold given false positive rate
        :param fpr: float in the range [0, 1] - false positive rate
        :return: nearest false positive rate, true positive rate and threshold
        """
        if not (0 <= fpr <= 1):
            raise ValueError('value of false positive rate is out of bounds [0, 1]')
        distances, labels = self._merge_data()
        d0 = np.sort(distances[labels == 0])
        n = round(fpr * len(d0))
        thresh = d0[n]
        conf = self.calculate_conf(thresh)
        return 1 - frr(conf), thresh
