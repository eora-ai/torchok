import math
from abc import ABC
from typing import List, Optional, Union

import faiss
import numpy as np
import pandas as pd
from rank_eval import map as mean_ap, recall_at_k, precision_at_k
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import row_norms

from src.registry import METRICS
from .common import Metric


def cosine_distances(x: np.ndarray, y: np.ndarray = None):
    """
    When Y is None it computes the cosine distance between every possible unordered pair of samples.
    The computed distances will be in the interval [0, 1]. The samples will be paired in
    the following way by indices:
        {(i, j) for j in i+1...n-1
           for i in 0...n-2}  where n is the number of samples
    When Y is given it computes the cosine distance between pairs from X and Y. In this case X and Y
    should have same size.
    Args:
       x: numpy array of shape (n_samples, n_features)
       y (optional): numpy array of shape (n_samples, n_features)

    Returns:
       If Y is None, returns a numpy array of size n*(n-1)/2 containing the computed distances,
       otherwise returns a numpy of array of size n.
   """
    if y is None:
        x_normalized = normalize(x)
        stack_x = []
        for i in range(x_normalized.shape[0] - 1):
            stack_x.append(x_normalized[i + 1:].dot(x_normalized[i]))

        d = np.concatenate(stack_x)
        return (1 - d) / 2
    else:
        x_normalized = normalize(x)
        y_normalized = normalize(y)
        p = np.sum(x_normalized * y_normalized, axis=1)
        return (1 - p) / 2


def euclidean_distances(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """
    When Y is None it computes the euclidean distance between every possible unordered pair of samples.
    The samples will be paired in the following way by indices:
        {(i, j) for j in i+1...n-1
            for i in 0...n-2}  where n is the number of samples
    When Y is given it computes the euclidean distance between pairs from X and Y. In this case X and Y
    should have same size.

    Args:
        x: numpy array of shape (n_samples, n_features)
        y (optional): numpy array of shape (n_samples, n_features)

    Returns:
       If Y is None, returns a numpy array of size n*(n-1)/2 containing the computed distances,
       otherwise returns a numpy of array of size n.
    """
    if y is None:
        xx = row_norms(x, squared=True)
        stack_x = []
        for i in range(x.shape[0] - 1):
            stack_x.append(np.sqrt(xx[i] - 2 * x[i + 1:].dot(x[i]) + xx[i + 1:]))
        d = np.concatenate(stack_x)

        return d
    else:
        return np.linalg.norm(x - y, axis=1)


def fpr_score(conf_mtx):
    """Computes FPR (false positive rate) given confusion matrix"""
    [tp, fp], [fn, tn] = conf_mtx
    r = tn + fp
    return fp / r if r > 0 else 0


def fnr_score(conf_mtx):
    """Computes FNR (false negative rate) given confusion matrix"""
    [tp, fp], [fn, tn] = conf_mtx
    a = tp + fn
    return fn / a if a > 0 else 0


def compute_tprs(fprs: List[float], distances: np.ndarray, labels: np.ndarray):
    """Computes TPRs given FPRs, distances and labels

    Args:
        fprs: A list of fpr values all in range [0, 1)
        distances: A numpy array of distances
        labels: A boolean numpy array of labels indicating weather the corresponding pair
            is of the same class
    """
    if labels.dtype != bool:
        labels = np.array(labels, dtype=bool)
    positives = distances[labels]
    negatives = np.sort(distances[~labels])
    n = len(negatives)
    p = labels.shape[0] - n
    tprs = []
    for fpr in fprs:
        fp = int(round(fpr * n))
        margin = negatives[fp]
        tp = np.sum(positives < margin)
        tpr = tp / p if p > 0 else 1
        tprs.append(tpr)
    return tprs


class ROCBase(Metric):
    def __init__(self, name: str, distance: str = 'cosine', target_fields: dict = None):
        name += '_' + distance
        super().__init__(name=name, target_fields=target_fields)
        self.distance = distance
        self._distances = []
        self._labels = []

    @property
    def distances(self):
        if not self._distances:
            return np.array([])
        return np.concatenate(self._distances)

    @property
    def labels(self):
        if not self._labels:
            return np.array([])
        return np.concatenate(self._labels)

    def dist(self, x: np.ndarray, y: np.ndarray = None):
        if self.distance == 'euclidean':
            distance_func = euclidean_distances
        elif self.distance == 'cosine':
            distance_func = cosine_distances
        else:
            raise ValueError(f"Distance should be one of 'euclidean' or 'cosine'")
        return distance_func(x, y)

    def update(self, **kwargs):
        """
        Computes the distances between given descriptors and saves it and label for inheritors' purposes
        :param vectors: Numpy array of face descriptors.
        :param target: Numpy array of labels of descriptors' classes.
        ------
        :param anchor: Numpy array of control face descriptors.
        :param positive: Numpy array of face descriptors belonging to the same person as the anchor.
        :param negative: Numpy array of face descriptors belonging to a user other than the anchor person.
        """
        if all([(kwargs.get(param) is not None) for param in ['target', 'vectors']]):
            labels = kwargs['target']
            vectors = kwargs['vectors']
            self._distances.append(self.dist(vectors))
            s = []
            for i in range(len(labels) - 1):
                s.append(labels[i] == labels[i + 1:])
            self._labels.append(np.concatenate(s))

        elif all([(kwargs.get(param) is not None) for param in ['anchor', 'positive', 'negative']]):
            anchor = kwargs['anchor']
            positive = kwargs['positive']
            negative = kwargs['negative']
            distance_ap = self.dist(anchor, positive)
            distance_an = self.dist(anchor, negative)

            label_p = np.ones_like(distance_ap, dtype=bool)
            label_n = np.zeros_like(distance_an, dtype=bool)

            distances = np.hstack([distance_ap, distance_an])
            labels = np.hstack([label_p, label_n])
            self._labels.append(labels)
            self._distances.append(distances)
        else:
            raise TypeError("missing requirement set of arguments: "
                            "('anchor', 'positive', 'negative') or ('vectors', 'target')")

    def calculate(self, **kwargs):
        old_distances = self._distances
        old_labels = self._labels
        self.reset()
        self.update(**kwargs)
        res = self.on_epoch_end()
        self._labels = old_labels
        self._distances = old_distances
        return res

    def on_epoch_end(self):
        raise NotImplementedError()

    def reset(self):
        self._distances = []
        self._labels = []

    def calculate_conf(self, threshold):
        """
        Build a confusion matrix for given differences labels and threshold.
        :param threshold: Threshold of the acceptance of the tho photos as belonged to the same person.
        :return: False accept rate.
        """
        distances = self.distances
        labels = self.labels

        not_labels = ~labels
        recognized = (distances < threshold)
        not_recognized = ~recognized

        tp = np.sum(recognized * labels)  # True accepted
        tn = np.sum(not_recognized * not_labels)  # True rejected
        fp = np.sum(recognized * not_labels)  # False accepted
        fn = np.sum(not_recognized * labels)  # False rejected
        return np.array([[tp, fp], [fn, tn]])


@METRICS.register_class
class TPRatFPRMeter(ROCBase):
    def __init__(self, fprs: Union[float, List[float]], name: str = 'TPR@FPR',
                 distance: str = 'cosine', target_fields: dict = None):
        """
        Args:
            fprs: Either a list of or a single fpr value
            name: Name of the metric
            distance: Name of the distance function, one of 'euclidean' or 'cosine'
        """
        if isinstance(fprs, float):
            name = f'{name}={fprs}'
        super().__init__(name=name, distance=distance, target_fields=target_fields)
        if isinstance(fprs, float):
            self.fprs = [fprs]
            # this flag will be used to determine weather to return a single float tpr value 
            # or a list of tpr values for each fpr (even if fprs list contained one value)
            self._single_fpr = True
        else:
            self._single_fpr = False
            self.fprs = fprs

    def on_epoch_end(self):
        tprs = compute_tprs(self.fprs, self.distances, self.labels)
        self.reset()
        if self._single_fpr:
            # it means a single fpr value was given in the constructor
            # so we will return a single tpr value
            return tprs[0]
        # return the list of computed tprs
        return tprs


@METRICS.register_class
class EERMeter(ROCBase):
    def __init__(self, distance='cosine', name='eer_score', target_fields: dict = None):
        """
        Calculates Equal error rate metric with updates in the given confusion matrix
        :param distance: type of distance function. `euclidean` and `cosine` are available now.
        """
        super().__init__(name, distance, target_fields=target_fields)
        self.eer = 0.
        self.threshold = 0.

    def on_epoch_end(self):
        eer, threshold = self.calculate_eer()
        self.reset()
        return eer

    def calculate_eer(self):
        """"
        Equal Error Rate. Used to predetermine the threshold values for its false acceptance rate and its false
        rejection rate. When the rates are equal, the common value is referred to as the equal error rate.
        In this particular implementation we define eer as:
            eer = (fpr + fnr)/2, where fpr is the largest fpr value not greater than its corresponding fnr
        :return: Equal Error Rate.
        """
        idx = 0
        possible_thresholds = np.sort(self.distances)
        cur_threshold = possible_thresholds[idx]
        conf = self.calculate_conf(cur_threshold)
        cur_fpr = fpr_score(conf)
        cur_fnr = fnr_score(conf)
        high = len(self.labels)
        j = 2 ** math.ceil(math.log2(high - idx))  # the smallest power of 2 not less than high-idx
        while j > 0:
            new_threshold = possible_thresholds[idx + j] if idx + j < high else possible_thresholds[-1] + 0.01
            conf = self.calculate_conf(new_threshold)
            new_fpr = fpr_score(conf)
            new_fnr = fnr_score(conf)
            if new_fpr <= new_fnr:
                idx += j
                cur_fnr = new_fnr
                cur_fpr = new_fpr
                cur_threshold = new_threshold
            j = j // 2

        self.eer = (cur_fnr + cur_fpr) / 2
        self.threshold = cur_threshold
        return self.eer, self.threshold


class IndexBasedMeter(Metric, ABC):
    metrics = {'IP': 0, 'L2': 1}

    def __init__(self, exact_index: bool, metric: Union[str, int], name: str, target_fields: dict,
                 self_retrieval_mode=True, normalize_input: bool = False):
        super().__init__(name, target_fields)
        self.exact_index = exact_index
        self.self_retrieval_mode = self_retrieval_mode
        if not (metric in self.metrics or metric in self.metrics.values()):
            raise ValueError("`metric` must be " + ' | '.join([f"{i}" for j in self.metrics.items() for i in j]))
        self.metric = self.metrics.get(metric, metric)
        self.normalize_input = normalize_input

        self.reset()

    def reset(self):
        self.__vectors = []
        self.__labels = []

    def update(self, vectors: np.ndarray, target: Optional[np.ndarray] = None, **kwargs):
        """
        Save vectors and labels
        :param vectors: descriptors or embeddings that will be used in metric calculation.
        :param target: labels of vectors' classes.
        """
        self.__vectors.append(vectors)
        self.__labels.append(target)

    def calculate(self, vectors: np.ndarray, target: Optional[np.ndarray] = None,
                  k: int = None, batch_search: bool = False, **kwargs):
        """
        Perform indexing and return indexes of closest vectors for each class. For every class the first appeared vector
        of this class is considered as a request vector and left vectors are used in indexing. If there is only one
        vector of the class then this vector is left for indexing.
        :param vectors: (N, d) array of descriptors or embeddings that will be used in metric calculation.
        :param target: (N,) array of labels of vectors' classes.
        :param k: number of closest vectors to be return
        :param batch_search: If True, all searches will be performed simultaneously and the array with predictions will
            be returned, otherwise the generator well be returned. Enabling this feature is memory consuming.
        :return iterator that produces tuple of (relevant_idx, closest_idx, closest_dist) if return_distances is True,
            otherwise it produces closest_idx.
            `relevant_idx` is the np.ndarray of `m` indexes of relevant vectors to query vector. `m` can be different.
            `closest_idx` is the np.ndarray of `k` indexes of closest vectors to query vector.
            `closest_dist` is the np.ndarray of `k` distances of closest vectors to query vector.
        """
        vectors = vectors.astype(np.float32)
        if self.normalize_input:
            vectors = normalize(vectors)

        if self.self_retrieval_mode:
            ts = pd.Series(target)

            # Group item indices of the same class
            groups = pd.Series(ts.groupby(ts).groups)

            # Take the first index of the group as a query
            # If there is only one element in the group, ignore this group
            query_idxs = groups.apply(lambda x: x[0] if len(x) > 1 else -1)

            # Rest indices are used as a relevant/database elements
            relevant = groups.apply(lambda x: x[1:].values if len(x) > 1 else x.values)

            # Create mapping from database index to original index
            db_idxs = np.array(sorted([i for j in relevant for i in j]))

            # Filter groups with more than one sample per group
            correct_classes = query_idxs != -1
            relevant = relevant[correct_classes]
            query_idxs = query_idxs[correct_classes]

            # Retrieve query and database vectors and build index base on latter.
            query_vecs = vectors[query_idxs]
            db_vecs = vectors[db_idxs]
            index = self.build_index(db_vecs)
            exclude_queries = False
        else:
            # Skip query items, take relevant and database items
            query_idxs = np.where(target[:, 0] == 1)[0]
            db_idxs = np.arange(target.shape[0])
            relevant = []
            for q_idx in query_idxs:
                cur_rel_idxs = np.where(target[:, 1 + q_idx] > 0.)[0]
                relevant.append(cur_rel_idxs)

            query_vecs = vectors[query_idxs]
            db_vecs = vectors[db_idxs]
            print(f"db_size: {len(db_vecs)}")

            index = self.build_index(db_vecs)
            exclude_queries = True

        k = len(db_vecs) if k is None else k
        if batch_search:
            if exclude_queries:
                closest_dist, closest_idx = index.search(query_vecs, k=k + 1)
                closest_dist, closest_idx = closest_dist[:, 1:], closest_idx[:, 1:]
            else:
                closest_dist, closest_idx = index.search(query_vecs, k=k)

            if self.metric == 0:
                closest_dist = 1 - closest_dist
            closest = np.dstack([db_idxs[closest_idx], closest_dist])
            relevant = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), relevant)

            return zip(relevant, closest)
        else:
            return self.query_generator(index, relevant, query_vecs, db_idxs, k, exclude_queries)

    def query_generator(self, index, relevants, queries, db_ids, k, exclude_queries):
        def generator():
            for relevant, query in zip(relevants, queries):
                if exclude_queries:
                    closest_dist, closest_idx = index.search(query[None], k=k + 1)
                    closest_dist, closest_idx = closest_dist[:, 1:], closest_idx[:, 1:]
                else:
                    closest_dist, closest_idx = index.search(query[None], k=k)

                if self.metric == 0:
                    closest_dist = 1 - closest_dist

                closest = np.dstack([db_ids[closest_idx], closest_dist])[0]
                relevant = np.stack((relevant, np.ones_like(relevant)), axis=1)

                yield relevant, closest

        return generator()

    def build_index(self, vectors: np.ndarray):
        """
        Performs indexing of a given set of vectors.
        Trains index on all the descriptors and adds the same descriptors to be able to search through
        :param vectors: np.ndarray of shape (N, d) representing N vectors of d dimensionality to be loaded
        :return: Constructed index.
        """
        n, d = vectors.shape

        index_class = faiss.IndexFlatIP if self.metric == 0 else faiss.IndexFlatL2
        if self.exact_index:
            index = index_class(d)
        else:
            nlist = 4 * math.ceil(n ** 0.5)
            quantiser = index_class(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist, self.metric)
            index.train(vectors)

        index.add(vectors)

        return index

    def on_epoch_end(self, do_reset=True, k: int = None, batch_search: bool = False):
        vectors = np.concatenate(self.__vectors, axis=0)
        target = np.concatenate(self.__labels, axis=0)

        if do_reset:
            self.reset()
        return self.calculate(vectors, target, k=k, batch_search=batch_search)


@METRICS.register_class
class MeanAveragePrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, k: int, batch_search: bool, exact_index: bool, metric: Union[str, int],
                 target_fields: dict, name: str = None, self_retrieval_mode: bool = True, normalize_input: bool = False):
        if name is None:
            name = f'mAP_at_{k}'
        super().__init__(exact_index, metric, name, target_fields, self_retrieval_mode, normalize_input)
        self.k = k
        self.batch_search = batch_search

    def calculate(self, vectors: np.ndarray, target: np.ndarray = None, **kwargs):
        scores = []
        generator = super().calculate(vectors, target, k=self.k, batch_search=self.batch_search)
        for relevant_idx, closest_idx in generator:
            scores.append(mean_ap(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)

    def on_epoch_end(self, do_reset=True, **kwargs):
        return super().on_epoch_end(do_reset)


@METRICS.register_class
class RecallAtKMeter(IndexBasedMeter):
    def __init__(self, k: int, batch_search: bool, exact_index: bool, metric: Union[str, int],
                 target_fields: dict, name: str = None, self_retrieval_mode: bool = True,
                 normalize_input: bool = False):
        if name is None:
            name = f'recall_at_{k}'
        super().__init__(exact_index, metric, name, target_fields, self_retrieval_mode, normalize_input)
        self.k = k
        self.batch_search = batch_search

    def calculate(self, vectors: np.ndarray, target: np.ndarray = None, **kwargs):
        scores = []
        generator = super().calculate(vectors, target, k=self.k, batch_search=self.batch_search)
        for relevant_idx, closest_idx in generator:
            scores.append(recall_at_k(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)

    def on_epoch_end(self, do_reset=True, **kwargs):
        return super().on_epoch_end(do_reset)


@METRICS.register_class
class PrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, k: int, batch_search: bool, exact_index: bool, metric: Union[str, int],
                 target_fields: dict, name: str = None, self_retrieval_mode: bool = True,
                 normalize_input: bool = False):
        if name is None:
            name = f'precision_at_{k}'
        super().__init__(exact_index, metric, name, target_fields, self_retrieval_mode, normalize_input)
        self.k = k
        self.batch_search = batch_search

    def calculate(self, vectors: np.ndarray, target: np.ndarray = None, **kwargs):
        scores = []
        generator = super().calculate(vectors, target, k=self.k, batch_search=self.batch_search)
        for relevant_idx, closest_idx in generator:
            scores.append(precision_at_k(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)

    def on_epoch_end(self, do_reset=True, **kwargs):
        return super().on_epoch_end(do_reset)
