import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, fbeta_score, precision_score, recall_score
import torch

from src.metrics import AccuracyMeter, f1, fbeta, precision, recall, \
    MeanIntersectionOverUnionMeter, far, frr, EERMeter, precision_k, map_k

from src.metrics.classification import MultiLabelRecallMeter, MultiLabelPrecisionMeter, MultiLabelF1Meter

__all__ = ["AccuracyTest", "F1ScoreTest", "FBetaScoreTest", "PrecisionTest", "RecallTest",
           "MeanIntersectionOverUnionTests", "EERMeterTest", "PrecisionKTest"]


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_MULTICLASS = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                   [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                   [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]]
        self._Y_TRUE_MULTICLASS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    def test_one_iteration(self):
        y_pred_sklearn = np.argmax(self._Y_PRED_MULTICLASS, axis=1)
        scikit_learn_score = accuracy_score(self._Y_TRUE_MULTICLASS, y_pred_sklearn)

        accuracy_test = AccuracyMeter(k=3)

        accuracy_test.add(torch.IntTensor(self._Y_PRED_MULTICLASS), torch.IntTensor(self._Y_TRUE_MULTICLASS))

        self.assertEqual(scikit_learn_score, accuracy_test.value())

    def test_multiple_iterations(self):
        half = len(self._Y_TRUE_MULTICLASS) // 2

        y_pred_sklearn = np.argmax(self._Y_PRED_MULTICLASS, axis=1)
        scikit_learn_score = accuracy_score(self._Y_TRUE_MULTICLASS, y_pred_sklearn)

        accuracy_test = AccuracyMeter(k=3)
        accuracy_test.add(torch.IntTensor(self._Y_PRED_MULTICLASS[:half]),
                          torch.IntTensor(self._Y_TRUE_MULTICLASS[:half]))
        accuracy_test.add(torch.IntTensor(self._Y_PRED_MULTICLASS[half:]),
                          torch.IntTensor(self._Y_TRUE_MULTICLASS[half:]))

        self.assertEqual(scikit_learn_score, accuracy_test.value())


class F1ScoreTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_BINARY = [0, 0, 0, 1, 1, 1, 1, 1]
        self._Y_TRUE_BINARY = [0, 1, 1, 0, 1, 1, 1, 1]

        self._Y_PRED_MULTICLASS = [0, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1]
        self._Y_TRUE_MULTICLASS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    def test_f1_score_with_binary_average(self):
        scikit_learn_score = f1_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY,
                                      average='binary')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        f1_score_calculated = f1(cf, 'binary')

        self.assertAlmostEqual(scikit_learn_score, f1_score_calculated)

    def test_f1_score_with_micro_average(self):
        scikit_learn_score = f1_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                      average='micro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        f1_score_calculated = f1(cf, 'micro')

        self.assertAlmostEqual(scikit_learn_score, f1_score_calculated)

    def test_f1_score_with_macro_average(self):
        scikit_learn_score = f1_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                      average='macro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        f1_score_calculated = f1(cf, 'macro')

        self.assertAlmostEqual(scikit_learn_score, f1_score_calculated)

    def test_f1_score_with_weighted_average_for_binary_classification(self):
        scikit_learn_score = f1_score(self._Y_TRUE_BINARY, self._Y_TRUE_BINARY,
                                      average='weighted')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_TRUE_BINARY)
        f1_score_calculated = f1(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_score, f1_score_calculated)

    def test_f1_score_with_weighted_average_for_multiclass_classification(self):
        scikit_learn_score = f1_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                      average='weighted')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        f1_score_calculated = f1(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_score, f1_score_calculated)

    def test_f1_score_for_each_class_of_multiclass_classification(self):
        scikit_learn_scores = list(f1_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                            average=None))

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        f1_score_calculateds = f1(cf, None)

        self.assertListEqual(scikit_learn_scores, f1_score_calculateds)


class FBetaScoreTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_BINARY = [0, 0, 0, 1, 1, 1, 1, 1]
        self._Y_TRUE_BINARY = [0, 1, 1, 0, 1, 1, 1, 1]

        self._Y_PRED_MULTICLASS = [0, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1]
        self._Y_TRUE_MULTICLASS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

        self._FBETA = 0.5

    def test_fbeta_score_with_binary_average(self):
        scikit_learn_score = fbeta_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY,
                                         average='binary', beta=self._FBETA)

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        fbeta_score_calculated = fbeta(cf, 'binary', self._FBETA)

        self.assertAlmostEqual(scikit_learn_score, fbeta_score_calculated)

    def test_fbeta_score_with_micro_average(self):
        scikit_learn_score = fbeta_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                         average='micro', beta=self._FBETA)

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        fbeta_score_calculated = fbeta(cf, 'micro', self._FBETA)

        self.assertAlmostEqual(scikit_learn_score, fbeta_score_calculated)

    def test_fbeta_score_with_macro_average(self):
        scikit_learn_score = fbeta_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                         average='macro', beta=self._FBETA)

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        fbeta_score_calculated = fbeta(cf, 'macro', self._FBETA)

        self.assertAlmostEqual(scikit_learn_score, fbeta_score_calculated)

    def test_fbeta_score_with_weighted_average_for_binary_classification(self):
        scikit_learn_score = fbeta_score(self._Y_TRUE_BINARY, self._Y_TRUE_BINARY,
                                         average='weighted', beta=self._FBETA)

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_TRUE_BINARY)
        fbeta_score_calculated = fbeta(cf, 'weighted', self._FBETA)

        self.assertAlmostEqual(scikit_learn_score, fbeta_score_calculated)

    def test_fbeta_score_with_weighted_average_for_multiclass_classification(self):
        scikit_learn_score = fbeta_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                         average='weighted', beta=self._FBETA)

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        fbeta_score_calculated = fbeta(cf, 'weighted', self._FBETA)

        self.assertAlmostEqual(scikit_learn_score, fbeta_score_calculated)

    def test_fbeta_score_for_each_class_of_multiclass_classification(self):
        scikit_learn_scores = list(fbeta_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                               average=None, beta=self._FBETA))

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        fbeta_scores_calculated = fbeta(cf, None, self._FBETA)

        self.assertListEqual(list(np.round(scikit_learn_scores, 7)), list(np.round(fbeta_scores_calculated, 7)))


class PrecisionTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_BINARY = [0, 0, 0, 1, 1, 1, 1, 1]
        self._Y_TRUE_BINARY = [0, 1, 1, 0, 1, 1, 1, 1]

        self._Y_PRED_MULTICLASS = [0, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1]
        self._Y_TRUE_MULTICLASS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    def test_precision_score_with_binary_average(self):
        scikit_learn_score = precision_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY,
                                             average='binary')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        precision_test = precision(cf, 'binary')

        self.assertAlmostEqual(scikit_learn_score, precision_test)

    def test_precision_score_with_micro_average(self):
        scikit_learn_score = precision_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                             average='micro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        precision_test = precision(cf, 'micro')

        self.assertAlmostEqual(scikit_learn_score, precision_test)

    def test_precision_score_with_macro_average(self):
        scikit_learn_score = precision_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                             average='macro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        precision_test = precision(cf, 'macro')

        self.assertAlmostEqual(scikit_learn_score, precision_test)

    def test_precision_score_with_weighted_average_for_binary_classification(self):
        scikit_learn_scores = precision_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY,
                                              average='weighted')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        precision_test = precision(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_scores, precision_test)

    def test_precision_score_with_weighted_average_for_multiclass_classification(self):
        scikit_learn_scores = precision_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                              average='weighted')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        precision_test = precision(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_scores, precision_test)

    def test_precision_score_for_each_class_of_multiclass_classification(self):
        scikit_learn_scores = list(precision_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                                   average=None))

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        precision_test = precision(cf, None)

        self.assertListEqual(scikit_learn_scores, precision_test)


class RecallTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_BINARY = [0, 0, 0, 1, 1, 1, 1, 1]
        self._Y_TRUE_BINARY = [0, 1, 1, 0, 1, 1, 1, 1]

        self._Y_PRED_MULTICLASS = [0, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1]
        self._Y_TRUE_MULTICLASS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    def test_recall_score_with_binary_average(self):
        scikit_learn_score = recall_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY, average='binary')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        recall_test = recall(cf, 'binary')

        self.assertAlmostEqual(scikit_learn_score, recall_test)

    def test_recall_score_with_micro_average(self):
        scikit_learn_score = recall_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS, average='micro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        recall_test = recall(cf, 'micro')

        self.assertAlmostEqual(scikit_learn_score, recall_test)

    def test_recall_score_with_macro_average(self):
        scikit_learn_score = recall_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                          average='macro')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        recall_test = recall(cf, 'macro')

        self.assertAlmostEqual(scikit_learn_score, recall_test)

    def test_recall_score_with_weighted_average_for_binary_classification(self):
        scikit_learn_scores = recall_score(self._Y_TRUE_BINARY, self._Y_PRED_BINARY,
                                           average='weighted')

        cf = confusion_matrix(self._Y_TRUE_BINARY, self._Y_PRED_BINARY)
        precision_test = recall(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_scores, precision_test)

    def test_recall_score_with_weighted_average_for_multiclass_classification(self):
        scikit_learn_scores = recall_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                           average='weighted')

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        precision_test = recall(cf, 'weighted')

        self.assertAlmostEqual(scikit_learn_scores, precision_test)

    def test_recall_score_for_each_class_of_multiclass_classification(self):
        scikit_learn_scores = list(recall_score(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS,
                                                average=None))

        cf = confusion_matrix(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)
        recall_test = recall(cf, None)

        self.assertListEqual(scikit_learn_scores, recall_test)


class MeanIntersectionOverUnionTests(unittest.TestCase):
    def setUp(self):
        self._predictions = torch.load('./tests/metrics/configs/predictions_4classes.pt')
        self._target = torch.load('./tests/metrics/configs/target_4classes.pt')

    def test_miou_averaged_compared_to_straight_calculations(self):
        preds = self._predictions.argmax(1).squeeze()
        # Calculate ious for each of two classes
        # (actually there are 4 classes, but they don't have predictions and targets in this example)
        tp0 = int(((preds == 0) & (self._target == 0)).sum())
        union0 = int(((self._target == 0) | (preds == 0)).sum())
        tp1 = int(((preds == 1) & (self._target == 1)).sum())
        union1 = int(((self._target == 1) | (preds == 1)).sum())

        miou_expected = np.mean([tp1 / union1, tp0 / union0])

        miou_meter = MeanIntersectionOverUnionMeter(k=4, weighted=False)
        miou_meter.add(self._predictions, self._target)
        miou_actual = miou_meter.value()

        self.assertAlmostEqual(miou_expected, miou_actual,
                               msg='MIOU calculated from confusion matrix must match'
                                   'the MIOU calculated straightly')

    def test_miou_weighted_compared_to_straight_calculations(self):
        preds = self._predictions.argmax(1).squeeze()
        tp0 = int(((preds == 0) & (self._target == 0)).sum())
        gt0 = int((self._target == 0).sum())
        union0 = int(((self._target == 0) | (preds == 0)).sum())
        tp1 = int(((preds == 1) & (self._target == 1)).sum())
        gt1 = int((self._target == 1).sum())
        union1 = int(((self._target == 1) | (preds == 1)).sum())
        total = torch.numel(self._target)  # two classes are counted

        weights = np.array([gt1 / total, gt0 / total])
        miou_expected = np.sum(np.array([tp1 / union1, tp0 / union0]) * weights)

        miou_meter = MeanIntersectionOverUnionMeter(k=4, weighted=True)
        miou_meter.add(self._predictions, self._target)
        miou_actual = miou_meter.value()

        self.assertAlmostEqual(miou_expected, miou_actual,
                               msg='MIOU calculated from confusion matrix must match'
                                   'the MIOU calculated straightly')


class EERMeterTest(unittest.TestCase):

    def setUp(self) -> None:
        # shape of descs is (13233, 512)
        # shape of triplets is (242257, 3)
        descs = np.load('tests/metrics/configs/descriptors.npy')
        self.triplets = np.load('tests/metrics/configs/descriptors_triplets.npy')
        self.descs = torch.from_numpy(descs)
        t1, t2, t3 = self.triplets.T
        anchors, positives, negatives = self.descs[t1], self.descs[t2], self.descs[t3]

        eer_meter_c = EERMeter(dist='cosine', e=1e-8)
        eer_meter_c.add(anchor=anchors, positive=positives, negative=negatives)

        eer_meter_e = EERMeter(dist='euclidean', e=1e-8)
        eer_meter_e.add(anchor=anchors, positive=positives, negative=negatives)

        def calculate_conf(threshold, distances, labels):
            not_labels = labels == 0
            recognized = (distances < threshold)
            not_recognized = recognized == 0

            ta = np.sum(recognized * labels)  # True accepted
            tr = np.sum(not_recognized * not_labels)  # True rejected
            fa = np.sum(recognized * not_labels)  # False accepted
            fr = np.sum(not_recognized * labels)  # False rejected
            return np.array([[ta, fa], [fr, tr]])

        self.functions = [calculate_conf]
        self.eer_meter = {'cosine': eer_meter_c, 'euclidean': eer_meter_e}

    def test_eer_calculator_cosine(self):
        distances, labels = self.eer_meter['cosine']._merge_data()
        res_eer, res_threshold = self.eer_meter['cosine'].calculate_eer()

        calculate_conf = self.functions[0]

        conf = calculate_conf(res_threshold, distances, labels)
        self.assertAlmostEqual(far(conf), frr(conf), places=5)

    def test_eer_calculator_euclidean(self):
        distances, labels = self.eer_meter['euclidean']._merge_data()
        res_eer, res_threshold = self.eer_meter['euclidean'].calculate_eer()

        calculate_conf = self.functions[0]

        conf = calculate_conf(res_threshold, distances, labels)
        self.assertAlmostEqual(far(conf), frr(conf), places=4)

    def test_adder_differ_batch(self):
        eer_meter = EERMeter(dist='cosine')
        for triplet in self.triplets:
            eer_meter.add(anchor=torch.FloatTensor(self.descs[triplet[0]]).unsqueeze(0),
                          positive=torch.FloatTensor(self.descs[triplet[1]]).unsqueeze(0),
                          negative=torch.FloatTensor(self.descs[triplet[2]]).unsqueeze(0))
        eer, thresh = eer_meter.calculate_eer()
        eer_batch, thresh_batch = self.eer_meter['cosine'].calculate_eer()

        self.assertAlmostEqual(eer, eer_batch)
        self.assertAlmostEqual(thresh, thresh_batch)


class PrecisionKTest(unittest.TestCase):

    def test_precision_k(self):
        scores = torch.tensor([0.0161, 0.8738, 0.9902, 0.6572, 0.7809, 0.8914, 0.4809, 0.1859, 0.5298, 0.4107])
        targets = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
        # top k largest - [0.9902, 0.8914, 0.8738, 0.7809, 0.6572]
        # its labels - [0, 1, 1, 0, 1]
        k = 5
        res = precision_k(targets, scores, k, average=False, largest=True)

        self.assertAlmostEqual(0.6, res)

    def test_average_precision_k(self):
        scores = torch.tensor([0.0161, 0.8738, 0.9902, 0.6572, 0.7809, 0.8914, 0.4809, 0.1859, 0.5298, 0.4107])
        targets = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
        k = 5
        # top k largest - [0.9902, 0.8914, 0.8738, 0.7809, 0.6572]
        # its labels - [0, 1, 1, 0, 1]
        # p@k scores [0/1, 1/2, 2/3, 2/4, 3/5]
        res = precision_k(targets, scores, k, average=True, largest=True)
        true_res = sum([0 / 1, 1 / 2, 2 / 3, 2 / 4, 3 / 5]) / 5
        self.assertAlmostEqual(true_res, res)

    def test_mean_average_precision_k(self):
        scores = torch.tensor([[0.2614, 0.0597, 0.0944, 0.8855, 0.3787, 0.5478, 0.3526, 0.6047, 0.8501],
                               [0.1236, 0.4361, 0.4701, 0.2407, 0.9463, 0.4645, 0.8434, 0.3978, 0.3678],
                               [0.1513, 0.8694, 0.8661, 0.2258, 0.9227, 0.8190, 0.5648, 0.6747, 0.9955]])
        targets = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 1, 1, 1],
                                [1, 1, 0, 1, 1, 0, 0, 0, 0]])
        k = 5
        # top k largest and its labels -
        # [0.8855, 0.8501, 0.6047, 0.5478, 0.3787], [0, 1, 1, 0, 0]
        # [0.9463, 0.8434, 0.4701, 0.4645, 0.4361], [0, 1, 0, 0, 1]
        # [0.9955, 0.9227, 0.8694, 0.8661, 0.8190], [0, 1, 1, 0, 0]
        # ap@k scores -
        # mean([0/1, 1/2, 2/3, 2/4, 2/5]) = 0.4133(3)
        # mean([0/1, 1/2, 1/3, 1/4, 2/5]) = 0.2966(6)
        # mean([0/1, 1/2, 2/3, 2/4, 2/5]) = 0.4133(3)
        res = map_k(targets, scores, k, largest=True)
        true_res = (1.24 / 3 + 0.89 / 3 + 1.24 / 3) / 3
        self.assertAlmostEqual(true_res, res)


class MultiLabelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = np.array([[0.2, 0.3, 0.4, 0.2, 0.1, 0.4, 0.6, 0.7, 0.8, 0.5],
                                [0.1, 0.3, 0.4, 0.5, 0.9, 0.6, 0.4, 0.3, 0.3, 0.3],
                                [0.5, 0.5, 0.3, 0.6, 0.7, 0.8, 0.2, 0.1, 0.6, 0.2]])

        self.targets = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                                 [1, 1, 0, 1, 1, 0, 0, 0, 0, 1]])

        self.true_pos = np.array([1., 1., 0., 1., 1., 0., 1., 1., 1., 1.])
        self.false_pos = np.array([0., 0., 0., 1., 1., 2., 0., 0., 1., 0.])
        self.false_neg = np.array([1., 1., 0., 0., 0., 0., 1., 1., 1., 2.])

        self.inverse_sigmoid = lambda x: -np.log(1. / (x + 1e-9) - 1.)
        self.scores_logits = self.inverse_sigmoid(self.scores)
        self.threshold = 0.5
        self.thresholds = [0.1, 0.5, 0.8]

class MultiLabelF1MeterTest(MultiLabelTest):
    def test_calculate(self):
        metric = MultiLabelF1Meter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(self.threshold)
            y_true = self.targets[None, i]
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = f1_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_calculate_different_thresholds(self):
        for threshold in self.thresholds:
            metric = MultiLabelF1Meter(threshold=threshold, num_classes=10, average='macro')
            y_pred = self.scores_logits
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(threshold)
            y_true = self.targets
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = f1_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = MultiLabelF1Meter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_on_epoch_end_macro(self):
        metric = MultiLabelF1Meter(threshold=self.threshold, num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = f1_score(self.targets, y_pred_thresholded, average='macro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_on_epoch_end_weighted(self):
        metric = MultiLabelF1Meter(threshold=self.threshold, num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = f1_score(self.targets, y_pred_thresholded, average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)


class MultiLabelRecallMeterTest(MultiLabelTest):
    def test_calculate(self):
        metric = MultiLabelRecallMeter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(self.threshold)
            y_true = self.targets[None, i]
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = recall_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_calculate_different_thresholds(self):
        for threshold in self.thresholds:
            metric = MultiLabelRecallMeter(threshold=threshold, num_classes=10, average='macro')
            y_pred = self.scores_logits
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(threshold)
            y_true = self.targets
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = recall_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = MultiLabelRecallMeter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_on_epoch_end_macro(self):
        metric = MultiLabelRecallMeter(threshold=self.threshold, num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = recall_score(self.targets, y_pred_thresholded, average='macro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_on_epoch_end_weighted(self):
        metric = MultiLabelRecallMeter(threshold=self.threshold, num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = recall_score(self.targets, y_pred_thresholded, average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)


class MultiLabelPrecisionMeterTest(MultiLabelTest):
    def test_calculate(self):
        metric = MultiLabelPrecisionMeter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(self.threshold)
            y_true = self.targets[None, i]
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = precision_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_calculate_different_thresholds(self):
        for threshold in self.thresholds:
            metric = MultiLabelPrecisionMeter(threshold=threshold, num_classes=10, average='macro')
            y_pred = self.scores_logits
            y_pred_thresholded = y_pred >= self.inverse_sigmoid(threshold)
            y_true = self.targets
            tested_metric_result = metric.calculate(y_true, y_pred)
            gt_sklearn_result = precision_score(y_true=y_true, y_pred=y_pred_thresholded, average='macro')
            self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = MultiLabelPrecisionMeter(threshold=self.threshold, num_classes=10)
        for i in range(3):
            y_pred = self.scores_logits[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_on_epoch_end_macro(self):
        metric = MultiLabelPrecisionMeter(threshold=self.threshold, num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = precision_score(self.targets, y_pred_thresholded, average='macro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_on_epoch_end_weighted(self):
        metric = MultiLabelPrecisionMeter(threshold=self.threshold, num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        y_pred_thresholded = self.scores_logits >= self.inverse_sigmoid(self.threshold)
        gt_sklearn_result = precision_score(self.targets, y_pred_thresholded, average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)


if __name__ == '__main__':
    unittest.main()
