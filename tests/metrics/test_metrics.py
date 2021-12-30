import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
import torch

from src.metrics import AccuracyMeter, F1Meter, FbetaMeter, PrecisionMeter, RecallMeter, \
    MeanIntersectionOverUnionMeter, far, frr, EERMeter

from src.metrics.classification import MultiLabelRecallMeter, MultiLabelPrecisionMeter, MultiLabelF1Meter

__all__ = ["AccuracyTest", "F1ScoreTest", "FBetaScoreTest", "PrecisionTest", "RecallTest",
           "MeanIntersectionOverUnionTests", "EERMeterTest"]


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        self._Y_PRED_MULTICLASS = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
            dtype=np.float32
        )
        self._Y_TRUE_MULTICLASS = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            dtype=np.float32
        )

    def test_one_iteration(self):
        y_pred_sklearn = np.argmax(self._Y_PRED_MULTICLASS, axis=1)
        scikit_learn_score = accuracy_score(self._Y_TRUE_MULTICLASS, y_pred_sklearn)

        accuracy_test = AccuracyMeter()

        accuracy_test.update(self._Y_TRUE_MULTICLASS, self._Y_PRED_MULTICLASS)

        self.assertEqual(scikit_learn_score, accuracy_test.on_epoch_end())

    def test_multiple_iterations(self):
        half = len(self._Y_TRUE_MULTICLASS) // 2

        y_pred_sklearn = np.argmax(self._Y_PRED_MULTICLASS, axis=1)
        scikit_learn_score = accuracy_score(self._Y_TRUE_MULTICLASS, y_pred_sklearn)

        accuracy_test = AccuracyMeter()
        accuracy_test.update(self._Y_TRUE_MULTICLASS[:half],
                             self._Y_PRED_MULTICLASS[:half])
        accuracy_test.update(self._Y_TRUE_MULTICLASS[half:],
                             self._Y_PRED_MULTICLASS[half:])

        self.assertEqual(scikit_learn_score, accuracy_test.on_epoch_end())


class ClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Multi-class classification
        self.scores = np.array(
            [[0.0783, 0.0866, 0.0957, 0.0783, 0.0709, 0.0957, 0.1169, 0.1292, 0.1427, 0.1057],
             [0.0717, 0.0875, 0.0968, 0.1069, 0.1595, 0.1182, 0.0968, 0.0875, 0.0875, 0.0875],
             [0.1025, 0.1025, 0.0840, 0.1133, 0.1252, 0.1384, 0.0760, 0.0687, 0.1133, 0.0760]]
        )
        self.targets = np.array([7, 4, 5])
        self.true_pos = np.array([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
        self.false_pos = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.false_neg = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])

        # Binary classification
        self.scores_binary = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        self.targets_binary = np.array([0, 1, 1, 0, 1, 1, 1, 1])
        self.true_pos_binary = 4
        self.false_pos_binary = 1
        self.false_neg_binary = 2

        # Target class classification
        self.target_class = 4
        self.scores_target_class = self.scores.argmax(1) == self.target_class
        self.targets_target_class = self.targets == self.target_class
        self.true_pos_target_class = 1
        self.false_pos_target_class = 0
        self.false_neg_target_class = 0


class F1ScoreTest(ClassificationTest):
    def test_calculate(self):
        metric = F1Meter(num_classes=10, average='macro')
        tested_metric_result = metric.calculate(self.targets, self.scores)
        gt_sklearn_result = f1_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                     average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = F1Meter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_update_binary(self):
        metric = F1Meter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_binary)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_binary)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_binary)

    def test_update_target_class(self):
        metric = F1Meter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_target_class)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_target_class)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_target_class)

    def test_on_epoch_end_macro(self):
        metric = F1Meter(num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = f1_score(self.targets, self.scores.argmax(1),
                                     average='macro', labels=np.arange(10))
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_macro(self):
        metric = F1Meter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = f1_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                     average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_micro(self):
        metric = F1Meter(num_classes=10, average='micro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = f1_score(self.targets, self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_micro(self):
        metric = F1Meter(num_classes=10, average='micro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = f1_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_weighted(self):
        metric = F1Meter(num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = f1_score(self.targets, self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_weighted(self):
        metric = F1Meter(num_classes=10, average='weighted')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = f1_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_binary(self):
        metric = F1Meter(average='binary')
        metric.true_pos = self.true_pos_binary
        metric.false_neg = self.false_neg_binary
        metric.false_pos = self.false_pos_binary
        gt_sklearn_result = f1_score(self.targets_binary, self.scores_binary, average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_binary(self):
        metric = F1Meter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = f1_score(y_true=self.targets_binary, y_pred=self.scores_binary, average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_target_class(self):
        metric = F1Meter(num_classes=10, target_class=self.target_class, average='binary')
        metric.true_pos = self.true_pos_target_class
        metric.false_neg = self.false_neg_target_class
        metric.false_pos = self.false_pos_target_class
        gt_sklearn_result = f1_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                     average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_target_class(self):
        metric = F1Meter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = f1_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                     average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())


class FBetaScoreTest(ClassificationTest):
    def setUp(self) -> None:
        super().setUp()
        self.beta = 2.

    def test_calculate(self):
        metric = FbetaMeter(num_classes=10, average='macro', beta=self.beta)
        tested_metric_result = metric.calculate(self.targets, self.scores)
        gt_sklearn_result = fbeta_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                        beta=self.beta, average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = FbetaMeter(num_classes=10, average='macro', beta=self.beta)
        for i in range(3):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_update_binary(self):
        metric = FbetaMeter(average='binary', beta=self.beta)
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_binary)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_binary)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_binary)

    def test_update_target_class(self):
        metric = FbetaMeter(num_classes=10, target_class=self.target_class, average='binary', beta=self.beta)
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_target_class)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_target_class)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_target_class)

    def test_on_epoch_end_macro(self):
        metric = FbetaMeter(num_classes=10, average='macro', beta=self.beta)
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = fbeta_score(self.targets, self.scores.argmax(1),
                                        beta=self.beta, average='macro', labels=np.arange(10))
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_macro(self):
        metric = FbetaMeter(num_classes=10, average='macro', beta=self.beta)
        for i in range(3):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = fbeta_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                        beta=self.beta, average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_micro(self):
        metric = FbetaMeter(num_classes=10, average='micro', beta=self.beta)
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = fbeta_score(self.targets, self.scores.argmax(1), average='micro',
                                        beta=self.beta)
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_micro(self):
        metric = FbetaMeter(num_classes=10, average='micro', beta=self.beta)
        for i in range(3):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = fbeta_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='micro',
                                        beta=self.beta)
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_weighted(self):
        metric = FbetaMeter(num_classes=10, average='weighted', beta=self.beta)
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = fbeta_score(self.targets, self.scores.argmax(1), average='weighted',
                                        beta=self.beta)
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_weighted(self):
        metric = FbetaMeter(num_classes=10, average='weighted', beta=self.beta)
        for i in range(3):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = fbeta_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='weighted',
                                        beta=self.beta)
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_binary(self):
        metric = FbetaMeter(average='binary', beta=self.beta)
        metric.true_pos = self.true_pos_binary
        metric.false_neg = self.false_neg_binary
        metric.false_pos = self.false_pos_binary
        gt_sklearn_result = fbeta_score(self.targets_binary, self.scores_binary,
                                        average='binary', beta=self.beta)
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_binary(self):
        metric = FbetaMeter(average='binary', beta=self.beta)
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = fbeta_score(y_true=self.targets_binary, y_pred=self.scores_binary,
                                        average='binary', beta=self.beta)
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_target_class(self):
        metric = FbetaMeter(num_classes=10, target_class=self.target_class, average='binary', beta=self.beta)
        metric.true_pos = self.true_pos_target_class
        metric.false_neg = self.false_neg_target_class
        metric.false_pos = self.false_pos_target_class
        gt_sklearn_result = fbeta_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                        average='binary', beta=self.beta)
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_target_class(self):
        metric = FbetaMeter(num_classes=10, target_class=self.target_class, average='binary', beta=self.beta)
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = fbeta_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                        average='binary', beta=self.beta)
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())


class PrecisionTest(ClassificationTest):
    def test_calculate(self):
        metric = PrecisionMeter(num_classes=10, average='macro')
        tested_metric_result = metric.calculate(self.targets, self.scores)
        gt_sklearn_result = precision_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = PrecisionMeter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_update_binary(self):
        metric = PrecisionMeter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_binary)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_binary)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_binary)

    def test_update_target_class(self):
        metric = PrecisionMeter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_target_class)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_target_class)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_target_class)

    def test_on_epoch_end_macro(self):
        metric = PrecisionMeter(num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = precision_score(self.targets, self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_macro(self):
        metric = PrecisionMeter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = precision_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_micro(self):
        metric = PrecisionMeter(num_classes=10, average='micro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = precision_score(self.targets, self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_micro(self):
        metric = PrecisionMeter(num_classes=10, average='micro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = precision_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_weighted(self):
        metric = PrecisionMeter(num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = precision_score(self.targets, self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_weighted(self):
        metric = PrecisionMeter(num_classes=10, average='weighted')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = precision_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_binary(self):
        metric = PrecisionMeter(average='binary')
        metric.true_pos = self.true_pos_binary
        metric.false_neg = self.false_neg_binary
        metric.false_pos = self.false_pos_binary
        gt_sklearn_result = precision_score(self.targets_binary, self.scores_binary, average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_binary(self):
        metric = PrecisionMeter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = precision_score(y_true=self.targets_binary, y_pred=self.scores_binary, average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_target_class(self):
        metric = PrecisionMeter(num_classes=10, target_class=self.target_class, average='binary')
        metric.true_pos = self.true_pos_target_class
        metric.false_neg = self.false_neg_target_class
        metric.false_pos = self.false_pos_target_class
        gt_sklearn_result = precision_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                            average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_target_class(self):
        metric = PrecisionMeter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = precision_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                            average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())


class RecallTest(ClassificationTest):
    def test_calculate(self):
        metric = RecallMeter(num_classes=10, average='macro')
        tested_metric_result = metric.calculate(self.targets, self.scores)
        gt_sklearn_result = recall_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_update(self):
        metric = RecallMeter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg)

    def test_update_binary(self):
        metric = RecallMeter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_binary)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_binary)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_binary)

    def test_update_target_class(self):
        metric = RecallMeter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)
        np.testing.assert_almost_equal(metric.true_pos, self.true_pos_target_class)
        np.testing.assert_almost_equal(metric.false_pos, self.false_pos_target_class)
        np.testing.assert_almost_equal(metric.false_neg, self.false_neg_target_class)

    def test_on_epoch_end_macro(self):
        metric = RecallMeter(num_classes=10, average='macro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = recall_score(self.targets, self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_macro(self):
        metric = RecallMeter(num_classes=10, average='macro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = recall_score(y_true=self.targets, y_pred=self.scores.argmax(1),
                                            average='macro', labels=np.arange(10))
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_micro(self):
        metric = RecallMeter(num_classes=10, average='micro')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = recall_score(self.targets, self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_micro(self):
        metric = RecallMeter(num_classes=10, average='micro')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = recall_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='micro')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_weighted(self):
        metric = RecallMeter(num_classes=10, average='weighted')
        metric.true_pos = self.true_pos
        metric.false_neg = self.false_neg
        metric.false_pos = self.false_pos
        gt_sklearn_result = recall_score(self.targets, self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_weighted(self):
        metric = RecallMeter(num_classes=10, average='weighted')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = recall_score(y_true=self.targets, y_pred=self.scores.argmax(1), average='weighted')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_binary(self):
        metric = RecallMeter(average='binary')
        metric.true_pos = self.true_pos_binary
        metric.false_neg = self.false_neg_binary
        metric.false_pos = self.false_pos_binary
        gt_sklearn_result = recall_score(self.targets_binary, self.scores_binary, average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_binary(self):
        metric = RecallMeter(average='binary')
        for i in range(len(self.targets_binary)):
            y_pred = self.scores_binary[None, i]
            y_true = self.targets_binary[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = recall_score(y_true=self.targets_binary, y_pred=self.scores_binary, average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())

    def test_on_epoch_end_target_class(self):
        metric = RecallMeter(num_classes=10, target_class=self.target_class, average='binary')
        metric.true_pos = self.true_pos_target_class
        metric.false_neg = self.false_neg_target_class
        metric.false_pos = self.false_pos_target_class
        gt_sklearn_result = recall_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                            average='binary')
        self.assertAlmostEqual(metric.on_epoch_end(), gt_sklearn_result)

    def test_update_on_epoch_end_target_class(self):
        metric = RecallMeter(num_classes=10, target_class=self.target_class, average='binary')
        for i in range(len(self.targets)):
            y_pred = self.scores[None, i]
            y_true = self.targets[None, i]
            metric.update(y_true, y_pred)

        tested_metric_result = metric.on_epoch_end()
        gt_sklearn_result = recall_score(y_true=self.targets_target_class, y_pred=self.scores_target_class,
                                            average='binary')
        self.assertAlmostEqual(tested_metric_result.item(), gt_sklearn_result.item())


class MeanIntersectionOverUnionTests(unittest.TestCase):
    def setUp(self):
        self._predictions = torch.load('./tests/metrics/test_data/predictions_4classes.pt')
        self._target = torch.load('./tests/metrics/test_data/target_4classes.pt')

    def test_miou_averaged_compared_to_straight_calculations(self):
        preds = self._predictions.argmax(1).squeeze()
        # Calculate ious for each of two classes
        # (actually there are 4 classes, but they don't have predictions and targets in this example)
        tp0 = int(((preds == 0) & (self._target == 0)).sum())
        union0 = int(((self._target == 0) | (preds == 0)).sum())
        tp1 = int(((preds == 1) & (self._target == 1)).sum())
        union1 = int(((self._target == 1) | (preds == 1)).sum())

        miou_expected = np.mean([tp1 / union1, tp0 / union0])

        miou_meter = MeanIntersectionOverUnionMeter(num_classes=4, weighted=False)
        miou_meter.update(self._target, self._predictions)
        miou_actual = miou_meter.on_epoch_end()

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

        miou_meter = MeanIntersectionOverUnionMeter(num_classes=4, weighted=True)
        miou_meter.update(self._target, self._predictions)
        miou_actual = miou_meter.on_epoch_end()

        self.assertAlmostEqual(miou_expected, miou_actual,
                               msg='MIOU calculated from confusion matrix must match'
                                   'the MIOU calculated straightly')


class EERMeterTest(unittest.TestCase):

    def setUp(self) -> None:
        # shape of descs is (13233, 512)
        # shape of triplets is (242257, 3)
        descs = np.load('tests/metrics/test_data/descriptors.npy')
        self.triplets = np.load('tests/metrics/test_data/descriptors_triplets.npy')
        self.descs = torch.from_numpy(descs)
        t1, t2, t3 = self.triplets.T
        anchors, positives, negatives = self.descs[t1], self.descs[t2], self.descs[t3]

        eer_meter_c = EERMeter(distance='cosine')
        eer_meter_c.update(anchor=anchors, positive=positives, negative=negatives)

        eer_meter_e = EERMeter(distance='euclidean')
        eer_meter_e.update(anchor=anchors, positive=positives, negative=negatives)

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
        distances, labels = self.eer_meter['cosine'].distances, self.eer_meter['cosine'].labels
        res_eer, res_threshold = self.eer_meter['cosine'].calculate_eer()

        calculate_conf = self.functions[0]

        conf = calculate_conf(res_threshold, distances, labels)
        self.assertAlmostEqual(far(conf), frr(conf), places=5)

    def test_eer_calculator_euclidean(self):
        distances, labels = self.eer_meter['cosine'].distances, self.eer_meter['cosine'].labels
        res_eer, res_threshold = self.eer_meter['euclidean'].calculate_eer()

        calculate_conf = self.functions[0]

        conf = calculate_conf(res_threshold, distances, labels)
        self.assertAlmostEqual(far(conf), frr(conf), places=4)

    def test_adder_differ_batch(self):
        eer_meter = EERMeter(distance='cosine')
        for triplet in self.triplets:
            eer_meter.update(
                anchor=torch.FloatTensor(self.descs[triplet[0]]).unsqueeze(0),
                positive=torch.FloatTensor(self.descs[triplet[1]]).unsqueeze(0),
                negative=torch.FloatTensor(self.descs[triplet[2]]).unsqueeze(0)
            )
        eer, thresh = eer_meter.calculate_eer()
        eer_batch, thresh_batch = self.eer_meter['cosine'].calculate_eer()

        self.assertAlmostEqual(eer, eer_batch)
        self.assertAlmostEqual(thresh, thresh_batch)


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
