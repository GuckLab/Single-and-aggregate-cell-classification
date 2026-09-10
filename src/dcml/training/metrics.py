from sklearn import metrics
import numpy as np
from collections import OrderedDict
from src.dcml.evaluation.helpers import get_key_ordered_values_from_dict
from typing import List
from tabulate import tabulate
from torch import Tensor
import torch


def balanced_accuracy(targets: Tensor, predictions: Tensor) -> np.ndarray:
    """
    Computes balanced accuracy for each class separately -
    mean between specificity and sensitivity for each binary problem.
    Undefined labels, which are values in targets that are neither 0.0 nor 1.0, are ignored.
    In case of no examples for either 0, 1, or both, for some of the binary problems, zero performance is returned for that problem

    Parameters
    ----------
    targets: ground truth, tensor of n x m shape with number of samples n, and number of classes m (binary problems)
    predictions: tensor of n x m shape with number of samples n, and number of classes m (binary problems)

    Returns
    -------
    balanced_accuracy - array of length m with balanced accuracy for each class
    """

    assert torch.is_floating_point(targets), "input targets must have float values"
    assert torch.is_floating_point(predictions), "input predictions must have float values"

    n_classes = targets.shape[1]

    cls_ba = np.full(n_classes, 0, dtype=float)

    for cls in range(n_classes):
        mask = (targets[:, cls] == 0)  | (targets[:, cls] == 1)
        targets_cls = targets[mask, cls]
        predictions_cls = predictions[mask, cls]

        # no labels for the class
        if len(targets_cls) == 0:
            continue

        # in case there are no examples for class 0, skip
        if (targets_cls == 1).sum() == 0:
            continue

        # in case there are no examples for case 1, skip
        if (targets_cls == 0).sum() == 0:
            continue

        # returns precision, recall, fscore, support
        cls_ba[cls] = metrics.balanced_accuracy_score(targets_cls, predictions_cls)

    return cls_ba

def multi_label_metrics_precision_recall_fscore_support(targets: Tensor, predictions: Tensor) -> tuple:
    """
    Allows computation of precision, recall, fscore, support like precision_recall_fscore_support from sklearn.metrics
    for multi label classification with the cases where samples for some classes may have undefined labels.
    Undefined labels are values in targets that are neither 0.0 nor 1.0.

    Parameters
    ----------
    targets: ground truth, tensor of n x m shape with number of samples n, and number of classes m
    predictions: tensor of n x m shape with number of samples n, and number of classes m

    Returns
    -------
    scores: tuple of size 4 with precision (array of size m with floats), recall (array of size m with floats), fscore (array of size m with floats),
    support values (array of size m with integers)
    """

    assert torch.is_floating_point(targets), "input targets must have float values"
    assert torch.is_floating_point(predictions), "input predictions must have float values"

    n_classes = targets.shape[1]

    init = np.nan
    scores = (
        np.full(n_classes, init, dtype=float), # precision
        np.full(n_classes, init, dtype=float), # recall
        np.full(n_classes, init, dtype=float), # fscore
        np.zeros(n_classes, dtype=int) # support
    )

    for cls in range(n_classes):
        mask = (targets[:, cls] == 0)  | (targets[:, cls] == 1)
        targets_cls = targets[mask, cls]
        predictions_cls = predictions[mask, cls]

        # no labels for the class
        if len(targets_cls) == 0:
            continue

        # returns precision, recall, fscore, support
        cls_scores = metrics.precision_recall_fscore_support(targets_cls, predictions_cls, average=None, labels=[0, 1],
                                                             zero_division=0)

        for n_metr in range(len(scores)):
            scores[n_metr][cls] = cls_scores[n_metr][1]

    return scores


def multi_label_metrics_average_precision_score(targets: Tensor, prediction_scores: Tensor) -> np.ndarray:
    """
    Allows computation of average_precision_score like average_precision_score from sklearn.metrics
    for multi label classification with the cases where samples for some classes may have undefined labels.
    Undefined labels are values in targets that are neither 0.0 nor 1.0.

    Parameters
    ----------
    targets: ground truth, tensor of n x m shape with number of samples n, and number of classes m
    prediction_scores: tensor of n x m shape with number of samples n, and number of classes m

    Returns
    -------
    average_precision: array of average precisions (floats, averaged over recall) for each class

    """

    assert torch.is_floating_point(targets), "input targets must have float values"
    assert torch.is_floating_point(prediction_scores), "input prediction_scores must have float values"

    n_classes = targets.shape[1]
    average_precision = np.full(n_classes, np.nan, dtype=float)

    for cls in range(n_classes):
        mask = (targets[:, cls] == 0)  | (targets[:, cls] == 1)
        targets_cls = targets[mask, cls]
        prediction_scores_cls = prediction_scores[mask, cls]

        # no labels for the class
        if len(targets_cls) == 0:
            continue

        average_precision[cls] = metrics.average_precision_score(targets_cls, prediction_scores_cls, average=None)

    return average_precision


def multi_label_metrics_roc_auc_score(targets: Tensor, prediction_scores: Tensor) -> np.ndarray:
    """
    Allows computation of roc_auc_score like roc_auc_score from sklearn.metrics for multi label classification with the
    cases where samples for some classes may have undefined labels. Undefined labels are values in targets that are
    neither 0.0 nor 1.0.

    Parameters
    ----------
    targets: ground truth, tensor of n x m shape with number of samples n, and number of classes m
    prediction_scores: tensor of n x m shape with number of samples n, and number of classes m

    Returns
    -------
    roc_auc_score: array of ROC scores (floats) for each class
    """

    assert torch.is_floating_point(targets), "input targets must have float values"
    assert torch.is_floating_point(prediction_scores), "input prediction_scores must have float values"

    n_classes = targets.shape[1]
    roc_auc_score = np.full(n_classes, np.nan, dtype=float)

    for cls in range(n_classes):
        mask = (targets[:, cls] == 0)  | (targets[:, cls] == 1)
        targets_cls = targets[mask, cls]
        prediction_scores_cls = prediction_scores[mask, cls]

        # no labels for the class
        if len(targets_cls) == 0:
            continue

        roc_auc_score[cls] = metrics.roc_auc_score(targets_cls, prediction_scores_cls, average=None)

    return roc_auc_score


class EvaluationMetrics:

    def __init__(self, target_names: dict[int, str], bio_weights: dict[str, int] = None,
                 beta_weights: dict[str, int] = None, used_metrics: List[str] = None, cls_type: str=""):
        """
        Metrics to be evaluated

        Parameters
        ----------
        target_names: class names as values of the dictionary and the corresponding integer values as keys of the dictionary

        bio_weights: weights are the values of the dictionary, which are used for creation the weighted average
        performance from performances per class. Class names are in the keys of the dictionary

        beta_weights: weights are the values of the dictionary, which are used for weighting between recall and
        precision when computing Fb score. Class names are in the keys of the dictionary

        used_metrics: names of the metrics to be computed and returned upon request - "recall", "f1", "precision", "bal_acc" (these
        are averages and computed anyway), "fb_weighted_beta",
        "weighted_recall", "weighted_f1", "weighted_precision", "weighted_fb_weighted_beta" (computed on demand).
        By default, all available  metrics will be computed.

        cls_type: either "MTL" (multi-task) or "MC" (Multi-class)

        Example:
        y_true = [0, 1, 2, 2, 2]
        y_pred = [0, 0, 2, 2, 1]
        av_weights = {'A': 1, 'B': 1, 'C': 2}
        beta_weights = {'A': 1, 'B': 1, 'C': 0.5}
        target_names = {0: 'A', 1: 'B', 2: 'C' }

        mm = EvaluationMetrics(target_names=target_names, bio_weights=av_weights, beta_weights=beta_weights)
        mm(targets=y_true, predictions=y_pred)
        print(mm.weighted_f1)

        """

        self.bio_weights = {}
        self.beta_weights ={}

        if bio_weights is not None:
            for tn_key, tn_value in target_names.items():
                self.bio_weights[tn_key] = bio_weights[tn_value]

            dict_vals, _ = get_key_ordered_values_from_dict(self.bio_weights)
            self.bio_weights = np.array(dict_vals)
        else:
            self.bio_weights = None

        if beta_weights is not None:
            for tn_key, tn_value in target_names.items():
                self.beta_weights[tn_key] = beta_weights[tn_value]

            dict_vals, _ = get_key_ordered_values_from_dict(self.beta_weights)
            self.beta_weights = np.array(dict_vals)
        else:
            self.beta_weights = None

        self.target_names, self.target_int_labels = get_key_ordered_values_from_dict(target_names)

        self.recall = None
        self.f1 = None
        self.precision = None
        self.bal_acc = None
        self.roc_auc_score = None
        self.average_precision_score = None

        self.best_precision = -np.inf
        self.best_recall = -np.inf
        self.best_f1 = -np.inf
        self.best_bal_acc = -np.inf
        self.best_fb_weighted_beta = -np.inf
        self.best_roc_auc_score = -np.inf
        self.best_average_precision_score = -np.inf

        self.best_weighted_precision = -np.inf
        self.best_weighted_recall = -np.inf
        self.best_weighted_f1 = -np.inf
        self.best_weighted_bal_acc = -np.inf
        self.best_weighted_fb_weighted_beta = -np.inf
        self.best_weighted_roc_auc_score = -np.inf
        self.best_weighted_average_precision_score = -np.inf

        self.precision_per_class = None
        self.recall_per_class = None
        self.f1_per_class = None
        self.bal_acc_per_class = None
        self.support_per_class = None
        self.roc_auc_score_per_class = None
        self.average_precision_score_per_class = None

        self._fb_weighted_beta_per_class = None
        self._fb_weighted_beta = None
        self._weighted_recall = None
        self._weighted_precision = None
        self._weighted_f1 = None
        self._weighted_bal_acc = None
        self._weighted_fb_weighted_beta = None
        self._weighted_roc_auc_score = None
        self._weighted_average_precision_score = None

        self._report = None
        self._confusion_matrix = None
        self._confusion_matrix_labels = None

        self.targets = None
        self.predictions = None

        self.wbc_ind = None
        self.aggr_index = None
        self.cls_type = "MC"
        if ('wbc' in self.target_names and not cls_type) or (cls_type=="MTL"):  # MTL case

            self.cls_type = "MTL"

            try:
                self.wbc_ind = self.target_names.index('wbc')
                self.aggr_index = self.target_names.index('Aggregation')
            except ValueError:
                pass


        # summary measures
        self._basic_metric_names = ("recall", "f1", "precision", "bal_acc", "roc_auc_score", "average_precision_score")
        self._derived_metric_names = ("fb_weighted_beta", "weighted_recall", "weighted_f1", "weighted_precision", "weighted_bal_acc"
                                      "weighted_fb_weighted_beta", "weighted_roc_auc_score",
                                      "weighted_average_precision_score")

        self._mapping_report_column = {"precision": "precision", "weighted_precision": "precision",
                                       "recall": "recall", "weighted_recall": "recall",
                                       "f1": "f1", "weighted_f1": "f1",
                                       "bal_acc": "bal_acc", "weighted_bal_acc": "bal_acc",
                                       "fb_weighted_beta": "fb_weighted_beta",
                                       "weighted_fb_weighted_beta": "fb_weighted_beta",
                                       "roc_auc_score": "roc_auc_score", "weighted_roc_auc_score": "roc_auc_score",
                                       "average_precision_score": "average_precision_score",
                                       "weighted_average_precision_score": "average_precision_score"}

        if used_metrics is None:
            self.requested_metric_names = self._basic_metric_names + self._derived_metric_names
        else:
            assert set(used_metrics) <= set(self._basic_metric_names + self._derived_metric_names)
            self.requested_metric_names = used_metrics

        # check that no weighted metrics were requested if weights were not given
        if self.bio_weights is None:
            for requested_metric_name in self.requested_metric_names:
                assert not requested_metric_name.startswith("weighted_")

        self.requested_columns = []
        for r in self.requested_metric_names:
            self.requested_columns.append(self._mapping_report_column[r])

        self.requested_columns = list(dict.fromkeys(self.requested_columns))

    def __call__(self, targets: Tensor, predictions: Tensor, prediction_scores: Tensor = None):

        # Compute precision, recall, f1 score, support for each class and their averages. Memorize their best values.
        # 'targets' and 'predictions' are tensors that can can either be 1D indices for true classes or sparse 2D matrix
        # of indicators (ones, all other are zeros) for true classes. Number of columns of the tensor is the number
        # of classes

        if self.cls_type == 'MTL':
            assert targets.dim() == 2, "multi-class targets for multi-label task"
        else:
            assert targets.dim() == 1, "multi-label targets for multi-class task"

        # handle the case with WBC that is of unknown type (e.g.). These cases should be excluded from evaluation.
        if self.wbc_ind is not None: # MTL case

            # only a single WBC type should have 1, others should have 0. If this is not the case then set for all WBC
            # types a value that is neither 0 not 1. Such undefined values will be later ignored in metric evaluation
            # functions.

            assert self.cls_type == 'MTL'
            # for each object with WBC, MC labels must be either all nan or have a single one among all other zeros
            wbc_indexes = targets[:, self.wbc_ind] == 1
            not_wbc_indexes = targets[:, self.wbc_ind] == 0
            unknown_wbc_indexes = torch.isnan(targets[:, self.wbc_ind])
            assert torch.all(
                              torch.all(torch.isnan(targets[wbc_indexes, (self.wbc_ind + 1):]), dim=1) |
                              torch.sum(targets[wbc_indexes, (self.wbc_ind + 1):] == 1, dim=1)
            ), "warning: for each object with WBC, MC labels must be either all nan or have a single one among all other zeros"

            # for each object without WBC, all MC labels must be zeros or all must ne nan
            assert torch.all(
                              torch.all(torch.isnan(targets[not_wbc_indexes, (self.wbc_ind + 1):]), dim=1) |
                              torch.all(targets[not_wbc_indexes, (self.wbc_ind + 1):] == 0, dim=1)
            ), "warning: for each object without WBC, all MC labels must be zeros or all must ne nan"

            # for each object without unknown WBC, all MC labels must be nans
            assert torch.all(torch.isnan(targets[unknown_wbc_indexes, (self.wbc_ind + 1):])), "warning: unknown WBC type should have all MC labels as nan"

            assert targets.shape[1] == len(self.target_names)


        self.targets = targets
        self.predictions = predictions
        self.prediction_scores = prediction_scores

        # outputs the tuple (precision, recall, fbeta_score, support) of lists, each of which contains
        # the corresponding metric for unique labels (taken from targets and predictions)
        if self.cls_type == 'MTL':
            scores = multi_label_metrics_precision_recall_fscore_support(self.targets, self.predictions)
            bal_acc_scores = balanced_accuracy(self.targets, self.predictions)
        else:
            scores = metrics.precision_recall_fscore_support(self.targets, self.predictions)
            ml_targets, ml_predictions = self._multiclass_to_multilabel(self.targets, self.predictions)
            bal_acc_scores = balanced_accuracy(ml_targets, ml_predictions)

            # gives exactly the same results for MC as when metrics.precision_recall_fscore_support(self.targets, self.predictions) is used
            # scores = multi_label_metrics_precision_recall_fscore_support(ml_targets, ml_predictions)



        assert len(scores[0]) == len(self.target_names), ("not proper number of classes, probably batch does "
                                                          "not contain indexes for all the classes")

        if prediction_scores is not None:

            if self.cls_type == 'MTL':
                self.roc_auc_score_per_class = multi_label_metrics_roc_auc_score(self.targets, self.prediction_scores)
                self.average_precision_score_per_class = multi_label_metrics_average_precision_score(self.targets,
                                                                                         self.prediction_scores)
            else:

                # multi_class="ovr" is only relevant for multi-class not multi-label
                self.roc_auc_score_per_class = metrics.roc_auc_score(self.targets, self.prediction_scores, average=None, multi_class="ovr")
                self.average_precision_score_per_class = metrics.average_precision_score(self.targets, self.prediction_scores, average=None)

            self.roc_auc_score = np.mean(self.roc_auc_score_per_class)
            self.average_precision_score = np.mean(self.average_precision_score_per_class)

            if self.roc_auc_score > self.best_roc_auc_score:
                self.best_roc_auc_score = self.roc_auc_score

            if self.average_precision_score > self.best_average_precision_score:
                self.best_average_precision_score = self.average_precision_score

        self.precision_per_class = scores[0]
        self.recall_per_class = scores[1]
        self.f1_per_class = scores[2]
        self.bal_acc_per_class = bal_acc_scores
        self.support_per_class = scores[3]
        self._fb_weighted_beta_per_class = None

        self.recall = np.mean(self.recall_per_class)
        self.f1 = np.mean(self.f1_per_class)  # mean of f beta
        self.precision = np.mean(self.precision_per_class)
        self.bal_acc = np.mean(self.bal_acc_per_class)
        self._fb_weighted_beta = None

        # weighted metrics
        self._weighted_recall = None
        self._weighted_precision = None
        self._weighted_f1 = None
        self._weighted_bal_acc = None
        self._weighted_fb_weighted_beta = None
        self._weighted_roc_auc_score = None
        self._weighted_average_precision_score = None

        self._report = None
        self._confusion_matrix = None

        if self.recall > self.best_recall:
            self.best_recall = self.recall

        if self.precision > self.best_precision:
            self.best_precision = self.precision

        if self.f1 > self.best_f1:
            self.best_f1 = self.f1

        if self.bal_acc > self.best_bal_acc:
            self.best_bal_acc = self.bal_acc

        return self.recall

    @property
    def fb_weighted_beta_per_class(self):

        assert self.beta_weights is not None, "f beta weights were not provided"
        assert self.precision_per_class is not None, "precison-recall scores were not yet computed with CLS(targets, predictions)"
        assert self.recall_per_class is not None, "precision-recall scores were not yet computed with CLS(targets, predictions)"

        if self._fb_weighted_beta_per_class is None:
            denominator = (self.beta_weights ** 2) * self.precision_per_class + self.recall_per_class
            ind_0 = denominator == 0
            denominator[ind_0] = 1
            self._fb_weighted_beta_per_class = (1 + self.beta_weights ** 2) * self.precision_per_class * self.recall_per_class / denominator
            self._fb_weighted_beta_per_class[ind_0] = 0

        return self._fb_weighted_beta_per_class

    @property
    def fb_weighted_beta(self):

        if self._fb_weighted_beta is None:
            if self._fb_weighted_beta_per_class is None:
                self._fb_weighted_beta = np.mean(self.fb_weighted_beta_per_class)
            else:
                self._fb_weighted_beta = np.mean(self._fb_weighted_beta_per_class)

            if self._fb_weighted_beta > self.best_fb_weighted_beta:
                self.best_fb_weighted_beta = self._fb_weighted_beta

        return self._fb_weighted_beta

    @property
    def available_metric_names(self) -> List:
        "Returns metric names that were already computed"

        names = list(self._basic_metric_names)
        for name in self._derived_metric_names:
            attr = '_' + name
            if getattr(self, attr) is not None:
                names.append(name)

        return names

    @property
    def weighted_recall(self):

        assert self.recall_per_class is not None, "scores were not yet computed with CLS(targets, predictions)"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_recall is None:
            self._weighted_recall = np.average(self.recall_per_class, weights=self.bio_weights)

            if self._weighted_recall > self.best_weighted_recall:
                self.best_weighted_recall = self._weighted_recall

        return self._weighted_recall

    @property
    def weighted_precision(self):

        assert self.precision_per_class is not None, "scores were not yet computed with CLS(targets, predictions)"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_precision is None:
            self._weighted_precision = np.average(self.precision_per_class, weights=self.bio_weights)

            if self._weighted_precision > self.best_weighted_precision:
                self.best_weighted_precision = self._weighted_precision

        return self._weighted_precision

    @property
    def weighted_f1(self):

        assert self.f1_per_class is not None, "scores were not yet computed with CLS(targets, predictions)"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_f1 is None:
            self._weighted_f1 = np.average(self.f1_per_class, weights=self.bio_weights)

            if self._weighted_f1 > self.best_weighted_f1:
                self.best_weighted_f1 = self._weighted_f1

        return self._weighted_f1

    @property
    def weighted_bal_acc(self):

        assert self.bal_acc_per_class is not None, "scores were not yet computed with CLS(targets, predictions)"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_bal_acc is None:
            self._weighted_bal_acc = np.average(self.bal_acc_per_class, weights=self.bio_weights)

            if self._weighted_bal_acc > self.best_weighted_bal_acc:
                self.best_weighted_bal_acc = self._weighted_bal_acc

        return self._weighted_bal_acc

    @property
    def weighted_average_precision_score(self):

        assert self.average_precision_score_per_class is not None, "average_precision_scores were not yet computed"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_average_precision_score is None:
            self._weighted_average_precision_score = np.average(self.average_precision_score_per_class, weights=self.bio_weights)

            if self._weighted_average_precision_score > self.best_weighted_average_precision_score:
                self.best_weighted_average_precision_score = self._weighted_average_precision_score

        return self._weighted_average_precision_score



    @property
    def weighted_roc_auc_score(self):

        assert self.roc_auc_score_per_class is not None, "roc_auc_scores were not yet computed"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_roc_auc_score is None:
            self._weighted_roc_auc_score = np.average(self.roc_auc_score_per_class, weights=self.bio_weights)

            if self._weighted_roc_auc_score > self.best_weighted_roc_auc_score:
                self.best_weighted_roc_auc_score = self._weighted_roc_auc_score

        return self._weighted_roc_auc_score

    @property
    def weighted_fb_weighted_beta(self):

        assert self.beta_weights is not None, "f beta weights were not provided"
        assert self.bio_weights is not None, "bio weights were not provided"

        if self._weighted_fb_weighted_beta is None:
            if self._fb_weighted_beta_per_class is None:
                fb_weighted_beta = self.fb_weighted_beta_per_class
            else:
                fb_weighted_beta = self._fb_weighted_beta_per_class

            self._weighted_fb_weighted_beta = np.average(fb_weighted_beta, weights=self.bio_weights)

            if self._weighted_fb_weighted_beta > self.best_weighted_fb_weighted_beta:
                self.best_weighted_fb_weighted_beta = self._weighted_fb_weighted_beta

        return self._weighted_fb_weighted_beta

    @property
    def report(self) -> dict:
        """ Outputs report with performance measures"""

        if self._report is None:

            self._report = self._classification_report()

        return self._report


    @property
    def confusion_matrix_labels(self):

        if self._confusion_matrix is None:

            self.confusion_matrix() # calculates also self._confusion_matrix_labels

        assert self._confusion_matrix_labels is not None

        return self._confusion_matrix_labels


    @property
    def confusion_matrix(self):

        if self._confusion_matrix is None:
            if self.cls_type == "MTL": # MTL case

                self._confusion_matrix_labels = self.target_names

                self._confusion_matrix = self._compute_generalized_confusion_matrix()


            else:
                targets = self.targets
                predictions = self.predictions
                self._confusion_matrix_labels = self.target_names
                confusion_matrix_int_labels = self.target_int_labels

                if targets.ndim == 2:

                    assert predictions.ndim ==2

                    # convert matrix labels to indexes 1D array
                    predictions = torch.argmax(predictions, dim=1)
                    targets = torch.argmax(targets, dim=1)

                self._confusion_matrix = self._compute_confusion_matrix(targets, predictions, confusion_matrix_int_labels)


        return self._confusion_matrix


    def _single_cell_labels_conf_mat(self):

        # for MTL case only, used for plotting confusion matrix

        # add unknown cell type label
        confusion_matrix_labels = self.target_names.copy()
        confusion_matrix_labels.append("Unknown type")

        # remove aggregation and wbc labels
        if self.wbc_ind <  self.aggr_index:
            del confusion_matrix_labels[self.aggr_index]
            del confusion_matrix_labels[self.wbc_ind]

        else:
            del confusion_matrix_labels[self.wbc_ind]
            del confusion_matrix_labels[self.aggr_index]

        # add unknown cell type integer label
        confusion_matrix_int_labels = list(np.arange(len(confusion_matrix_labels)))

        return confusion_matrix_labels, confusion_matrix_int_labels


    def _compute_generalized_confusion_matrix(self) -> Tensor:
        """
        Computes kind of confusion matrix with rows as observations and columns as classes. In contrast to standard
        confusion matrix, here each row is not restricted to have only one detection. Output is a 2D tensor of float
        numbers.

        -------

        """

        n_samples, n_classes = self.targets.shape
        cfg = np.zeros((n_classes, n_classes))
        for sample_ind in range(n_samples):
            cfg_current = np.zeros((n_classes, n_classes))
            for class_ind in range(n_classes):
                if self.targets[sample_ind, class_ind] == 1:
                    cfg_current[class_ind, :] = self.predictions[sample_ind, :]

            cfg = cfg + cfg_current

        # normalize generalized confusion matrix. Each row is divided by the number of targets for the corresponding
        # to the row class
        supports = torch.sum(self.targets == 1, dim=0, keepdim=True)
        cfg = cfg / supports.T

        return cfg


    # TODO can be static
    def _compute_confusion_matrix(self, targets, predictions, int_labels) -> np.ndarray:

        assert targets is not None, "scores were not yet computed with CLS(targets, predictions)"
        assert predictions is not None, "scores were not yet computed with CLS(targets, predictions)"

        conf_mat = metrics.confusion_matrix(targets, predictions, labels=int_labels)

        # normalize confusion matrix (to row sums equal one)
        norm_factors = conf_mat.sum(axis=1)[:, np.newaxis]
        norm_factors[norm_factors == 0] = 1 # avoid division by zero (no true data for a category)
        conf_mat_normalized = conf_mat.astype('float') / norm_factors

        return conf_mat_normalized

    def print_metrics(self):
        """Prints on the screen computed metrics per class"""

        headers = ["Metrics"] + self.target_names
        tab = []
        for met in self.requested_columns:  # here it will be printed as rows
            tab.append([met] + list(getattr(self, met + '_per_class')))

        print(tabulate(tab, headers))

    # adapted from on scikitlearn
    def _classification_report(self) -> dict:
        """Outputs report with performance measures"""

        # define rows of the report
        if self.bio_weights is not None:
            target_names = self.target_names + ["avg", "bio_weighted_avg"]
        else:
            target_names = self.target_names + ["avg"]

        # define columns
        headers = OrderedDict()
        columns = []

        for m in self.requested_columns:
            headers[m] = getattr(self, m + "_per_class")

            if self.bio_weights is not None:
                headers[m] = np.append(headers[m], [getattr(self, m), getattr(self, "weighted_" + m)])
            else:
                headers[m] = np.append(headers[m], [getattr(self, m)])

            columns.append(headers[m])

        assert self.support_per_class is not None, "scores were not yet computed with CLS(targets, predictions)"
        headers["support"] = self.support_per_class
        if self.bio_weights is not None:
            headers["support"] = np.append(headers["support"], [np.nan, np.nan])
        else:
            headers["support"] = np.append(headers["support"], [np.nan])

        columns.append(headers["support"])

        rows = zip(target_names, *columns)

        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers.keys(), [float(i) for i in scores]))

        return report_dict


    def _multiclass_to_multilabel(self, targets: Tensor, predictions: Tensor):

        assert min(targets) >= 0, "minimal label must be zero a above"
        assert min(predictions) >= 0, "minimal label must be zero a above"

        assert len(targets) == len(predictions)

        n_samples = len(targets)
        n_classes = len(self.target_int_labels)

        targets_ml = torch.zeros((n_samples, n_classes), dtype=torch.float)
        predictions_ml = torch.zeros((n_samples, n_classes), dtype=torch.float)

        for n_sample, (tar_lab, pred_lab) in enumerate(zip(targets, predictions)):
            targets_ml[n_sample, tar_lab] = 1
            predictions_ml[n_sample, pred_lab] = 1

        return targets_ml, predictions_ml

