import numpy as np
from sklearn.metrics import roc_curve, auc

# def calculate_roc(ground_truth, prediction)

def evaluate_binary(ground_truth, prediction_scores):
    """
    Evaluate roc curve in binary classification
    :param ground_truth: 1-d array, e.g. gt: [1, 0, 1, 0, 1], note: 0 or 1
    :param prediction_scores: 1-d array, probability for positive example,
                              e.g. pred: [0.3, 0.4, 0.1, 0.3]
    :return:  false_positive_rate, true_positive_rate, thresholds, roc_auc
    """

    false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth, prediction_scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    return false_positive_rate, true_positive_rate, thresholds, roc_auc


def evaluate_multiple(ground_truth, prediction_scores):
    """

    :param ground_truth: 1-d array annotated with class labels start from 0, e.g. gt: [0, 0, 1, 3, 2, 1, 0]
    :param prediction_scores: 2-d array recorded the corresponding probability scores for each class
    :return: 1-d arrays with number of class: false_positive_rates, true_positive_rates, thresholds, roc_aucs

    """

    # Check dimension
    if len(prediction_scores.shape) != 2:
        print 'The dimension of \'prediction_scores\' should be 2.'
        return

    N = prediction_scores.shape[0]
    M = prediction_scores.shape[1]

    false_positive_rates = []
    true_positive_rates = []
    thresholds = []
    roc_aucs = []

    for class_label in range(0, M):

        # Generate Class Label
        ground_truth_label = np.zeros(N, dtype=int)
        idx = (ground_truth == class_label)
        ground_truth_label[idx] = 1

        # Extract positive scores
        prediction_score = prediction_scores[:, class_label]

        # Compute ROC curve
        false_positive_rate, true_positive_rate, threshold = roc_curve(ground_truth, prediction_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        false_positive_rates.append(false_positive_rate)
        true_positive_rates.append(true_positive_rate)
        thresholds.append(threshold)
        roc_aucs.append(roc_auc)

        false_positive_rates = np.asarray(false_positive_rates)
        true_positive_rates = np.asarray(true_positive_rates)
        thresholds = np.asarray(thresholds)
        roc_aucs = np.asarray(roc_aucs)

    return false_positive_rates, true_positive_rates, thresholds, roc_aucs


# def compute_multiple_micro_macro_avg(ground_truth, prediction_scores):
#
#     # Check dimension
#     if len(prediction_scores.shape) != 2:
#         print 'The dimension of \'prediction_scores\' should be 2.'
#         return
#
#     N = prediction_scores.shape[0]
#     M = prediction_scores.shape[1]
#
#     gt_label_array = []
#     prediction_score_array = []
#
#     for class_label in range(0, M):
#
#         # Generate Class Label
#         ground_truth_label = np.zeros(N, dtype=int)
#         idx = (ground_truth == class_label)
#         ground_truth_label[idx] = 1
#
#         # Extract positive scores
#         prediction_score = prediction_scores[:, class_label]
#
#         gt_label_array.append(ground_truth_label)
#         prediction_score_array.append(prediction_score)
#
#     gt_label_array = np.asarray(gt_label_array)
#     prediction_score_array = np.asarray(prediction_score_array)
#
#     false_positive_rates = {}
#     true_positive_rates = {}
#     roc_auc = {}
#
#     # Compute Micro Avg.
#     false_positive_rates["micro"], true_positive_rates["micro"], _ = roc_curve(gt_label_array.ravel(),
#                                                                                prediction_score_array.ravel())
#     roc_auc["micro"] = auc(false_positive_rates["micro"], true_positive_rates["micro"])
#
#     # Compute Macro Avg.
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([false_positive_rates[i] for i in range(n_classes)]))
