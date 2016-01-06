import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

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


def plot_roc_curve_binary(false_positive_rate, true_positive_rate, roc_auc, plot_title=None):

    if plot_title is None:
        plot_title = 'ROC Curves'

    # Init matplotlib
    fig = plt.figure()

    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = {0:0.2f})'
                                                                        ''.format(roc_auc))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc="lower right")

    return fig


def draw_roc_curve_binary(ground_truth, prediction_score, plot_title=None):

    # Compute Roc values
    false_positive_rate, true_positive_rate, threshold, roc_auc = \
        evaluate_binary(ground_truth, prediction_score)

    # Plot ROC curve
    fig = plot_roc_curve_binary(false_positive_rate, true_positive_rate, roc_auc, plot_title)

    plt.show(fig)


def evaluate_multiple(ground_truths, prediction_scores, compute_micro_macro_avg=False):
    """

    :param ground_truths: 1-d array annotated with class labels start from 0, e.g. gt: [0, 0, 1, 3, 2, 1, 0]
    :param prediction_scores: 2-d array recorded the corresponding probability scores for each class
    :param compute_micro_macro_avg: switch if the micro and macro average roc are needed
    :return: Dictory with number of class: false_positive_rates, true_positive_rates, thresholds, roc_aucs

    """

    # Check dimension
    if len(prediction_scores.shape) != 2:
        print 'The dimension of \'prediction_scores\' should be 2.'
        return

    N = prediction_scores.shape[0]
    M = prediction_scores.shape[1]

    false_positive_rates = {}
    true_positive_rates = {}
    thresholds = {}
    roc_aucs = {}

    if compute_micro_macro_avg:
        gt_label_array = []
        prediction_score_array = []

    for class_label in range(0, M):

        # Generate Class Label
        ground_truth_label = np.zeros(N, dtype=int)
        idx = (ground_truths == class_label)
        ground_truth_label[idx] = 1

        # Extract positive scores
        prediction_score = prediction_scores[:, class_label]

        # Compute ROC curve
        false_positive_rate, true_positive_rate, threshold = roc_curve(ground_truth_label, prediction_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        false_positive_rates[class_label] = false_positive_rate
        true_positive_rates[class_label] = true_positive_rate
        thresholds[class_label] = threshold
        roc_aucs[class_label] = roc_auc

        if compute_micro_macro_avg:
            gt_label_array.append(ground_truth_label)
            prediction_score_array.append(prediction_score)

    if compute_micro_macro_avg:
        gt_label_array = np.asarray(gt_label_array)
        prediction_score_array = np.asarray(prediction_score_array)

        # Compute Micro Avg.
        false_positive_rates["micro"], true_positive_rates["micro"], _ = roc_curve(gt_label_array.ravel(),
                                                                                   prediction_score_array.ravel())
        roc_aucs["micro"] = auc(false_positive_rates["micro"], true_positive_rates["micro"])

        # Compute Macro Avg.
        all_fpr = np.unique(np.concatenate([false_positive_rates[i] for i in range(M)]))

        # Interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(M):
            mean_tpr += interp(all_fpr, false_positive_rates[i], true_positive_rates[i])

        # Finally average it and compute AUC
        mean_tpr /= M

        false_positive_rates["macro"] = all_fpr
        true_positive_rates["macro"] = mean_tpr
        roc_aucs["macro"] = auc(false_positive_rates["macro"], true_positive_rates["macro"])

    return false_positive_rates, true_positive_rates, thresholds, roc_aucs


def plot_roc_curve_multiple(false_positive_rates, true_positive_rates, roc_aucs, plot_title=None):

    if plot_title is None:
        plot_title = 'ROC Curves'

    # Determine how many classes
    M = len(false_positive_rates)

    has_micro_macro_avg = False
    if false_positive_rates.has_key('macro'):
        has_micro_macro_avg = True
        M -= 2

    # Init matplotlib
    fig = plt.figure()

    # Draw ROC curve for each class
    for i in range(M):
        plt.plot(false_positive_rates[i], true_positive_rates[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                                                        ''.format(i, roc_aucs[i]))

    # Draw Macro and Micro roc curve
    if has_micro_macro_avg:
        plt.plot(false_positive_rates["micro"], true_positive_rates["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_aucs["micro"]),
                 linewidth=2)

        plt.plot(false_positive_rates["macro"], true_positive_rates["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_aucs["macro"]),
                 linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc="lower right")

    return fig


def draw_roc_curve_multiple(ground_truths, prediction_scores, plot_title=None):

    # Compute Roc values
    false_positive_rates, true_positive_rates, thresholds, roc_aucs = \
        evaluate_multiple(ground_truths, prediction_scores, compute_micro_macro_avg=True)

    # Plot ROC curve
    fig = plot_roc_curve_multiple(false_positive_rates, true_positive_rates, roc_aucs, plot_title)

    plt.show(fig)