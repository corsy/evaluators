import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from scipy import interp


def evaluate_binary(ground_truth, prediction_scores):
    """
    Evaluate roc curve in binary classification
    :param ground_truth: 1-d array, e.g. gt: [1, 0, 1, 0, 1], note: 0 or 1
    :param prediction_scores: 1-d array, probability for positive example,
                              e.g. pred: [0.3, 0.4, 0.1, 0.3]
    :return:  precision, recall, thresholds, avg_precision
    """

    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction_scores)
    avg_precision = average_precision_score(ground_truth, prediction_scores)

    return precision, recall, thresholds, avg_precision


def plot_precision_recall_curve_binary(precision, recall, avg_precision, plot_title=None):

    if plot_title is None:
        plot_title = 'ROC Curves'

    # Init matplotlib
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(precision, recall, label='Precision & Recall (AUC = {0:0.2f})'.format(avg_precision))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(plot_title)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size':7})

    return fig


def draw_roc_curve_binary(ground_truth, prediction_score, plot_title=None):

    # Compute Precision Recall curve
    precision, recall, thresholds, avg_precision = \
        evaluate_binary(ground_truth, prediction_score)

    # Plot ROC curve
    fig = plot_precision_recall_curve_binary(precision, recall, avg_precision, plot_title)

    plt.show(fig)


def savefig_roc_curve_binary(fig_file_path, ground_truth, prediction_score, plot_title=None):

    plt.clf()

    # Compute Precision Recall curve
    precision, recall, thresholds, avg_precision = \
        evaluate_binary(ground_truth, prediction_score)

    # Plot ROC curve
    fig = plot_precision_recall_curve_binary(precision, recall, avg_precision, plot_title)

    plt.imsave(fig_file_path)


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

    precisions = {}
    recalls = {}
    thresholds = {}
    avg_precisions = {}

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
        precision, recall, threshold = precision_recall_curve(ground_truth_label, prediction_score)
        avg_precision = average_precision_score(ground_truth_label, prediction_score)

        precisions[class_label] = precision
        recalls[class_label] = recall
        thresholds[class_label] = threshold
        avg_precisions[class_label] = avg_precision

        if compute_micro_macro_avg:
            gt_label_array.append(ground_truth_label)
            prediction_score_array.append(prediction_score)

    if compute_micro_macro_avg:
        gt_label_array = np.asarray(gt_label_array)
        prediction_score_array = np.asarray(prediction_score_array)

        # Compute Micro Avg.
        precisions["micro"], recalls["micro"], _ = precision_recall_curve(gt_label_array.ravel(),
                                                                                   prediction_score_array.ravel())
        avg_precisions["micro"] = average_precision_score(gt_label_array, prediction_score_array, average="micro")

    return precisions, recalls, thresholds, avg_precisions


def plot_precision_recall_curve_multiple(precisions, recalls, avg_precisions, plot_title=None):

    if plot_title is None:
        plot_title = 'ROC Curves'

    # Determine how many classes
    M = len(precisions)

    has_micro_macro_avg = False
    if precisions.has_key('micro'):
        has_micro_macro_avg = True
        M -= 1

    # Init matplotlib
    fig = plt.figure()
    ax = plt.subplot(111)

    # Draw ROC curve for each class
    for i in range(M):
        ax.plot(precisions[i], recalls[i], label='class {0} (AUC = {1:0.2f})'
                                                                        ''.format(i, avg_precisions[i]))

    # Draw Macro and Micro roc curve
    if has_micro_macro_avg:
        ax.plot(precisions["micro"], recalls["micro"],
                 label='micro-avg (AUC = {0:0.2f})'
                       ''.format(avg_precisions["micro"]),
                 linewidth=2)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(plot_title)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={'size':7})

    return fig


def draw_precision_recall_curve_multiple(ground_truths, prediction_scores, plot_title=None):

    # Compute Roc values
    precisions, recalls, thresholds, avg_precisions = \
        evaluate_multiple(ground_truths, prediction_scores, compute_micro_macro_avg=True)

    # Plot ROC curve
    fig = plot_precision_recall_curve_multiple(precisions, recalls, avg_precisions, plot_title)

    plt.show(fig)


def savefig_precision_recall_curve_multiple(fig_file_path, ground_truths, prediction_scores, plot_title=None):

    plt.clf()

    # Compute Roc values
    precisions, recalls, thresholds, avg_precisions = \
        evaluate_multiple(ground_truths, prediction_scores, compute_micro_macro_avg=True)

    # Plot ROC curve
    fig = plot_precision_recall_curve_multiple(precisions, recalls, avg_precisions, plot_title)

    plt.savefig(fig_file_path)
