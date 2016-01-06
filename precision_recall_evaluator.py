import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def find_max_indices(prediction_results):

    """
    Find the max indices in prediction results
    :param prediction_results:
    :return:
    """
    N = prediction_results.shape[0]
    indices = np.zeros(N, dtype=int)

    # Find max probability in each row
    for i in range(0, N):
        result = prediction_results[i]
        idx = np.argmax(result)
        indices[i] = idx

    return indices

# Evaluate Multiple Class
def evaluate_mutiple(ground_truth, prediction, find_max=False, f_beta = 1.0, avg_method=None):
    """
    :param ground_truth: 1-d array, e.g. gt: [1, 1, 2, 2, 3]
    :param prediction: 1-d array, e.g. prediction: [1, 1, 2, 2, 4]
    :return: recall, precision, f-value
    """

    prediction_indices = prediction

    if find_max or len(prediction.shape) == 2:
        prediction_indices = find_max_indices(prediction)

    # Find Precision & Recall & F-value
    precision, recall, f_value, support = None, None, None, None

    if len(prediction.shape) == 2:
        M = prediction.shape[1]
        precision, recall, f_value, support \
            = precision_recall_fscore_support(ground_truth,
                                              prediction_indices,
                                              beta=f_beta,
                                              pos_label=M,
                                              average=avg_method)
    else:
        precision, recall, f_value, support \
            = precision_recall_fscore_support(ground_truth,
                                              prediction_indices,
                                              beta=f_beta,
                                              average=avg_method)

    return precision, recall, f_value


def find_binary_values(prediction_results, multiple_index, threshold):
    """
    Find the item that beyond the threshold
    :param prediction_results:
    :param multiple_index:
    :param threshold:
    :return:
    """
    N = prediction_results.shape[0]
    values = np.zeros(N, dtype=int)

    # Find max probability in each row
    for i in range(0, N):
        result = prediction_results[i]

        if result[multiple_index] >= threshold:
            values[i] = 1

    return values


# Evaluate Binary Class
def evaluate_binary(ground_truth, prediction, multiple_index=None, threshold=None, f_beta=1.0):
    """
    :param ground_truth: 1-d array, e.g. gt: [1, 0, 1, 0, 0], 1 is True
    :param prediction: 1-d array, e.g. prediction: [0, 0, 1, 0, 0]
                       if prediction is 2-d array, then multiple_index, threshold should be configured
    :return: recall, precision, f-value
    """

    prediction_results = prediction
    if threshold is not None and multiple_index is not None:
        prediction_results = find_binary_values(prediction, multiple_index, threshold)

    # Find Precision & Recall & F-value
    precision, recall, f_value, support \
        = precision_recall_fscore_support(ground_truth,
                                          prediction_results,
                                          average='binary',
                                          beta=f_beta)
    return precision, recall, f_value

