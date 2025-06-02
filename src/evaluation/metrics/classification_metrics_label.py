import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .vus_metrics import metricor

__all__ = [
    "accuracy",
    "f_score",
    "precision",
    "recall",
    "adjust_accuracy",
    "adjust_f_score",
    "adjust_precision",
    "adjust_recall",
    "rrecall",
    "rprecision",
    "precision_at_k",
    "rf",
    "affiliation_f",
    "affiliation_precision",
    "affiliation_recall",
]


def precision(actual: np.ndarray, predicted: np.ndarray):
    """

    Args:
        actual (np.ndarray): _description_
        predicted (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    return metrics.precision_score(actual, predicted, zero_division=0)


def recall(actual: np.ndarray, predicted: np.ndarray):
    return metrics.recall_score(actual, predicted, zero_division=0)


def f_score(actual: np.ndarray, predicted: np.ndarray):
    return metrics.f1_score(actual, predicted, zero_division=0)


def accuracy(actual: np.ndarray, predicted: np.ndarray):
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def adjust_predicts(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    调整检测结果
    异常检测算法在一个异常区间检测到某点存在异常，则认为算法检测到整个异常区间的所有异常点
    先从检测到的异常点从后往前调整检测结果，随后再从该点从前往后调整检测结果，直到真实的异常为False
    退出异常状态，结束当前区间的调整

    :param actual: 真实的异常。
    :param predicted: 检测所得的异常。
    :return: 调整后的异常检测结果。
    """
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
            for j in range(i, len(actual)):
                if actual[j] == 0:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = 1
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state:
            predicted[i] = 1
    return predicted


def adjust_precision(actual: np.ndarray, predicted: np.ndarray):
    predicted = adjust_predicts(actual, predicted)
    return metrics.precision_score(actual, predicted, zero_division=0)


def adjust_recall(actual: np.ndarray, predicted: np.ndarray):
    predicted = adjust_predicts(actual, predicted)
    return metrics.recall_score(actual, predicted, zero_division=0)


def adjust_f_score(actual: np.ndarray, predicted: np.ndarray):
    predicted = adjust_predicts(actual, predicted)
    return metrics.f1_score(actual, predicted, zero_division=0)


def adjust_accuracy(actual: np.ndarray, predicted: np.ndarray):
    predicted = adjust_predicts(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    return accuracy


def affiliation_f(actual: np.ndarray, predicted: np.ndarray):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result["precision"]
    R = result["recall"]
    F = 2 * P * R / (P + R)

    return F


def affiliation_precision(actual: np.ndarray, predicted: np.ndarray):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result["precision"]

    return P


def affiliation_recall(actual: np.ndarray, predicted: np.ndarray):
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    R = result["recall"]

    return R


metricor_grader = metricor()


def rrecall(actual: np.ndarray, predicted: np.ndarray):
    result = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    if isinstance(result, tuple):
        return result[0]["range_recall"]
    return result["range_recall"]


def rprecision(actual: np.ndarray, predicted: np.ndarray):
    result = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    if isinstance(result, tuple):
        return result[0]["range_precision"]
    return result["range_precision"]


def rf(actual: np.ndarray, predicted: np.ndarray):
    result = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    if isinstance(result, tuple):
        return result[0]["range_f"]
    return result["range_f"]


def precision_at_k(actual: np.ndarray, predicted: np.ndarray):
    result = metricor_grader.metric_new(actual, predicted, plot_ROC=False)
    if isinstance(result, tuple):
        return result[0]["precision_at_k"]
    return result["precision_at_k"]
