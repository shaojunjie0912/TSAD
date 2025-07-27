"""异常检测模型评价指标（基于异常标签）"""

import numpy as np

from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events


def affiliation_f(actual: np.ndarray, predicted: np.ndarray):
    """关联 F1-Score

    Args:
        actual (np.ndarray): _description_
        predicted (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result["precision"]
    R = result["recall"]
    F = 2 * P * R / (P + R)

    return F


def affiliation_precision(actual: np.ndarray, predicted: np.ndarray):
    """关联 Precision

    Args:
        actual (np.ndarray): _description_
        predicted (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result["precision"]

    return P


def affiliation_recall(actual: np.ndarray, predicted: np.ndarray):
    """关联 Recall

    Args:
        actual (np.ndarray): _description_
        predicted (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    events_pred = convert_vector_to_events(predicted)
    events_label = convert_vector_to_events(actual)
    Trange = (0, len(predicted))

    result = pr_from_events(events_pred, events_label, Trange)
    R = result["recall"]

    return R
