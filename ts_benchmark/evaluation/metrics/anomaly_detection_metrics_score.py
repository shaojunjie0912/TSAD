"""异常检测模型评价指标（基于异常评分）"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray):
    """计算 ROC 曲线下面积 (AUC-ROC)

    Args:
        y_true (np.ndarray): 真实标签
        y_scores (np.ndarray): 预测得分

    Returns:
        float: AUC-ROC 值
    """
    return roc_auc_score(y_true, y_scores)
