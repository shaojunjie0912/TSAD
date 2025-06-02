import numpy as np
from metrics.classification_metrics_label import *

if __name__ == "__main__":
    actual = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    print("原始precision:", precision(actual, pred))
    print("调整后precision:", adjust_precision(actual, pred))
    print("原始recall:", recall(actual, pred))
    print("调整后recall:", adjust_recall(actual, pred))
