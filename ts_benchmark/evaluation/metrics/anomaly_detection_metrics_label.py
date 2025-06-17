import numpy as np

from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events


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


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    def classic_metrics(actual, predicted):
        """计算经典的精确率、召回率和F1分数"""
        P = precision_score(actual, predicted)
        R = recall_score(actual, predicted)
        F = f1_score(actual, predicted)
        return P, R, F

    def print_metrics(actual, predicted, case_name):
        """打印经典指标和affiliation指标的比较结果"""
        print(f"\n=== {case_name} ===")
        print("实际标签:", actual)
        print("预测标签:", predicted)

        # 经典指标
        P, R, F = classic_metrics(actual, predicted)
        print("\n经典指标:")
        print(f"Precision: {P:.4f}")
        print(f"Recall: {R:.4f}")
        print(f"F1: {F:.4f}")

        # Affiliation指标
        P_aff = affiliation_precision(actual, predicted)
        R_aff = affiliation_recall(actual, predicted)
        F_aff = affiliation_f(actual, predicted)
        print("\nAffiliation指标:")
        print(f"Precision: {P_aff:.4f}")
        print(f"Recall: {R_aff:.4f}")
        print(f"F1: {F_aff:.4f}")

    # 测试用例1: 完全匹配
    actual1 = np.array([0, 1, 1, 0, 0, 1, 0])
    predicted1 = np.array([0, 1, 1, 0, 0, 1, 0])
    print_metrics(actual1, predicted1, "完全匹配")

    # 测试用例2: 部分重叠
    actual2 = np.array([0, 1, 1, 1, 0, 0, 0])
    predicted2 = np.array([0, 0, 1, 1, 1, 0, 0])
    print_metrics(actual2, predicted2, "部分重叠")

    # 测试用例3: 时间偏移
    actual3 = np.array([0, 1, 1, 0, 0, 0, 0])
    predicted3 = np.array([0, 0, 1, 1, 0, 0, 0])
    print_metrics(actual3, predicted3, "时间偏移")

    # 测试用例4: 完全错位
    actual4 = np.array([0, 1, 1, 0, 0, 0, 0])
    predicted4 = np.array([0, 0, 0, 0, 1, 1, 0])
    print_metrics(actual4, predicted4, "完全错位")

    # 测试用例5: 多个异常事件
    actual5 = np.array([0, 1, 1, 0, 1, 1, 0])
    predicted5 = np.array([0, 1, 0, 0, 1, 1, 0])
    print_metrics(actual5, predicted5, "多个异常事件")
