import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# 假设有一些真实的标签 (y_true) 和模型预测的概率 (y_scores)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6, 0.25, 0.15])

# 计算 FPR, TPR 和阈值
# roc_curve 函数返回三个值：fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

print(f"FPR: {fpr}")
print(f"TPR: {tpr}")
print(f"Thresholds: {thresholds}")
print(f"AUC: {roc_auc:.4f}")

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")

# 也可以直接使用 roc_auc_score
from sklearn.metrics import roc_auc_score

auc_score_direct = roc_auc_score(y_true, y_scores)
print(f"Direct AUC score: {auc_score_direct:.4f}")
