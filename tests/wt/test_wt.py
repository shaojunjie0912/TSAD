import matplotlib.pyplot as plt
import numpy as np
import pywt

# matplotlib 中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

print(pywt.wavelist("db"))  # type: ignore

# 创建一个简单的测试信号：低频+高频
t = np.linspace(0, 1, 512, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# 选择小波基（例如 Daubechies 2）
wavelet = "db2"

# 执行一级离散小波分解
cA, cD = pywt.dwt(signal, wavelet)

# 使用逆变换重构原信号
reconstructed_signal = pywt.idwt(cA, cD, wavelet)

# 可视化结果
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, signal)
plt.title("原始信号")

plt.subplot(4, 1, 2)
plt.plot(cA)
plt.title("近似系数（低频）")

plt.subplot(4, 1, 3)
plt.plot(cD)
plt.title("细节系数（高频）")

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, "--", label="重构信号")
plt.plot(t, signal, alpha=0.5, label="原信号")
plt.legend()
plt.title("重构信号（逆小波变换）")

plt.tight_layout()
plt.show()
