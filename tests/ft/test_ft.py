import matplotlib.pyplot as plt
import numpy as np

# 修复中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 采样参数
fs = 500  # 采样率 (Hz)
t = np.linspace(0, 1, fs, endpoint=False)

# 生成测试信号: 50Hz + 120Hz
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# 绘制时域信号
plt.figure(figsize=(10, 3))
plt.plot(t, signal, label="时域信号")
plt.xlabel("时间 (s)")
plt.ylabel("幅值")
plt.title("时域信号：叠加 50Hz 和 120Hz 正弦波")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 傅里叶变换，得到频域表示
freqs = np.fft.rfftfreq(fs, 1 / fs)
fft_values = np.abs(np.fft.rfft(signal)) / len(signal)

# 绘制频域幅值谱
plt.figure(figsize=(10, 3))
plt.stem(freqs, fft_values, linefmt="-")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅值")
plt.title("频域信号（幅值谱）")
plt.grid(True)
plt.tight_layout()
plt.show()
