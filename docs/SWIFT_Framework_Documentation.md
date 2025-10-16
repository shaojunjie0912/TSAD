# SWIFT: 无监督多变量时间序列异常值检测框架

## 概述

SWIFT（**S**tationary **W**avelet-patched **I**nter-channel **F**usion **T**ransformer for Anomaly Detection）是一个先进的无监督多变量时间序列异常检测框架。该框架基于多尺度小波变换和图引导通道融合技术，实现了鲁棒的多变量时间序列异常检测。

### 核心创新

1. **多尺度小波变换与时序分块**：通过平稳小波变换（SWT）将时间序列分解为多个尺度的频域表示，结合滑窗分块策略提取时空特征。

2. **图引导通道融合**：设计了基于图注意力网络（GAT）的通道掩码器，动态学习通道间的相关性，指导Transformer的注意力机制。

3. **双域重构学习**：同时在时间域和尺度域进行重构学习，提高异常检测的准确性和鲁棒性。

4. **通道相关性发掘损失**：创新性地设计了CCD损失函数，协同优化GAT掩码器和Transformer注意力机制。

## 整体架构

SWIFT框架采用编码器-解码器结构，主要包含三个核心模块：

```text
输入时间序列 → 前向模块(FM) → 通道融合模块(CFM) → 时尺重构模块(TSRM) → 重构输出
```

### 架构图

```text
┌─────────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│   前向模块(FM)  │ →  │  通道融合模块(CFM)   │ →  │ 时尺重构模块(TSRM)│
│                 │    │                      │    │                   │
│ • RevIN归一化   │    │ • GAT通道掩码器      │    │ • 重构头          │
│ • 小波变换      │    │ • 通道掩码Transformer│    │ • 逆小波变换      │
│ • 时序分块      │    │ • CCD损失优化        │    │ • RevIN逆归一化   │
└─────────────────┘    └──────────────────────┘    └───────────────────┘
```

## 核心模块详解

### 1. 前向模块（Forward Module, FM）

前向模块负责对原始时间序列进行预处理和特征提取：

#### 1.1 RevIN（可逆实例归一化）

- **功能**：消除时间序列的分布差异，提高模型泛化能力
- **特点**：
  - 支持可学习的仿射参数
  - 可选择使用最后一个值或均值进行归一化
  - 训练后可完全逆转归一化过程

```python
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        # 实例归一化参数初始化
        
    def forward(self, x, mode):
        # mode: "norm"(归一化) | "denorm"(逆归一化) | "transform"(转换)
```

#### 1.2 小波变换分块器（WaveletPatcher）

- **功能**：将时间序列分解为多尺度频域表示
- **技术**：平稳小波变换（SWT）
- **参数**：
  - `level`：分解层数（默认4层）
  - `wavelet`：小波基类型（默认"db4"）
  - `patch_size`：分块大小（默认16）
  - `patch_stride`：分块步长（默认8）

**处理流程**：

1. 输入：`(batch_size, seq_len, num_features)`
2. 小波分解：产生 `L+1` 层系数（1个近似系数 + L个细节系数）
3. 时序分块：沿时间维度进行滑窗分块
4. 输出：`(batch_size * patch_num, extended_num_features, patch_size)`
   - `extended_num_features = num_features * (L + 1)`

### 2. 通道融合模块（Channel Fusion Module, CFM）

CFM是SWIFT框架的核心创新，通过图引导的注意力机制学习通道间的复杂关联：

#### 2.1 GAT通道掩码器（GATChannelMasker）

- **设计理念**：将多变量时间序列的各通道视为图节点，学习通道间的动态关联性
- **技术特点**：
  - 多头图注意力机制
  - 生成软掩码指导后续Transformer注意力
  - 支持动态通道重要性权重学习

```python
class GATChannelMasker(nn.Module):
    def __init__(self, node_feature_dim, num_features, num_gat_heads, gat_head_dim):
        # GAT参数初始化
        
    def forward(self, x_nodes):
        # 输入: (B_eff, num_channels, node_feature_dim)
        # 输出: (B_eff, num_channels, num_channels) 软掩码
```

#### 2.2 通道掩码Transformer（ChannelMaskedTransformer）

- **核心功能**：在GAT软掩码指导下进行通道间特征融合
- **架构特点**：
  - 多层Transformer编码器
  - 预归一化设计
  - 集成CCD损失优化

**关键组件**：

- **PreNorm**：预归一化层，缓解高幅值频率分量的过度关注
- **ChannelMaskedAttention**：掩码引导的多头注意力机制
- **FeedForward**：前馈网络层

#### 2.3 通道相关性发掘损失（CCD Loss）

CCD损失是SWIFT框架的关键创新，包含两个组成部分：

```python
class AdaptiveCCDLoss(nn.Module):
    def __init__(self, alignment_temperature, regularization_lambda, alignment_lambda):
        # 损失函数参数初始化
        
    def forward(self, cfm_attention_logits, soft_channel_mask):
        # 1. 通道注意力对齐损失
        alignment_loss = -(soft_channel_mask * log_probs_cfm_attention).sum(dim=-1).mean()
        
        # 2. 掩码正则化损失  
        mask_regularization_loss = torch.norm(identity_matrix - soft_channel_mask, p='fro').mean()
        
        return alignment_lambda * alignment_loss + regularization_lambda * mask_regularization_loss
```

**损失函数作用**：

1. **对齐损失**：让Transformer注意力学习GAT发现的显式图结构
2. **正则化损失**：鼓励学习稀疏且重要的通道关联

### 3. 时尺重构模块（Time-Scale Reconstruction Module, TSRM）

TSRM负责将融合后的特征重构回原始时间序列：

#### 3.1 重构头（ReconstructionHead）

- **支持模式**：
  - `individual=True`：每个变量使用独立网络
  - `individual=False`：所有变量共享网络（含残差连接）
- **功能**：将多维特征映射回时间序列长度

#### 3.2 逆小波变换器（InverseWaveletPatcher）

- **功能**：通过逆平稳小波变换（iSWT）重构时域信号
- **处理**：将扩展通道的小波系数重构为原始通道数的时间序列

## 训练流程

### 损失函数设计

SWIFT采用多域联合损失训练：

```python
total_loss = time_rec_loss + λ_scale * scale_rec_loss + λ_ccd * ccd_loss
```

其中：

- **time_rec_loss**：时域重构损失（HuberLoss）
- **scale_rec_loss**：尺度域重构损失（HuberLoss）  
- **ccd_loss**：通道相关性发掘损失

### 训练策略

1. **早停机制**：基于验证集损失，防止过拟合
2. **学习率调度**：OneCycleLR策略，包含预热阶段
3. **优化器**：Adam优化器，支持权重衰减
4. **批量训练**：支持GPU加速训练

### 关键配置参数

```toml
[training]
batch_size = 32       # 批量大小
num_epochs = 30       # 训练轮次  
learning_rate = 0.001 # 学习率
weight_decay = 0.0    # L2正则化
pct_start = 0.3       # 预热比例
es_patience = 10      # 早停耐心值
```

## 异常检测机制

### 异常评分计算

SWIFT采用双域异常评分策略：

```python
# 时间域分数
time_score = MSE(x_reconstructed, x_original)

# 尺度域分数  
scale_score = MSE(coeffs_reconstructed, coeffs_original)

# 综合分数
final_score = time_score + λ_scale * scale_score
```

### 自适应阈值策略

SWIFT设计了智能的自适应阈值计算：

1. **偏度分析**：根据验证集分数的偏度选择策略
2. **高偏度情况**：使用鲁棒的尾部分析方法
3. **低偏度情况**：使用改进的百分位数方法
4. **动态调整**：基于变异系数动态调整阈值

```python
def _calculate_threshold(self, val_scores):
    if abs(skewness) > 1.5:  # 高偏度
        # 使用鲁棒方法
        q_robust = 92.0 + min(3.0, abs(skewness))
        threshold = np.percentile(tail_scores, 75.0)
    else:  # 低偏度  
        # 使用改进百分位数
        cv = std_score / (mean_score + 1e-8)
        adjusted_percentile = base_percentile - min(2.0, cv * 10)
        threshold = np.percentile(val_scores, max(90.0, adjusted_percentile))
```

### 分数聚合策略

为处理滑窗重叠问题，SWIFT采用加权聚合：

```python
final_scores = α * mean_scores + (1-α) * max_scores
```

- `α`：均值权重（默认0.3）
- `1-α`：最大值权重，突出异常峰值

## 配置系统

SWIFT采用TOML格式的层次化配置系统：

### 核心配置块

```toml
[anomaly_detection]
scale_score_lambda = 0.2      # 尺度域分数权重
anomaly_ratio = 3.0           # 异常率百分比
score_aggregation_alpha = 0.3 # 分数聚合权重

[data]
seq_len = 128        # 滑窗大小
patch_size = 16      # 分块大小
patch_stride = 8     # 分块步长

[model.FM]
level = 4            # 小波分解层数
wavelet = "db4"      # 小波类型

[model.CFM]  
num_layers = 4       # Transformer层数
d_cf = 64           # 内部特征维度
num_heads = 2       # 注意力头数
num_gat_heads = 4   # GAT注意力头数

[loss]
scale_loss_lambda = 0.5    # 尺度域损失权重
ccd_loss_lambda = 0.001    # CCD损失权重
```

## 评估指标

### 基于标签的评估

- **Affiliation-F1**：考虑时间容忍的F1分数
- **Affiliation-Precision**：时间容忍精确率
- **Affiliation-Recall**：时间容忍召回率

### 基于分数的评估

- **AUC-ROC**：ROC曲线下面积

## 超参数优化

SWIFT集成了基于Optuna的自动超参数优化：

### 优化空间定义

```python
def objective(trial):
    # 数据参数
    seq_len = trial.suggest_categorical('seq_len', [64, 96, 128, 192, 256])
    patch_size = trial.suggest_categorical('patch_size', [8, 12, 16, 20, 24])
    
    # 模型参数
    level = trial.suggest_int('level', 2, 6)
    num_layers = trial.suggest_int('num_layers', 2, 8)
    d_cf = trial.suggest_categorical('d_cf', [32, 64, 128, 256])
    
    # 损失参数
    scale_loss_lambda = trial.suggest_float('scale_loss_lambda', 0.1, 2.0)
    ccd_loss_lambda = trial.suggest_float('ccd_loss_lambda', 0.0001, 0.01)
```

### 可视化分析

框架提供丰富的优化过程可视化：

- 优化历史图
- 参数重要性分析
- 参数关系图
- 并行坐标图
- 收敛分析图

## 使用示例

### 基本使用

```python
import numpy as np
from swift_pipeline import SWIFTPipeline
import tomli

# 加载配置
with open('configs/base/swift.toml', 'rb') as f:
    config = tomli.load(f)

# 创建pipeline
pipeline = SWIFTPipeline(config)

# 训练模型
pipeline.fit(train_val_data)

# 异常检测
predictions, scores = pipeline.find_anomalies(test_data)
```

### 快捷函数

```python
from swift_pipeline import swift_find_anomalies, swift_score_anomalies

# 直接异常检测
predictions = swift_find_anomalies(all_data, config)

# 直接异常评分
scores = swift_score_anomalies(all_data, config)
```

## 技术优势

### 1. 多尺度表示学习

- 通过小波变换捕获不同频率特征
- 避免单一尺度表示的局限性
- 提高对复杂异常模式的检测能力

### 2. 图引导注意力机制

- GAT动态学习通道关联性
- 软掩码指导Transformer注意力
- 避免全连接注意力的计算冗余

### 3. 双域联合优化

- 时间域保持时序连续性
- 尺度域捕获频域异常
- 提高检测准确性和鲁棒性

### 4. 自适应异常检测

- 智能阈值选择策略
- 考虑分数分布特性
- 减少人工参数调整

### 5. 端到端可训练

- 联合优化所有模块
- CCD损失协同训练
- 避免分阶段训练的次优解

## 适用场景

SWIFT框架特别适用于以下场景：

1. **多传感器系统监控**：工业设备、智能建筑等
2. **金融时间序列分析**：股票价格、交易量等多维数据
3. **网络流量监控**：多节点、多指标的网络异常检测
4. **医疗信号分析**：多导联心电图、脑电图等
5. **环境监测**：气象站、污染监测等多参数数据

## 性能特点

- **无监督学习**：无需异常标签，适应性强
- **实时检测**：支持流式数据处理
- **可扩展性**：支持任意维度的多变量数据
- **鲁棒性**：对噪声和缺失数据具有良好容错性
- **可解释性**：提供通道重要性和注意力权重分析

## 总结

SWIFT框架通过创新的多尺度小波变换、图引导通道融合和双域重构学习，为无监督多变量时间序列异常检测提供了一个强大且灵活的解决方案。其独特的设计理念和技术创新使其在准确性、鲁棒性和可扩展性方面都表现出色，为时间序列异常检测领域带来了新的突破。
