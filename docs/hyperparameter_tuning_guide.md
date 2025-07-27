# SWIFT 超参数调优指南

本指南介绍如何使用 Optuna 进行 SWIFT 算法的超参数调优，包括可视化分析和收敛判断。

## 目录

1. [基础调优](#基础调优)
2. [可视化分析](#可视化分析)
3. [收敛判断](#收敛判断)
4. [试验次数选择](#试验次数选择)
5. [最佳实践](#最佳实践)

## 基础调优

### 基本命令

```bash
# 异常评分任务
.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name CalIt2 \
    --train-val-len 2520 \
    --algorithm-name swift \
    --anomaly-ratio 1.0 \
    --n-trials 50

# 异常检测任务
.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name find_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 100
```

### 参数说明

- `--task-name`: 任务类型 (`score_anomalies` 或 `find_anomalies`)
- `--dataset-name`: 数据集名称
- `--train-val-len`: 训练验证集总长度
- `--algorithm-name`: 算法名称 (`swift`)
- `--anomaly-ratio`: 异常率百分比
- `--n-trials`: Optuna 试验次数
- `--restart`: 强制重新开始（删除已有进度）

## 可视化分析

### 安装依赖

```bash
pip install plotly
```

### 生成可视化图表

调优完成后，使用独立的可视化脚本：

```bash
.venv/bin/python ts_benchmark/visualize_optuna.py \
    --study-db optuna_studies/score_anomalies_CalIt2_swift_ratio_1.0.db \
    --study-name score_anomalies_CalIt2_swift_ratio_1.0
```

### 可视化图表类型

生成的 HTML 文件包含以下交互式图表：

1. **optimization_history.html**: 优化历史
   - 显示每次试验的结果
   - 最佳值趋势线
   - 用于观察整体优化进展

2. **param_importances.html**: 参数重要性
   - 显示哪些参数对结果影响最大
   - 帮助理解模型行为
   - 指导后续调优重点

3. **param_slice.html**: 参数切片分析
   - 显示单个参数与目标值的关系
   - 帮助理解参数的最优范围

4. **parallel_coordinate.html**: 并行坐标图
   - 显示高性能试验的参数组合
   - 帮助发现参数间的相关性

5. **convergence_analysis.html**: 收敛分析
   - 自定义收敛图表
   - 显示最佳值变化趋势
   - 帮助判断是否需要更多试验

## 收敛判断

### 自动收敛分析

调优脚本会自动分析收敛情况：

```
🔍 收敛分析:
  - 是否收敛: 否
  - 原因: 改进率过高
  - 改进率: 0.0234
  - 分析试验数: 50
```

### 收敛标准

- **改进率 < 0.1%**: 认为已收敛
- **窗口大小**: 默认分析最近 20 个试验
- **最小试验数**: 至少需要 20 个完成的试验

### 手动判断方法

1. **查看优化历史图**:
   - 最佳值曲线趋于平缓
   - 最近的试验很少产生改进

2. **观察参数重要性**:
   - 重要参数已被充分探索
   - 参数范围覆盖合理

3. **检查试验分布**:
   - 高性能区域被充分采样
   - 参数组合趋于稳定

## 试验次数选择

### 推荐试验次数

系统会根据数据集和算法自动推荐试验次数：

```python
# 基础试验次数: 50
# 数据集调整系数:
#   - PSM: 2.0 (大数据集)
#   - MSL: 1.5 (中等数据集)  
#   - CalIt2: 1.0 (小数据集)
# 算法调整系数:
#   - SWIFT: 1.5 (复杂算法)
```

### 实际建议

| 数据集 | 推荐试验次数 | 说明 |
|--------|--------------|------|
| CalIt2 | 75 | 小数据集，中等复杂度 |
| MSL | 115 | 中等数据集，需要更多探索 |
| PSM | 150 | 大数据集，参数敏感 |

### 渐进式调优

```bash
# 第一轮：快速探索
--n-trials 50

# 第二轮：精细调优（会自动继续）
--n-trials 100

# 第三轮：最终优化
--n-trials 150
```

## 最佳实践

### 1. 分阶段调优

```bash
# 阶段1: 快速探索 (50 trials)
# 阶段2: 深入优化 (100 trials)  
# 阶段3: 精细调优 (150 trials)
```

### 2. 监控收敛

- 每 50 个试验检查一次收敛情况
- 使用可视化图表辅助判断
- 关注参数重要性变化

### 3. 参数范围调整

如果发现最优值在边界：

```python
# 在 tune_params_swift.py 中调整 PARAM_CONFIG
"learning_rate": {
    "low": 1e-5,    # 降低下界
    "high": 1e-1,   # 提高上界
}
```

### 4. 结果验证

```bash
# 使用最佳配置多次运行验证稳定性
# 检查不同数据集的泛化性能
```

### 5. 资源管理

- 使用 `--restart` 重新开始
- 监控 GPU 内存使用
- 合理设置批量大小

## 故障排除

### 常见问题

1. **GPU 内存不足**
   - 减小 batch_size 搜索范围
   - 降低模型复杂度参数

2. **收敛过慢**
   - 增加试验次数
   - 调整参数搜索范围
   - 检查数据质量

3. **可视化失败**
   - 确保安装了 plotly
   - 检查数据库文件完整性

### 性能优化

- 使用 TPE 采样器（默认）
- 启用 MedianPruner（默认）
- 合理设置早停参数

## 示例工作流

```bash
# 1. 运行完整的调优+可视化流程
bash scripts/tuning/example_with_visualization.sh

# 2. 查看结果
open optuna_visualizations/CalIt2/swift/optimization_history.html

# 3. 根据收敛情况决定是否继续
# 如果未收敛，增加试验次数继续调优
```

这个工作流程能够帮助您系统地进行超参数调优，并通过可视化分析做出明智的决策。 