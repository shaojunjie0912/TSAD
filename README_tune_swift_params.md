# SWIFT 超参数批量调优工具

## 概述

这是一个优化合并的超参数调优工具，整合了原来的 `run_all_tunings.py` 和 `ts_benchmark/tune_params.py` 的功能。新工具支持一次命令行调用就完成多个异常率的批量超参数优化。

## 主要改进

### 🚀 性能优化

- **消除子进程开销**: 避免多次创建和切换进程
- **共享数据加载**: 数据集只加载一次，在内存中共享
- **减少文件I/O**: 无需创建和清理临时配置文件

### 🎯 使用便利性

- **统一命令行接口**: 只需一次传参即可批量处理
- **灵活参数配置**: 支持自定义异常率列表、试验次数等
- **智能默认值**: 提供合理的默认参数，开箱即用

### 📊 功能增强

- **实时进度显示**: 清晰显示当前处理的异常率和进度
- **错误容错**: 单个异常率失败不影响其他异常率的调优
- **断点续传**: 基于 Optuna SQLite 存储，支持中断后继续

## 文件结构

```
.
├── tune_swift_params.py          # 新的合并调优脚本
├── example_usage.py              # 使用示例脚本
├── README_tune_swift_params.md   # 本说明文档
├── run_all_tunings.py           # 原始批量脚本（可删除）
└── ts_benchmark/tune_params.py  # 原始单次调优脚本（可删除）
```

## 使用方法

### 基本用法

```bash
python tune_swift_params.py \
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \
  --dataset datasets/ASD_dataset_1.csv \
  --task-name find_anomalies \
  --dataset-name ASD_dataset_1 \
  --anomaly-ratios 1,3,5,8 \
  --n-trials 100
```

### 完整参数列表

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--config` | str | ✅ | - | 基础配置文件路径 |
| `--dataset` | str | ✅ | - | 数据集文件路径 |
| `--task-name` | str | ✅ | - | 任务名称 (如: find_anomalies) |
| `--dataset-name` | str | ✅ | - | 数据集名称 (如: ASD_dataset_1) |
| `--anomaly-ratios` | str | ❌ | "1,3,5,8" | 异常率列表，逗号分隔 |
| `--n-trials` | int | ❌ | 100 | 每个异常率的试验次数 |
| `--threshold-strategy` | str | ❌ | "adaptive" | 阈值计算策略 |
| `--aggregation-method` | str | ❌ | "weighted_max" | 分数聚合方法 |
| `--output-pattern` | str | ❌ | 见下方 | 输出文件路径模式 |
| `--study-name-pattern` | str | ❌ | 见下方 | Optuna 研究名称模式 |
| `--seed` | int | ❌ | 1037 | 随机种子 |

**默认输出模式:**

- `--output-pattern`: `"configs/{task_name}/{dataset_name}/swift_best_ar_{ratio}.toml"`
- `--study-name-pattern`: `"{task_name}-{dataset_name}-ar{ratio}"`

### 示例命令

#### 1. 快速测试 (少量试验)

```bash
python tune_swift_params.py \
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \
  --dataset datasets/ASD_dataset_1.csv \
  --task-name find_anomalies \
  --dataset-name ASD_dataset_1 \
  --anomaly-ratios 3,5 \
  --n-trials 20
```

#### 2. 完整调优 (更多试验)

```bash
python tune_swift_params.py \
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \
  --dataset datasets/ASD_dataset_1.csv \
  --task-name find_anomalies \
  --dataset-name ASD_dataset_1 \
  --anomaly-ratios 1,3,5,8,10 \
  --n-trials 200
```

#### 3. 自定义输出路径

```bash
python tune_swift_params.py \
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \
  --dataset datasets/ASD_dataset_1.csv \
  --task-name find_anomalies \
  --dataset-name ASD_dataset_1 \
  --output-pattern "results/{task_name}/{dataset_name}/best_{ratio}.toml" \
  --study-name-pattern "study_{task_name}_{dataset_name}_{ratio}"
```

## 输出文件

### 配置文件

每个异常率会生成一个最佳配置文件：

```
configs/find_anomalies/ASD_dataset_1/swift_best_ar_1.toml
configs/find_anomalies/ASD_dataset_1/swift_best_ar_3.toml
configs/find_anomalies/ASD_dataset_1/swift_best_ar_5.toml
configs/find_anomalies/ASD_dataset_1/swift_best_ar_8.toml
```

### Optuna 数据库

每个异常率对应一个 SQLite 数据库文件：

```
find_anomalies-ASD_dataset_1-ar1.db
find_anomalies-ASD_dataset_1-ar3.db
find_anomalies-ASD_dataset_1-ar5.db
find_anomalies-ASD_dataset_1-ar8.db
```

## 运行示例

运行提供的示例脚本：

```bash
python example_usage.py
```

查看详细帮助：

```bash
python example_usage.py --help
```

## 性能对比

| 指标 | 原始方案 | 新方案 | 改进 |
|------|----------|--------|------|
| 进程创建 | 4次 | 1次 | 减少75% |
| 数据加载 | 4次 | 1次 | 减少75% |
| 临时文件 | 4个 | 0个 | 完全避免 |
| 命令行调用 | 1次主+4次子 | 1次 | 简化80% |
| 内存使用 | 4x基础 | 1x基础 | 减少75% |

## 注意事项

1. **内存要求**: 由于数据集在内存中共享，确保有足够内存
2. **中断恢复**: 可以随时中断，下次运行会自动从断点继续
3. **并行运行**: 多个异常率按顺序执行，每个异常率内部并行优化
4. **错误处理**: 单个异常率失败不会影响其他异常率的处理

## 依赖要求

确保安装以下Python包：

```
optuna
pandas
numpy
tomli
tomli-w
# 以及您项目特定的依赖
```

## 迁移指南

### 从原始脚本迁移

**原始用法:**

```bash
python run_all_tunings.py  # 固化在脚本中的参数
```

**新用法:**

```bash
python tune_swift_params.py \
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \
  --dataset datasets/ASD_dataset_1.csv \
  --task-name find_anomalies \
  --dataset-name ASD_dataset_1
```

### 清理旧文件

完成迁移后，可以删除以下文件：

- `run_all_tunings.py`
- `ts_benchmark/tune_params.py` (如果不再单独使用)

## 故障排除

### 常见问题

1. **导入错误**: 确保相关模块路径正确
2. **内存不足**: 减少 `--n-trials` 或处理更小的数据集
3. **路径错误**: 检查配置文件和数据集路径是否存在
4. **权限问题**: 确保对输出目录有写入权限

### 调试建议

1. 先用少量试验测试: `--n-trials 5`
2. 减少异常率数量: `--anomaly-ratios 3`
3. 检查日志输出中的详细错误信息
4. 验证配置文件格式是否正确
