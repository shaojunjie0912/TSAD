# Optuna 进度管理功能

## 概述

为了解决 `tune_params_swift.py` 在调优过程中缺少进度保存逻辑的问题，我们添加了基于 SQLite 数据库的进度保存和恢复功能。现在，如果程序因为任何原因终止，下次重新启动时会自动从上次停止的地方继续。

## 主要功能

### 1. 自动进度保存
- 使用 SQLite 数据库保存所有试验结果
- 数据库文件保存在 `optuna_studies/` 目录下
- 文件名格式：`{task_name}_{dataset_name}_{algorithm_name}_ratio_{anomaly_ratio}.db`

### 2. 自动进度恢复
- 程序启动时自动检查是否存在之前的调优进度
- 如果存在，会显示已完成的试验数量和当前最佳分数
- 自动计算剩余需要执行的试验数量

### 3. 强制重新开始
- 使用 `--restart` 参数可以强制重新开始调优
- 会删除现有的数据库文件并从头开始

## 使用方法

### 基本使用

```bash
# 第一次运行
.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 100

# 如果程序中断，再次运行相同命令会自动继续
.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 100
```

### 强制重新开始

```bash
# 使用 --restart 参数强制重新开始
.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 100 \
    --restart
```

## 进度管理工具

我们还提供了一个专门的工具 `optuna_manager.py` 来管理 Optuna 数据库：

### 查看所有调优进度

```bash
# 查看所有数据库中的studies
.venv/bin/python ts_benchmark/optuna_manager.py --list
```

### 查看特定study的详细信息

```bash
# 查看特定study的详细信息
.venv/bin/python ts_benchmark/optuna_manager.py \
    --show "score_anomalies_PSM_swift_ratio_5.0" \
    --db-path "optuna_studies/score_anomalies_PSM_swift_ratio_5.0.db"
```

### 删除特定study

```bash
# 删除特定study
.venv/bin/python ts_benchmark/optuna_manager.py \
    --delete "score_anomalies_PSM_swift_ratio_5.0" \
    --db-path "optuna_studies/score_anomalies_PSM_swift_ratio_5.0.db"
```

### 清理所有数据库

```bash
# 清理所有数据库文件
.venv/bin/python ts_benchmark/optuna_manager.py --clean-all
```

## 输出示例

### 恢复进度时的输出

```
🚀 开始超参数调优...
🎯 任务: score_anomalies | 数据集: PSM | 算法: swift
🔬 异常率: 5.0% | 试验次数: 100
💾 Study DB 路径: optuna_studies/score_anomalies_PSM_swift_ratio_5.0.db
📊 从数据库恢复了 45 个已完成的试验
🏆 当前最佳分数: 0.887 (Trial 32)
🔄 将继续执行剩余的 55 个试验
```

### 完成所有试验时的输出

```
✅ 所有试验已完成!
🏆 最佳分数: 0.892
```

## 技术细节

### 数据库结构
- 使用 SQLite 数据库存储试验历史
- 每个调优任务对应一个独立的数据库文件
- 支持多个并发调优任务

### Study 命名规则
- Study 名称格式：`{task_name}_{dataset_name}_{algorithm_name}_ratio_{anomaly_ratio}`
- 数据库文件名与 Study 名称对应

### 错误处理
- GPU 内存不足时会自动清理内存并剪枝试验
- 程序异常终止时进度会自动保存到数据库
- 支持 Ctrl+C 中断，进度会保存

## 注意事项

1. **数据库文件位置**：数据库文件保存在 `optuna_studies/` 目录下，已添加到 `.gitignore`
2. **并发安全**：SQLite 数据库支持多进程并发访问
3. **存储空间**：每个数据库文件通常很小（几MB），不会占用太多存储空间
4. **备份建议**：重要的调优结果建议定期备份数据库文件

## 故障排除

### 数据库损坏
如果数据库文件损坏，可以删除对应的 `.db` 文件重新开始：

```bash
rm optuna_studies/score_anomalies_PSM_swift_ratio_5.0.db
```

### 清理所有进度
如果需要清理所有调优进度：

```bash
.venv/bin/python ts_benchmark/optuna_manager.py --clean-all
```

### 查看调优历史
使用管理工具查看详细的调优历史：

```bash
.venv/bin/python ts_benchmark/optuna_manager.py --list
``` 