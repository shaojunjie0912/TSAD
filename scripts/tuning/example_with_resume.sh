#!/bin/bash

# 示例：使用进度保存和恢复功能的调优脚本

echo "🚀 开始WGCF超参数调优（支持进度保存和恢复）"

# 第一次运行 - 进行50个试验
echo "📊 第一次运行：50个试验"
.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name score_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name wgcf \
    --anomaly-ratio 5.0 \
    --n-trials 50

echo ""
echo "💾 查看当前进度："
.venv/bin/python ts_benchmark/optuna_manager.py --list

echo ""
echo "🔄 继续运行更多试验（会自动从上次停止的地方继续）"
.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name score_anomalies \
    --dataset-name PSM \
    --train-val-len 58317 \
    --algorithm-name wgcf \
    --anomaly-ratio 5.0 \
    --n-trials 100  # 总共100个试验，会继续执行剩余的50个

echo ""
echo "📈 最终进度："
.venv/bin/python ts_benchmark/optuna_manager.py --list

echo ""
echo "✅ 调优完成！"
echo "💡 如果想重新开始，可以使用 --restart 参数" 