#!/bin/bash

# 示例：WGCF 超参数调优 + 可视化分析

echo "🚀 开始WGCF超参数调优（包含可视化分析）"

# 设置参数
TASK_NAME="score_anomalies"
DATASET_NAME="CalIt2"
TRAIN_VAL_LEN=2520
ALGORITHM_NAME="wgcf"
ANOMALY_RATIO=1.0
N_TRIALS=50

# 第一步：运行调优
echo "📊 第一步：运行超参数调优"
.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name $TASK_NAME \
    --dataset-name $DATASET_NAME \
    --train-val-len $TRAIN_VAL_LEN \
    --algorithm-name $ALGORITHM_NAME \
    --anomaly-ratio $ANOMALY_RATIO \
    --n-trials $N_TRIALS

# 第二步：生成可视化图表
echo ""
echo "📊 第二步：生成可视化图表"

# 构建study名称和数据库路径
STUDY_NAME="${TASK_NAME}_${DATASET_NAME}_${ALGORITHM_NAME}_ratio_${ANOMALY_RATIO}"
STUDY_DB="optuna_studies/${STUDY_NAME}.db"

# 检查是否安装了plotly
if python -c "import plotly" 2>/dev/null; then
    echo "✅ plotly 已安装，生成可视化图表..."
    .venv/bin/python ts_benchmark/visualize_optuna.py \
        --study-db $STUDY_DB \
        --study-name $STUDY_NAME
else
    echo "⚠️ plotly 未安装，跳过可视化图表生成"
    echo "💡 要启用可视化功能，请运行: pip install plotly"
fi

echo ""
echo "✅ 调优和可视化完成！"
echo "📁 调优结果保存在: $STUDY_DB"
echo "📊 可视化图表保存在: optuna_visualizations/${DATASET_NAME}/${ALGORITHM_NAME}/"
echo ""
echo "💡 使用建议："
echo "   1. 查看 optimization_history.html 了解优化过程"
echo "   2. 查看 param_importances.html 了解重要参数"
echo "   3. 查看 convergence_analysis.html 判断是否需要更多试验"
echo "   4. 如果未收敛，可以增加 --n-trials 继续调优" 