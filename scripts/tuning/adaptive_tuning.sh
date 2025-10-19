#!/bin/bash

# 自适应 WGCF 超参数调优脚本
# 根据收敛情况自动决定是否继续调优

# 设置参数
TASK_NAME="score_anomalies"
DATASET_NAME="CalIt2"
TRAIN_VAL_LEN=2520
ALGORITHM_NAME="wgcf"
ANOMALY_RATIO=1.0

# 构建study信息
STUDY_NAME="${TASK_NAME}_${DATASET_NAME}_${ALGORITHM_NAME}_ratio_${ANOMALY_RATIO}"
STUDY_DB="optuna_studies/${STUDY_NAME}.db"

# 调优阶段和试验次数
STAGES=(50 100 150 200)
CURRENT_STAGE=0

echo "🚀 开始自适应WGCF超参数调优"
echo "📊 数据集: $DATASET_NAME | 异常率: $ANOMALY_RATIO%"

for TRIALS in "${STAGES[@]}"; do
    CURRENT_STAGE=$((CURRENT_STAGE + 1))
    echo ""
    echo "🔄 阶段 $CURRENT_STAGE: 运行至 $TRIALS 个试验"
    
    # 运行调优
    .venv/bin/python ts_benchmark/tune_params_wgcf.py \
        --task-name $TASK_NAME \
        --dataset-name $DATASET_NAME \
        --train-val-len $TRAIN_VAL_LEN \
        --algorithm-name $ALGORITHM_NAME \
        --anomaly-ratio $ANOMALY_RATIO \
        --n-trials $TRIALS
    
    # 检查是否存在study数据库
    if [ ! -f "$STUDY_DB" ]; then
        echo "❌ Study数据库不存在，停止调优"
        exit 1
    fi
    
    # 生成可视化（如果安装了plotly）
    if python -c "import plotly" 2>/dev/null; then
        echo "📊 生成可视化图表..."
        .venv/bin/python ts_benchmark/visualize_optuna.py \
            --study-db $STUDY_DB \
            --study-name $STUDY_NAME \
            --window-size 20
    fi
    
    echo "✅ 阶段 $CURRENT_STAGE 完成 ($TRIALS 个试验)"
    echo "💡 请查看可视化图表判断是否需要继续："
    echo "   - optuna_visualizations/${DATASET_NAME}/${ALGORITHM_NAME}/optimization_history.html"
    echo "   - optuna_visualizations/${DATASET_NAME}/${ALGORITHM_NAME}/convergence_analysis.html"
    
    # 如果不是最后一个阶段，询问用户是否继续
    if [ $CURRENT_STAGE -lt ${#STAGES[@]} ]; then
        echo ""
        read -p "🤔 是否继续下一阶段调优？(y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏹️ 用户选择停止调优"
            break
        fi
    fi
done

echo ""
echo "🎉 调优完成！"
echo "📁 最终结果保存在: $STUDY_DB"
echo "📊 可视化图表: optuna_visualizations/${DATASET_NAME}/${ALGORITHM_NAME}/"
echo "🏆 查看最佳配置文件: configs/${TASK_NAME}/${DATASET_NAME}/${ALGORITHM_NAME}_*.toml" 