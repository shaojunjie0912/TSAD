import logging
import os
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from .model.swift import SWIFT
from .utils.training import EarlyStopping, get_dataloader, split_data


class SWIFTPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.loss_config = config["loss"]

        self.time_loss_fn = nn.HuberLoss(delta=self.loss_config["time_loss_delta"])  # 时域损失函数
        self.scale_loss_fn = nn.HuberLoss(
            delta=self.loss_config["scale_loss_delta"]
        )  # 尺度域损失函数

        self.ccd_loss_lambda = self.loss_config["ccd_loss_lambda"]
        self.scale_loss_lambda = self.loss_config["scale_loss_lambda"]

        self.anomaly_config = config["anomaly_detection"]
        self.scale_score_lambda = self.anomaly_config["scale_score_lambda"]
        self.anomaly_ratio: float = self.anomaly_config["anomaly_ratio"]

        self.batch_size: int = self.training_config["batch_size"]
        self.seq_len: int = self.data_config["seq_len"]

        # anomaly detection  # NOTE: 保留所有位置 square error
        # TODO: 时间域 + 尺度域异常评分标准
        self.time_anomaly_criterion = nn.MSELoss(reduction="none")
        self.scale_anomaly_criterion = nn.MSELoss(reduction="none")

        self.fitted: bool = False

        # 验证集分数缓存
        self.validation_scores: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None

    # train + val
    def fit(self, train_val_data: np.ndarray):
        print(f"训练+验证集长度: {train_val_data.shape[0]}")
        trial_number = self.config.get("trial_number", None)
        logger = logging.getLogger(f"Trial-{trial_number}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.hasHandlers():
            # 如果是 Optuna 运行，则为每个 trial 创建一个文件
            if trial_number is not None:
                log_dir = "logs/optuna"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"trial_{trial_number}.log")
                file_handler = logging.FileHandler(log_file, mode="w")  # 'w' 模式会覆盖旧日志
            else:  # 如果不是 Optuna 运行，则使用通用日志文件
                log_dir = "logs/normal"
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        train_ratio = self.data_config["train_ratio"]
        len_train = int(len(train_val_data) * train_ratio)

        train_data = train_val_data[:len_train]
        val_data = train_val_data[len_train:]

        # NOTE: 将原始验证集保存为实例属性
        self.val_data = val_data

        self.train_dataloader = get_dataloader(
            stage="train",
            data=train_data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            shuffle=True,
            transform=None,
            target_transform=None,
        )

        self.val_dataloader = get_dataloader(
            stage="val",
            data=val_data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            shuffle=False,
            transform=None,
            target_transform=None,
        )

        fm_config = self.model_config["FM"]
        cfm_config = self.model_config["CFM"]
        tsrm_config = self.model_config["TSRM"]

        self.model = SWIFT(
            # data config
            num_features=train_val_data.shape[1],
            seq_len=self.data_config["seq_len"],
            patch_size=self.data_config["patch_size"],
            patch_stride=self.data_config["patch_stride"],
            # model config
            affine=fm_config["affine"],
            subtract_last=fm_config["subtract_last"],
            level=fm_config["level"],
            wavelet=fm_config["wavelet"],
            mode=fm_config["mode"],
            num_layers=cfm_config["num_layers"],
            dim=cfm_config["d_cf"],
            d_model=cfm_config["d_model"],
            num_heads=cfm_config["num_heads"],
            d_head=cfm_config["d_head"],
            d_ff=cfm_config["d_ff"],
            dropout=cfm_config["dropout"],
            attention_dropout=cfm_config["attention_dropout"],
            num_gat_heads=cfm_config["num_gat_heads"],
            gat_head_dim=cfm_config["gat_head_dim"],
            gat_dropout_rate=cfm_config["gat_dropout_rate"],
            is_flatten_individual=tsrm_config["is_flatten_individual"],
            rec_head_dropout=tsrm_config["rec_head_dropout"],
            # loss config
            ccd_regular_lambda=self.loss_config["ccd_regular_lambda"],
            ccd_align_lambda=self.loss_config["ccd_align_lambda"],
            ccd_align_temperature=self.loss_config["ccd_align_temperature"],
        )
        self.model.to(self.device)

        self.early_stopping = EarlyStopping(
            patience=self.training_config["es_patience"],
            delta=self.training_config["es_delta"],
            mode="min",
            verbose=True,
        )

        train_steps = len(self.train_dataloader)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config["learning_rate"]
        )
        # 学习率调度器
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            epochs=self.training_config["num_epochs"],
            pct_start=self.training_config["pct_start"],
            max_lr=self.training_config["learning_rate"],
        )

        # ----------------------------------------
        # ----------------- 训练 -----------------
        # ----------------------------------------
        # 创建epoch进度条
        epoch_pbar = tqdm(range(self.training_config["num_epochs"]), desc="训练进度", unit="epoch")

        for epoch_idx in epoch_pbar:
            self.model.train()
            train_loss = []

            # 创建batch进度条
            batch_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch_idx+1}", unit="batch", leave=False
            )

            for i, (x, _) in enumerate(batch_pbar):
                self.optimizer.zero_grad()  # 梯度清零

                x = x.float().to(self.device)

                x_orig, x_hat, s, s_hat, ccd_loss = self.model(x)

                # --- 计算总损失 ---
                time_rec_loss = self.time_loss_fn(x_hat, x_orig)
                scale_rec_loss = self.scale_loss_fn(s_hat, s)
                loss = (
                    time_rec_loss
                    + self.scale_loss_lambda * scale_rec_loss
                    + self.ccd_loss_lambda * ccd_loss
                )

                train_loss.append(loss.item())

                # --- 反向传播与更新 ---
                loss.backward()
                self.optimizer.step()

                # 在每个 batch 后更新学习率
                self.scheduler.step()

                # 更新batch进度条
                current_loss = np.mean(train_loss)
                batch_pbar.set_postfix(
                    {
                        "Loss": f"{current_loss:.6f}",
                        "LR": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    }
                )

            # --- Epoch 结束后的验证与打印 ---
            train_loss_avg = np.mean(train_loss)
            valid_loss = self.validate(self.val_dataloader, self.time_loss_fn)

            # 更新epoch进度条
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss_avg:.6f}",
                    "Valid Loss": f"{valid_loss:.6f}",
                    "LR": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

            log_msg = (
                f"Epoch [{epoch_idx+1}/{self.training_config['num_epochs']}], "
                f"Train Loss: {train_loss_avg:.6f}, Valid Loss: {valid_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            logger.info(log_msg)

            self.early_stopping(float(valid_loss), self.model)

            if self.early_stopping.should_stop:
                logger.info("Early stopping triggered. Loading best model weights.")
                self.early_stopping.load_best_weights(self.model)
                break

        self.fitted = True
        # 缓存验证集异常分数
        self.model.eval()
        if self.val_data is not None:
            print("正在缓存验证集异常分数...")
            self.validation_scores = self.score_anomalies(self.val_data)
        self.model.train()

        # 释放文件资源
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def validate(self, val_dataloader, loss_fn):
        self.model.eval()  # -> eval
        total_loss = []
        with torch.no_grad():
            # 创建验证进度条
            val_pbar = tqdm(val_dataloader, desc="验证中", unit="batch", leave=False)
            for x, _ in val_pbar:
                x = x.float().to(self.device)
                x_orig, x_hat, s_orig, s_hat, _ = self.model(x)
                time_rec_loss = loss_fn(x_hat, x_orig)
                scale_rec_loss = loss_fn(s_hat, s_orig)
                loss = time_rec_loss + self.scale_loss_lambda * scale_rec_loss
                total_loss.append(loss.item())

                # 更新验证进度条
                if len(total_loss) > 0:
                    val_pbar.set_postfix({"Valid Loss": f"{np.mean(total_loss):.6f}"})

        self.model.train()  # -> train
        return np.mean(total_loss)

    def _calculate_threshold(self, val_scores: np.ndarray) -> float:
        """自适应阈值策略"""

        # 计算分数的统计特征
        mean_score = np.mean(val_scores)
        std_score = np.std(val_scores)
        skewness = self._calculate_skewness(val_scores)

        # 根据偏度选择策略
        # TODO: 为什么是 1.5?
        if abs(skewness) > 1.5:  # 高偏度，使用鲁棒方法
            # TODO: 为什么是 92.0? 3.0?
            q_robust = 92.0 + min(3.0, float(abs(skewness)))  # 动态调整
            tail_threshold = np.percentile(val_scores, q_robust)
            tail_scores = val_scores[val_scores > tail_threshold]

            if len(tail_scores) > 0:
                # TODO: 为什么是 75.0?
                threshold = np.percentile(tail_scores, 75.0)
            else:
                threshold = np.percentile(val_scores, 100 - self.anomaly_ratio)
        else:  # 低偏度，使用改进的百分位数方法
            # 使用更保守的百分位数
            base_percentile = 100 - self.anomaly_ratio
            # 根据标准差调整
            cv = std_score / (mean_score + 1e-8)  # 变异系数
            # TODO: 为什么是 2.0?
            adjusted_percentile = base_percentile - min(2.0, float(cv * 10))  # 动态调整
            # TODO: 为什么是 90.0?
            threshold = np.percentile(val_scores, max(90.0, adjusted_percentile))

        return float(threshold)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算数据的偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))  # TODO: 为什么是 3?

    def score_anomalies(self, test_data: np.ndarray) -> np.ndarray:
        """异常分数计算"""
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        self.predict_dataloader = get_dataloader(
            stage="test",
            data=test_data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            shuffle=False,
            transform=None,
            target_transform=None,
        )
        self.model.to(self.device)
        self.model.eval()

        anomaly_scores_sum = np.zeros(len(test_data))
        anomaly_scores_max = np.zeros(len(test_data))
        counts = np.zeros(len(test_data))

        with torch.no_grad():
            # 创建异常分数计算进度条
            score_pbar = tqdm(self.predict_dataloader, desc="计算异常分数", unit="batch")
            for i, (x, _, padding_mask, start_indices) in enumerate(score_pbar):
                x = x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                x_orig, x_hat, s_orig, s_hat, _ = self.model(x)

                # 计算时间域和尺度域分数
                time_score = torch.mean(self.time_anomaly_criterion(x_hat, x_orig), dim=-1)
                scale_score = torch.mean(self.scale_anomaly_criterion(s_hat, s_orig), dim=-1)

                score = time_score + self.scale_score_lambda * scale_score
                score_np = score.cpu().numpy()
                padding_mask_np = padding_mask.cpu().numpy()

                for j in range(len(start_indices)):
                    start = start_indices[j]
                    end = start + self.seq_len
                    window_score = score_np[j]
                    window_mask = padding_mask_np[j]

                    actual_end = min(end, len(test_data))
                    valid_length = actual_end - start

                    # 求和聚合
                    anomaly_scores_sum[start:actual_end] += window_score[:valid_length]
                    # 最大值聚合
                    anomaly_scores_max[start:actual_end] = np.maximum(
                        anomaly_scores_max[start:actual_end], window_score[:valid_length]
                    )
                    counts[start:actual_end] += window_mask[:valid_length]

                # 更新进度条信息
                progress_pct = (i + 1) / len(self.predict_dataloader) * 100
                score_pbar.set_postfix({"进度": f"{progress_pct:.1f}%"})

        counts[counts == 0] = 1

        # 均值聚合 + 最大值聚合
        mean_scores = anomaly_scores_sum / counts
        alpha = 0.3  # 均值权重
        beta = 0.7  # 最大值权重
        final_scores = alpha * mean_scores + beta * anomaly_scores_max

        return final_scores

    def find_anomalies(self, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted or self.validation_scores is None:
            raise ValueError("Please fit the model first!")

        print("正在查找异常点...")
        test_scores = self.score_anomalies(test_data)
        threshold = self._calculate_threshold(self.validation_scores)
        predictions = (test_scores > threshold).astype(int)

        return predictions, test_scores


def swift_score_anomalies(all_data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    计算异常分数
    """
    pipeline = SWIFTPipeline(config)
    train_val_data, test_data = split_data(all_data, config["data"]["tain_val_len"])
    pipeline.fit(train_val_data)
    scores = pipeline.score_anomalies(test_data)

    return scores


def swift_find_anomalies(all_data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    找到异常点
    """
    pipeline = SWIFTPipeline(config)
    train_val_data, test_data = split_data(all_data, config["data"]["tain_val_len"])
    pipeline.fit(train_val_data)
    predictions, _ = pipeline.find_anomalies(test_data)

    return predictions
