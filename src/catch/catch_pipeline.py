from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .model.catch import CATCH
from .utils.training import EarlyStopping, get_dataloader


class CATCHPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.loss_config = config["loss"]

        self.time_loss_fn = nn.HuberLoss(delta=1.0)  # 时域损失函数 TODO: 超参数
        self.scale_loss_fn = nn.HuberLoss(delta=1.0)  # 尺度域损失函数

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

    # train + val
    def fit(self, data: np.ndarray):
        train_ratio = self.data_config["train_ratio"]
        len_train = int(len(data) * train_ratio)

        train_data = data[:len_train]
        val_data = data[len_train:]

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

        self.model = CATCH(
            # data config
            num_features=data.shape[1],
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

        # NOTE: 一个优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config["learning_rate"]
        )

        # NOTE: 一个调度器
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
        for epoch_idx in range(self.training_config["num_epochs"]):
            self.model.train()
            train_loss = []

            for i, (x, _) in enumerate(self.train_dataloader):
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

            # --- Epoch 结束后的验证与打印 ---
            train_loss_avg = np.mean(train_loss)
            valid_loss = self.validate(self.val_dataloader, self.time_loss_fn)
            print(
                f"Epoch [{epoch_idx+1}/{self.training_config['num_epochs']}], "
                f"Train Loss: {train_loss_avg:.6f}, Valid Loss: {valid_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if self.early_stopping.should_stop:
                print("Early stopping triggered. Loading best model weights.")
                # 在中断循环前，自动加载性能最佳的模型权重
                self.early_stopping.load_best_weights(self.model)
                break

        self.fitted = True

    def validate(self, val_dataloader, loss_fn):
        self.model.eval()  # NOTE: 将模型设置为评估模式
        total_loss = []
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.float().to(self.device)
                x_orig, x_hat, _, _, _ = self.model(x)
                loss = loss_fn(x_hat, x_orig).item()  # TODO: 验证时只看时域重构损失?
                total_loss.append(loss)
        self.model.train()  # NOTE: 重设回训练模式
        return np.mean(total_loss)

    def score_anomalies(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        self.predict_dataloader = get_dataloader(
            stage="predict",  # NOTE: 窗口不重叠
            data=data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=self.seq_len,
            shuffle=False,
            transform=None,
            target_transform=None,
        )
        self.model.to(self.device)
        self.model.eval()

        batch_anomaly_scores = []
        with torch.no_grad():
            for i, (x, _, padding_mask) in enumerate(self.predict_dataloader):
                x = x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                x_orig, x_hat, s_orig, s_hat, _ = self.model(x)

                # (batch_size, seq_len) 特征维度上取均值 NOTE: 统一时刻所有变量都被判为异常
                # 时间域分数
                time_score = torch.mean(self.time_anomaly_criterion(x_hat, x_orig), dim=-1)

                # 尺度域分数
                scale_score = torch.mean(self.scale_anomaly_criterion(s_hat, s_orig), dim=-1)

                score = (time_score + self.scale_score_lambda * scale_score) * padding_mask

                batch_anomaly_scores.append(score.cpu().numpy())

        anomaly_scores = np.concatenate(batch_anomaly_scores, axis=0).reshape(-1)
        return anomaly_scores[: len(data)]

    def find_anomalies(self, data: np.ndarray):
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        # 1. 获取测试数据的异常分数
        print("Scoring anomalies on the test data...")
        test_scores = self.score_anomalies(data)

        # 2. 高效地计算阈值
        # 不再遍历整个训练集，而是使用验证集的分数来估计正常分数的分布。
        # 验证集未经训练，可以较好地代表正常数据的分布。
        print("Calculating threshold on the validation data...")
        val_scores = self.score_anomalies(self.val_data)

        # 3. 确定阈值并进行预测
        threshold = np.percentile(val_scores, 100 - self.anomaly_ratio)
        print(f"Anomaly threshold determined: {threshold:.6f}")

        predictions = (test_scores > threshold).astype(int)

        # 同时返回预测的0/1标签和每个点的具体分数，方便后续评估（如计算AUC）
        return predictions, test_scores


def catch_score_anomalies(data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    计算异常分数
    """
    pipeline = CATCHPipeline(config)
    pipeline.fit(data)
    scores = pipeline.score_anomalies(data)

    return scores


def catch_find_anomalies(data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    找到异常点
    """
    pipeline = CATCHPipeline(config)
    pipeline.fit(data)
    predictions, scores = pipeline.find_anomalies(data)

    return predictions
