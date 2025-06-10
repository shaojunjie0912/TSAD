from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .model.catch import CATCH
from .utils.data import Normalizer, get_dataloader, get_train_val_dataloader
from .utils.early_stopping import EarlyStopping


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
        self.freq_score_lambda = self.anomaly_config["freq_score_lambda"]
        self.anomaly_ratio: Union[List[float], float] = self.anomaly_config["anomaly_ratio"]

        self.batch_size: int = self.training_config["batch_size"]
        self.seq_len: int = self.data_config["seq_len"]

        # anomaly detection  # NOTE: 保留所有位置 square error
        # TODO: 时间域 + 尺度域异常评分标准
        self.time_anomaly_criterion = nn.MSELoss(reduction="none")
        self.freq_anomaly_criterion = nn.MSELoss(reduction="none")

        self.fitted: bool = False

    # train + val
    def fit(self, data: np.ndarray):
        self.normalizer = Normalizer()
        self.normalizer.fit(data)
        self.transform = self.normalizer.as_transform()
        self.train_dataloader, self.val_dataloader = get_train_val_dataloader(
            data=data,
            train_rate=0.8,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            transform=self.transform,
            target_transform=self.transform,  # NOTE: 重构 X=Y 因此相同处理
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

                x_hat, s, s_hat, ccd_loss = self.model(x)

                # --- 计算总损失 ---
                time_rec_loss = self.time_loss_fn(x_hat, x)
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
                x_hat, _, _, _ = self.model(x)
                loss = loss_fn(x_hat, x).item()  # TODO: 验证时只看时域重构损失?
                total_loss.append(loss)
        self.model.train()  # NOTE: 重设回训练模式
        return np.mean(total_loss)

    def score_anomalies(self, data: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): 测试数据

        Raises:
            ValueError: _description_

        Returns:
            np.ndarray: 测试数据每个时刻的异常评分
        """
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        # 非重叠窗口
        self.predict_dataloader = get_dataloader(
            stage="predict",  # NOTE: 非重叠窗口
            data=data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=self.seq_len,
            shuffle=False,
            transform=self.transform,
            target_transform=self.transform,  # NOTE: 重构 X=Y 因此相同处理
        )
        self.model.to(self.device)
        self.model.eval()

        batch_anomaly_scores = []
        with torch.no_grad():
            for i, (x, _, padding_mask) in enumerate(self.predict_dataloader):
                x = x.float().to(self.device)  # (batch_size, seq_len, num_features)
                padding_mask = padding_mask.float().to(self.device)  # (batch_size, seq_len)
                x_hat, _, _ = self.model(x)  # (batch_size, seq_len, num_features)

                # (batch_size, seq_len) 特征维度上取均值(TODO: 统一时刻所有变量都被判为异常?)
                time_score = torch.mean(self.time_anomaly_criterion(x_hat, x), dim=-1)
                # TODO: 时域重构结果放进频域异常评判标准?
                freq_score = torch.mean(self.freq_anomaly_criterion(x_hat, x), dim=-1)

                # Apply padding mask to zero out the scores from padded regions
                score = (time_score + self.freq_score_lambda * freq_score) * padding_mask
                score = score.detach().cpu().numpy()
                batch_anomaly_scores.append(score)

                print(
                    f"Batch [{i+1}/{len(self.predict_dataloader)}], "
                    f"testing time loss: {time_score.detach().cpu().numpy()[0, :5]}, "
                    f"testing fre loss: {freq_score.detach().cpu().numpy()[0, :5]}"
                )

        # shape (batch_size * seq_len,) 虽然是非重叠窗口
        # 但也不一定等于 test_data 的 num_samples? TODO: 考虑对最后一个窗口 padding 还是丢弃?
        anomaly_scores = np.concatenate(batch_anomaly_scores, axis=0).reshape(-1)
        anomaly_scores = np.array(anomaly_scores)[: len(data)]

        # 打印最大值, 最小值, 均值
        print(
            f"Anomaly scores: max: {np.max(anomaly_scores)}, min: {np.min(anomaly_scores)}, mean: {np.mean(anomaly_scores)}"
        )
        return anomaly_scores

    def find_anomalies(self, data: np.ndarray):
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        self.test_dataloader = get_dataloader(
            stage="test",
            data=data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            shuffle=False,
            transform=self.transform,
            target_transform=self.transform,
        )

        self.predict_dataloader = get_dataloader(
            stage="predict",
            data=data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=self.seq_len,  # NOTE: 非重叠窗口
            shuffle=False,
            transform=self.transform,
            target_transform=self.transform,
        )

        self.model.to(self.device)
        self.model.eval()

        # --------------- 阈值计算(基于训练集和测试集) ---------------
        attens_energy = []

        with torch.no_grad():
            for z, _ in self.train_dataloader:  # NOTE: train_dataloader
                z = z.float().to(self.device)
                z_hat, _, _ = self.model(z)

                time_score = torch.mean(self.time_anomaly_criterion(z_hat, z), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(z_hat, z), dim=-1)

                score = (time_score + self.freq_score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

        # (batch_size * seq_len,) 远大于 train_data 的 num_samples, 因为步长为 1
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_energy = []

        with torch.no_grad():
            for z, _ in self.test_dataloader:  # NOTE: test_dataloader
                z = z.float().to(self.device)
                z_hat, _, _ = self.model(z)

                time_score = torch.mean(self.time_anomaly_criterion(z_hat, z), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(z_hat, z), dim=-1)

                score = (time_score + self.freq_score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

        # (batch_size * seq_len,)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        with torch.no_grad():
            for i, (z, _, padding_mask) in enumerate(self.predict_dataloader):
                z = z.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                z_hat, _, _ = self.model(z)

                time_score = torch.mean(self.time_anomaly_criterion(z_hat, z), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(z_hat, z), dim=-1)

                score = (time_score + self.freq_score_lambda * freq_score) * padding_mask
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

                print(
                    f"Batch [{i+1}/{len(self.predict_dataloader)}], "
                    f"testing time loss: {time_score.detach().cpu().numpy()[0, :5]}, "
                    f"testing fre loss: {freq_score.detach().cpu().numpy()[0, :5]}"
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        predict_energy = np.array(attens_energy)

        if not isinstance(self.anomaly_ratio, List):
            self.anomaly_ratio = [self.anomaly_ratio]

        predictions = {}
        for ratio in self.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            predictions[ratio] = np.where(predict_energy > threshold, 1, 0)

        # Remove padding from predictions
        for ratio in self.anomaly_ratio:
            predictions[ratio] = predictions[ratio][: len(data)]

        return predictions


def catch_score_anomalies(data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    计算异常分数
    """
    pipeline = CATCHPipeline(config)
    pipeline.fit(data)
    scores = pipeline.score_anomalies(data)

    return scores


def catch_find_anomalies(data: np.ndarray, config: Dict[str, Any]) -> Dict[float, np.ndarray]:
    """
    找到异常点
    """
    pipeline = CATCHPipeline(config)
    pipeline.fit(data)
    predictions = pipeline.find_anomalies(data)

    return predictions
