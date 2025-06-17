from typing import Any, Dict, Literal, Optional

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

        # 验证集分数缓存
        self.validation_scores: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config["learning_rate"])

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
                    time_rec_loss + self.scale_loss_lambda * scale_rec_loss + self.ccd_loss_lambda * ccd_loss
                )

                # # ---- 添加这行用于诊断 ----
                # if i % 100 == 0:  # 每100个batch打印一次
                #     print(
                #         f"batch {i}: time_loss={time_rec_loss.item():.4f}, "
                #         f"scale_loss={scale_rec_loss.item():.4f}, "
                #         f"ccd_loss={ccd_loss.item():.4f}"
                #     )

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

            self.early_stopping(float(valid_loss), self.model)

            if self.early_stopping.should_stop:
                print("Early stopping triggered. Loading best model weights.")
                # 在中断循环前，自动加载性能最佳的模型权重
                self.early_stopping.load_best_weights(self.model)
                break

        self.fitted = True
        # 在训练结束后, 计算并缓存验证集分数
        print("\nCalculating and caching validation scores for future use...")
        self.model.eval()
        if self.val_data is not None:
            self.validation_scores = self.score_anomalies(self.val_data)
        self.model.train()
        print("Fitting process complete. Validation scores are now cached.")

    def validate(self, val_dataloader, loss_fn):
        self.model.eval()  # -> eval
        total_loss = []
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.float().to(self.device)
                x_orig, x_hat, s_orig, s_hat, _ = self.model(x)
                time_rec_loss = loss_fn(x_hat, x_orig)
                scale_rec_loss = loss_fn(s_hat, s_orig)
                loss = time_rec_loss + self.scale_loss_lambda * scale_rec_loss
                total_loss.append(loss.item())
        self.model.train()  # -> train
        return np.mean(total_loss)

    def score_anomalies(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        self.predict_dataloader = get_dataloader(
            stage="predict",
            data=data,
            batch_size=self.batch_size,
            window_size=self.seq_len,
            step_size=1,
            shuffle=False,
            transform=None,
            target_transform=None,
        )
        self.model.to(self.device)
        self.model.eval()

        anomaly_scores_sum = np.zeros(len(data))
        counts = np.zeros(len(data))

        with torch.no_grad():
            for i, (x, _, padding_mask, start_indices) in enumerate(self.predict_dataloader):
                x = x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                x_orig, x_hat, s_orig, s_hat, _ = self.model(x)

                # -> (B, T)
                # 特征维度上取均值 NOTE: 统一时刻所有变量都被判为异常
                time_score = torch.mean(self.time_anomaly_criterion(x_hat, x_orig), dim=-1)  # 时间域
                scale_score = torch.mean(self.scale_anomaly_criterion(s_hat, s_orig), dim=-1)  # 尺度域

                score = time_score + self.scale_score_lambda * scale_score
                score_np = score.cpu().numpy()
                padding_mask_np = padding_mask.cpu().numpy()

                for j in range(len(start_indices)):
                    start = start_indices[j]
                    end = start + self.seq_len
                    window_score = score_np[j]
                    window_mask = padding_mask_np[j]

                    actual_end = min(end, len(data))
                    anomaly_scores_sum[start:actual_end] += window_score[: actual_end - start]
                    counts[start:actual_end] += window_mask[: actual_end - start]

        counts[counts == 0] = 1
        final_scores = anomaly_scores_sum / counts
        return final_scores

    def _calculate_threshold(
        self,
        val_scores: np.ndarray,
        strategy: Literal["percentile", "robust_percentile", "std"] = "robust_percentile",
        **kwargs,
    ) -> float:
        """根据不同策略计算异常阈值"""
        print(f"Calculating threshold using '{strategy}' strategy...")

        if strategy == "percentile":
            # 百分位数
            threshold = np.percentile(val_scores, 100 - self.anomaly_ratio)

        elif strategy == "robust_percentile":
            # 鲁棒百分位数
            # q_robust: 用于识别“尾部”数据的分位数，例如 99.0 表示我们关注分数最高的1%数据。
            q_robust = kwargs.get("q_robust", 99.0)
            # p_robust: 在上面识别出的“尾部”数据中，再取一个百分位作为最终阈值。
            # 例如 90.0 表示我们将尾部数据中较低的90%部分作为正常值的极限，切掉了尾部最顶端的10%。
            p_robust = kwargs.get("p_robust", 90.0)

            # 1. 找到初始的高分位数 q，确定“尾部”的起点
            tail_threshold = np.percentile(val_scores, q_robust)

            # 2. 筛选出所有高于这个起点的分数，得到“尾部”数据分布
            tail_scores = val_scores[val_scores > tail_threshold]

            if len(tail_scores) == 0:
                # 如果尾部没有数据（可能验证集非常“干净”且分布集中），则退回到使用初始的尾部阈值
                print(
                    f"  (Warning: No scores found above the {q_robust}th percentile. Using tail_threshold as fallback.)"
                )
                return float(tail_threshold)

            # 3. 在“尾部”数据中，使用 p_robust 计算最终阈值
            final_threshold = np.percentile(tail_scores, p_robust)

            print(
                f"  (Robust params: tail defined by q={q_robust}th percentile, final threshold at p={p_robust}th percentile of the tail)"
            )
            threshold = final_threshold

        elif strategy == "std":
            # 标准差策略
            n_std = kwargs.get("n_std", 3.0)
            mean = np.mean(val_scores)
            std = np.std(val_scores)
            threshold = mean + n_std * std
            print(f"  (STD params: mean={mean:.4f}, std={std:.4f}, n_std={n_std})")

        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        return float(threshold)

    def find_anomalies(
        self,
        data: np.ndarray,
        threshold_strategy: Literal["percentile", "robust_percentile", "std"] = "robust_percentile",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Please fit the model first!")

        print("Scoring anomalies on the test data...")
        test_scores = self.score_anomalies(data)

        if self.validation_scores is None:
            raise RuntimeError("Validation scores were not cached. Please check the fit() method.")

        threshold = self._calculate_threshold(self.validation_scores, strategy=threshold_strategy, **kwargs)
        print(f"Anomaly threshold determined: {threshold:.6f}")

        predictions = (test_scores > threshold).astype(int)

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
