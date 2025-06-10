from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .model.catch import CATCH
from .utils.data import Normalizer, get_dataloader, get_train_val_dataloader
from .utils.tools import EarlyStopping, adjust_learning_rate


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
        self.time_anomaly_criterion = nn.MSELoss(reduction="none")

        self.fitted: bool = False

    # train + val
    def fit(self, data: np.ndarray):
        # Enable anomaly detection during debugging
        torch.autograd.set_detect_anomaly(True)
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
            patience=self.training_config["patience"],
            verbose=True,
        )

        train_steps = len(self.train_dataloader)
        main_params = [
            param for name, param in self.model.named_parameters() if "mask_generator" not in name
        ]
        mask_params = self.model.masker.parameters()

        # 创建优化器 (NOTE: learning rate 会在 OneCycleLR 中更新)
        self.main_optimizer = torch.optim.Adam(
            main_params, lr=self.training_config["learning_rate"]
        )
        self.mask_optimizer = torch.optim.Adam(
            mask_params, lr=self.training_config["mask_learning_rate"]
        )

        # 创建学习率调度器
        # HACK: 100 只是占位, trainer 中会更新
        # NOTE: steps_per_epoch 不指定则相当于 num_batch
        main_scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.main_optimizer,
            steps_per_epoch=train_steps,
            epochs=self.training_config["num_epochs"],
            pct_start=self.training_config["pct_start"],
            max_lr=self.training_config["learning_rate"],
        )
        mask_scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.mask_optimizer,
            steps_per_epoch=train_steps,
            epochs=self.training_config["num_epochs"],
            pct_start=self.training_config["pct_start"],
            max_lr=self.training_config["mask_learning_rate"],
        )

        # 开始手写训练循环
        for epoch_idx in range(self.training_config["num_epochs"]):
            self.model.train()  #  将模型设置为训练模式
            train_loss = []
            step = min(int(train_steps / 10), 100)  #
            for i, (x, _) in enumerate(self.train_dataloader):
                self.main_optimizer.zero_grad()  # 主优化器梯度清零

                x = x.float().to(self.device)

                x_hat, s, s_hat, ccd_loss = self.model(x)

                # ------------- 计算损失 -------------

                # 时域重建损失
                time_rec_loss = self.time_loss_fn(x_hat, x)

                # 尺度域重建损失
                scale_rec_loss = self.scale_loss_fn(s_hat, s)

                # 总损失 = 时域重建损失 + 频域重建损失 + 动态对比损失
                loss = (
                    time_rec_loss
                    + self.scale_loss_lambda * scale_rec_loss
                    + self.ccd_loss_lambda * ccd_loss
                )
                train_loss.append(loss.item())

                if (i + 1) % step == 0:
                    self.mask_optimizer.step()  # 更新 mask 优化器
                    self.mask_optimizer.zero_grad()  # mask 优化器梯度清零

                if (i + 1) % step == 100:
                    print(
                        f"Epoch [{epoch_idx+1}/{self.training_config['num_epochs']}], "
                        f"Batch [{i+1}/{len(self.train_dataloader)}], "
                        f"time rec loss: {time_rec_loss.item():.4f}, "
                        f"scale rec loss: {scale_rec_loss.item():.4f}, "
                        f"ccd loss: {ccd_loss.item():.4f}"
                    )

                loss.backward()
                self.main_optimizer.step()

            train_loss = np.mean(train_loss)
            valid_loss = self.validate(self.val_dataloader, self.time_loss_fn)
            print(
                f"Epoch [{epoch_idx+1}/{self.training_config['num_epochs']}], "
                f"Train loss: {train_loss:.4f}, "
                f"Valid loss: {valid_loss:.4f}"
            )

            # 早停
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                self.main_optimizer,
                mask_scheduler,
                epoch_idx + 1,
                self.training_config["lr_adj"],
                self.training_config["learning_rate"],
            )
            adjust_learning_rate(
                self.mask_optimizer,
                main_scheduler,
                epoch_idx + 1,
                self.training_config["lr_adj"],
                self.training_config["mask_learning_rate"],  # TODO: 原 CACTH 中是 learning_rate
            )
        self.fitted = True

    def validate(self, val_dataloader, loss_fn):
        total_loss = []
        self.model.eval()  # NOTE: 将模型设置为评估模式

        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.float().to(self.device)
                x_hat, _, _, _ = self.model(x)
                # TODO: 验证时只看时域重构损失?
                loss = loss_fn(x_hat, x).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()  # NOTE: 重设回训练模式
        return total_loss

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
        self.model.load_state_dict(self.early_stopping.check_point)
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

        self.model.load_state_dict(self.early_stopping.check_point)

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
