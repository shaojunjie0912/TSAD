from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .model.catch import CATCH
from .utils.data import Normalizer, get_dataloader, get_train_val_dataloader
from .utils.fre_rec_loss import FrequencyCriterion, FrequencyLoss
from .utils.tools import EarlyStopping, adjust_learning_rate

# TODO: CATCH 学习率的动态调整 adjust_lr? 那 OneCycleLR 还需要吗?
# 而且原来代码中的 ajust_lr 直接传入了 config.lr, 而不是分别 lr 和 mask_lr
# 导致 main_optimizer 和 mask_optimizer 的学习率一样了
# NOTE: 不用 lightning 版本了, 记得 to(device)
# TODO: 有关 epoch 花费多少时间懒得写了
# TODO: 验证一下 CATCH 中的 thre 是否是不重叠窗口
# TODO: Transformer 需要大量数据?

# NOTE: CATCH 根据数据集元信息 csv 来分割整个数据集用以训练和测试
# CATCH 最后一个不足窗口大小的部分会被丢弃
# TODO: Maybe 得写一个通用数据处理模块?
# TODO: 设计模式: 工厂方法, 测试多个算法, 用相同接口?
# TODO: 将模型训练, 数据加载从主逻辑分离


class CATCHPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_config = config["model"]
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.loss_config = config["loss"]

        self.time_loss_fn = nn.MSELoss()  # 时域损失函数
        self.freq_loss_fn = FrequencyLoss(  # 频域损失函数
            fft_mode=self.loss_config["fft_mode"],
            complex_error_type=self.loss_config["complex_error_type"],
            loss_type=self.loss_config["loss_type"],
            module_first=self.loss_config["module_first"],
        )
        self.dc_lambda = self.loss_config["dc_lambda"]
        self.freq_loss_lambda = self.loss_config["freq_loss_lambda"]
        self.freq_score_lambda = self.config["anomaly_detection"]["freq_score_lambda"]
        self.anomaly_ratio: Union[List[float], float] = self.config["anomaly_detection"][
            "anomaly_ratio"
        ]

        self.batch_size: int = self.training_config["batch_size"]
        self.seq_len: int = self.data_config["seq_len"]

        # anomaly detection
        self.time_anomaly_criterion = nn.MSELoss(
            reduction="none"
        )  # NOTE: 保留所有位置 square error
        self.freq_anomaly_criterion = FrequencyCriterion(
            fft_mode=self.loss_config["fft_mode"],
            complex_error_type=self.loss_config["complex_error_type"],
            loss_type=self.loss_config["loss_type"],
            inference_patch_size=self.data_config["inference_patch_size"],
            inference_patch_stride=self.data_config["inference_patch_stride"],
            seq_len=self.data_config["seq_len"],
            module_first=self.loss_config["module_first"],
        )

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

        # TODO: CATCH 中的 train_data 输入没有时间列?
        num_features = data.shape[1]

        # TODO: 参数配置简化
        self.model = CATCH(
            num_features=num_features,  # TODO: 放配置文件?
            # model config
            num_layers=self.model_config["num_layers"],
            dim=self.model_config["d_cf"],
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            d_head=self.model_config["d_head"],
            d_ff=self.model_config["d_ff"],
            flatten_individual=self.model_config["flatten_individual"],
            dropout=self.model_config["regularization"]["dropout"],
            head_dropout=self.model_config["regularization"]["head_dropout"],
            regular_lambda=self.model_config["regularization"]["regular_lambda"],
            temperature=self.model_config["regularization"]["temperature"],
            # data config
            patch_size=self.data_config["patch_size"],
            patch_stride=self.data_config["patch_stride"],
            seq_len=self.data_config["seq_len"],
            affine=self.data_config["normalization"]["affine"],
            subtract_last=self.data_config["normalization"]["subtract_last"],
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(
            patience=self.training_config["patience"],
            verbose=True,
        )

        train_steps = len(self.train_dataloader)
        main_params = [
            param for name, param in self.model.named_parameters() if "mask_generator" not in name
        ]
        mask_params = self.model.mask_generator.parameters()

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
            train_loss = []
            step = min(int(train_steps / 10), 100)  #
            for i, (x, _) in enumerate(self.train_dataloader):
                self.main_optimizer.zero_grad()  # 主优化器梯度清零

                x = x.float().to(self.device)
                self.model.train()  # TODO: 将模型设置为训练模式
                x_hat, z_hat, dc_loss = self.model(x)  # NOTE: dc_loss: 动态对比损失

                # ------------- 计算损失 -------------

                # 时域重建损失 (MSE)
                time_rec_loss = self.time_loss_fn(x_hat, x)
                # 频域重建损失 (包含实虚部的 fft + 直接计算复数频谱差异 complex + 平均绝对误差 MAE)
                z_insnorm = self.model.revin_layer(
                    x, "transform"
                )  # 实例归一化(使用存储的均值和标准差)
                freq_rec_loss = self.freq_loss_fn(z_hat, z_insnorm)  # (重构值, 真实值)

                # 总损失 = 时域重建损失 + 频域重建损失 + 动态对比损失
                loss = (
                    time_rec_loss + self.freq_loss_lambda * freq_rec_loss + self.dc_lambda * dc_loss
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
                        f"freq rec loss: {freq_rec_loss.item():.4f}, "
                        f"dc loss: {dc_loss.item():.4f}"
                    )

                loss.backward()
                self.main_optimizer.step()

            train_loss = np.mean(train_loss)  # TODO: 原: average 可加权
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
        self.model.eval()  # TODO: 将模型设置为评估模式

        with torch.no_grad():
            for z, _ in val_dataloader:
                z = z.float().to(self.device)
                z_hat, _, _ = self.model(z)
                z_hat = z_hat.detach().cpu()
                true = z.detach().cpu()
                # TODO: 验证时只看时域重构损失?
                loss = loss_fn(z_hat, true).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()  # TODO: 重设回训练模式
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


def catch_anomaly_score(data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
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
