"""工具模块

- 数据加载器
- 早停策略
"""

import copy
import json
import logging
import os
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def get_dataloader(
    stage: str,
    data: np.ndarray,
    batch_size: int,
    window_size: int,
    step_size: int,
    shuffle: bool,
    padding_value: float = 0.0,
    transform: Optional[Any] = None,
    target_transform: Optional[Any] = None,
):
    # predict 阶段需要 padding!!
    if stage == "predict":
        dataset = SlidingWindowDatasetWithPadding(
            data=data,
            window_size=window_size,
            step_size=step_size,
            padding_value=padding_value,
            transform=transform,
            target_transform=target_transform,
        )
        drop_last = False
    else:
        dataset = SlidingWindowDatasetNoPadding(
            data=data,
            window_size=window_size,
            step_size=step_size,
            transform=transform,
            target_transform=target_transform,
        )
        drop_last = True if stage == "train" else False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# Padding Dataset: 用于预测
class SlidingWindowDatasetWithPadding(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        step_size: int,
        padding_value: float,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.step_size = step_size
        self.padding_value = padding_value
        self.transform = transform
        self.target_transform = target_transform
        self.windows, self.masks, self.starts = self.create_windows()

    def create_windows(self):
        num_samples, num_features = self.data.shape
        windows, padding_masks, starts = [], [], []

        for start in range(0, num_samples, self.step_size):
            end = start + self.window_size
            window = self.data[start:end]

            pad_len = max(0, end - num_samples)
            if pad_len > 0:
                pad_array = np.full((pad_len, num_features), self.padding_value, dtype=np.float32)
                window = np.concatenate([window, pad_array], axis=0)

            padding_mask = np.ones(self.window_size, dtype=np.float32)
            if pad_len > 0:
                padding_mask[-pad_len:] = 0.0

            windows.append(window)
            padding_masks.append(padding_mask)
            starts.append(start)

            if end >= num_samples:
                break

        return np.stack(windows), np.stack(padding_masks), np.array(starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = x.copy()
        padding_mask = self.masks[idx]
        start_idx = self.starts[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        # NOTE:
        # X: (batch_size, window_size, num_features)
        # Y: (batch_size, window_size, num_features)
        # mask: (batch_size, window_size) 对应 window_size 中每个时间步是否被填充
        # start_idx: (batch_size) 对应每个窗口的起始索引
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(padding_mask), start_idx


# NoPadding Dataset: 用于训练、验证、测试
class SlidingWindowDatasetNoPadding(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        step_size: int = 1,
        transform=None,
        target_transform=None,
    ):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.target_transform = target_transform
        self.windows = self.create_windows()

    def create_windows(self):
        num_samples = self.data.shape[0]
        windows = []
        for start in range(0, num_samples - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(self.data[start:end])
        return np.stack(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = x.copy()

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return torch.from_numpy(x), torch.from_numpy(y)


class Normalizer:
    def __init__(self, eps: float = 1e-6):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, data: Union[np.ndarray, torch.Tensor]):
        """
        计算并存储每个特征维度的 mean 和 std
        Args:
            data (np.ndarray or torch.Tensor): shape (num_samples, num_features)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data: Union[np.ndarray, torch.Tensor]):
        """
        标准化数据
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("You must call fit() before transform().")

        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.tensor(self.std, dtype=data.dtype, device=data.device)
            return (data - mean) / (std + self.eps)
        else:
            return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]):
        """
        将标准化后的数据还原为原始值
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("You must call fit() before inverse_transform().")

        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.tensor(self.std, dtype=data.dtype, device=data.device)
            return data * (std + self.eps) + mean
        else:
            return data * (self.std + self.eps) + self.mean

    def as_transform(self):
        """
        返回可用于 PyTorch Dataset 的 transform 函数
        """
        return lambda x: self.transform(x)

    def save(self, path):
        """
        保存 mean 和 std 到 JSON 文件
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("You must call fit() before save().")
        stats = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        with open(path, "w") as f:
            json.dump(stats, f)

    def load(self, path):
        """
        从 JSON 文件加载 mean 和 std
        """
        with open(path, "r") as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)


class EarlyStopping:
    """早停策略类

    Args:
        patience: 连续未改善的轮数
        delta: 最小变化量
        mode: "min" (较小更好) 或 "max" (较大更好)
        ckpt_path: 保存最佳权重的文件路径。如果为 None，则保存在内存中。
        verbose: 是否记录改善消息。
    """

    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        ckpt_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.verbose = verbose

        # 内部状态
        self._best_score: Optional[float] = None
        self._counter: int = 0
        self._early_stop: bool = False
        self._best_state_dict: Optional[dict] = None

    # --------------------------------------------------------------------- #
    @property
    def should_stop(self) -> bool:
        """是否应该停止训练"""
        return self._early_stop

    # --------------------------------------------------------------------- #
    def __call__(self, metric: float, model: torch.nn.Module) -> None:
        """更新早停逻辑

        Args:
            metric: 验证集上监控指标的值。
            model: 当前模型。如果性能提高，则保存。
        """
        score = metric if self.mode == "max" else -metric

        if self._best_score is None:
            self._save_checkpoint(metric, model)
            self._best_score = score
            return

        if score < self._best_score + self.delta:
            self._counter += 1
            if self.verbose:
                logging.info(
                    "No improvement (%.5f). Counter %d/%d",
                    metric,
                    self._counter,
                    self.patience,
                )
            if self._counter >= self.patience:
                self._early_stop = True
        else:
            self._save_checkpoint(metric, model)
            self._best_score = score
            self._counter = 0

    # --------------------------------------------------------------------- #
    def _save_checkpoint(self, metric: float, model: torch.nn.Module) -> None:
        """保存当前最佳模型到内存或磁盘"""
        if self.ckpt_path:
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            self._best_state_dict = copy.deepcopy(state_dict)

        if self.verbose:
            logging.info("Improved metric to %.5f. Model saved.", metric)

    # --------------------------------------------------------------------- #
    def load_best_weights(self, model: torch.nn.Module) -> None:
        """恢复最佳权重"""
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            model.load_state_dict(torch.load(self.ckpt_path))
        elif self._best_state_dict is not None:
            model.load_state_dict(self._best_state_dict)
        else:
            raise RuntimeError("No checkpoint available to load.")
