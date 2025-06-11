"""数据集处理模块"""

import json
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# TODO: 学习一下数据集 getitem 的实现, 人家这都是返回 tensor 类型

# NOTE: 滑窗拾遗
# 假设总数据长度为 N, 窗口大小为 W, 步长为 S
# 则窗口的数量为 (N - W) // S + 1
# 如果要覆盖所有数据
# - N, W, S 满足: (N - W) % S == 0
# 或者
# - padding

# 对于原始数据集, 总长度不固定, 因此只能手动做 padding 确保覆盖所有数据
# 对于一个自定义大小的 window 窗口再 patching 时, 此时已经知道了 N, 那么可以人为设置好 W 和 S, 以确保覆盖所有数据


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
        self.windows, self.masks = self.create_windows()

    def create_windows(self):
        num_samples, num_features = self.data.shape
        windows, padding_masks = [], []

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

            if end >= num_samples:
                break

        return np.stack(windows), np.stack(padding_masks)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = x.copy()
        padding_mask = self.masks[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        # NOTE:
        # X: (batch_size, window_size, num_features)
        # Y: (batch_size, window_size, num_features)
        # mask: (batch_size, window_size) 对应 window_size 中每个时间步是否被填充
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(padding_mask)


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


if __name__ == "__main__":
    # 测试 SlidingWindowDatasetWithPadding
    data = np.random.rand(10, 2).astype(np.float32)
    predict_dataloader = get_dataloader(
        stage="predict",
        data=data,
        batch_size=2,
        window_size=4,
        step_size=4,
        shuffle=False,
        padding_value=0.0,
    )
    for x, y, mask in predict_dataloader:
        print(f"x: {x.shape}, y: {y.shape}, mask: {mask.shape}")
        print(f"x: {x}")
        # print(f"y: {y}")
        print(f"mask: {mask}")
        break
