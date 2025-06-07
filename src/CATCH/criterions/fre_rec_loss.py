from typing import Literal, Optional

import numpy as np
import torch
from einops import rearrange

# TODO: 删除了 mask 参数, 根本没用到


class FrequencyLoss(torch.nn.Module):
    def __init__(
        self,
        fft_mode: Literal["fft", "rfft"],
        complex_error_type: Literal[
            "complex", "complex-phase", "complex-mag-phase", "phase", "mag", "mag-phase"
        ],
        loss_type: Literal["MAE", "MSE"],
        module_first: bool,
        mean_dim: Optional[int] = None,
        keep_mean_dim: bool = False,
    ):
        """_summary_

        Args:
            fft_mode (Literal["fft", "rfft"]): 傅里叶变换的方式
                "fft": 直接计算复数频谱
                "rfft": 计算实数频谱
            complex_error_type (Literal["complex", "complex-phase", "complex-mag-phase", "phase", "mag", "mag-phase"]): 计算复数误差的方式
                "complex": 直接计算复数频谱的差异
                "complex-phase": 先计算复数频谱差异，然后仅提取其相位角度
                "complex-mag-phase": 先计算复数频谱差异，然后提取其幅值和相位角度
                "phase": 分别计算重建值和真实值的相位, 然后比较差异
                "mag": 分别计算重建值和真实值的幅值, 然后比较差异
                "mag-phase": 分别计算幅度差异和相位差异，然后组合
            loss_type (Literal["MAE", "MSE"]):
            module_first (bool): 复数误差的处理顺序
                True: 先计算每个样本点的误差幅值(模), 再对幅值取平均(NOTE: 保留了每个频率分量误差的绝对大小)
                False: 先对每个样本点的复数误差取平均, 再计算平均误差的幅值( NOTE: 允许不同样本点的正负误差相互抵消)
            mean_dim (Optional[int], optional): 对误差取均值的维度
            keep_mean_dim (bool, optional): 是否保留误差均值维度

        Raises:
            NotImplementedError:
        """
        super().__init__()

        if fft_mode not in ["fft", "rfft"]:
            raise NotImplementedError

        if loss_type not in ["MAE", "MSE"]:
            raise NotImplementedError

        if complex_error_type not in [
            "complex",
            "complex-phase",
            "complex-mag-phase",
            "phase",
            "mag",
            "mag-phase",
        ]:
            raise NotImplementedError

        self.mean_dim = mean_dim
        self.keep_mean_dim = keep_mean_dim
        self.fft = torch.fft.fft if fft_mode == "fft" else torch.fft.rfft
        self.complex_error_type = complex_error_type
        self.loss_type = loss_type
        self.module_first = module_first

    def forward(self, z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            z_hat (torch.Tensor): 重构值 (batch_size, seq_len, num_features)
            z (torch.Tensor): 真实值 (batch_size, seq_len, num_features)

        Raises:
            NotImplementedError:
            NotImplementedError:

        Returns:
            torch.Tensor: _description_
        """
        # NOTE: 如果不是复数则在时间维度 (seq_len) 上进行傅里叶变换
        if not z_hat.is_complex():
            z_hat = self.fft(z_hat, dim=1)

        z = self.fft(z, dim=1)

        frequency_loss = torch.tensor(0, device=z.device)

        # 计算复数误差
        match self.complex_error_type:
            case "complex":
                frequency_loss = z_hat - z
            case "complex-phase":
                frequency_loss = (z_hat - z).angle()
            case "complex-mag-phase":
                frequency_loss_mag = (z_hat - z).abs()
                frequency_loss_phase = (z_hat - z).angle()
                frequency_loss = torch.stack([frequency_loss_mag, frequency_loss_phase])
            case "phase":
                frequency_loss = z_hat.angle() - z.angle()
            case "mag":
                frequency_loss = z_hat.abs() - z.abs()
            case "mag-phase":
                frequency_loss_mag = z_hat.abs() - z.abs()
                frequency_loss_phase = z_hat.angle() - z.angle()
                frequency_loss = torch.stack([frequency_loss_mag, frequency_loss_phase])

        # 计算误差均值
        match self.loss_type:
            case "MAE":
                frequency_loss = (
                    frequency_loss.abs().mean(dim=self.mean_dim, keepdim=self.keep_mean_dim)
                    if self.module_first
                    else frequency_loss.mean(dim=self.mean_dim, keepdim=self.keep_mean_dim).abs()
                )
            case "MSE":
                frequency_loss = (
                    (frequency_loss.abs() ** 2).mean(dim=self.mean_dim, keepdim=self.keep_mean_dim)
                    if self.module_first
                    else (frequency_loss**2).mean(dim=self.mean_dim, keepdim=self.keep_mean_dim).abs()
                )

        return frequency_loss


class FrequencyCriterion(torch.nn.Module):
    def __init__(
        self,
        fft_mode: Literal["fft", "rfft"],
        complex_error_type: Literal[
            "complex", "complex-phase", "complex-mag-phase", "phase", "mag", "mag-phase"
        ],
        loss_type: Literal["MAE", "MSE"],
        module_first: bool,
        seq_len: int,
        inference_patch_size: int,
        inference_patch_stride: int,
    ):
        """_summary_

        Args:
            fft_mode (Literal["fft", "rfft"]): 傅里叶变换的方式
                "fft": 直接计算复数频谱
                "rfft": 计算实数频谱
            complex_error_type (Literal["complex", "complex-phase", "complex-mag-phase", "phase", "mag", "mag-phase"]): 计算复数误差的方式
                "complex": 直接计算复数频谱的差异
                "complex-phase": 先计算复数频谱差异，然后仅提取其相位角度
                "complex-mag-phase": 先计算复数频谱差异，然后提取其幅值和相位角度
                "phase": 分别计算重建值和真实值的相位, 然后比较差异
                "mag": 分别计算重建值和真实值的幅值, 然后比较差异
                "mag-phase": 分别计算幅度差异和相位差异，然后组合
            loss_type (Literal["MAE", "MSE"]):
            module_first (bool): 复数误差的处理顺序
                True: 先计算每个样本点的误差幅值(模), 再对幅值取平均(NOTE: 保留了每个频率分量误差的绝对大小)
                False: 先对每个样本点的复数误差取平均, 再计算平均误差的幅值( NOTE: 允许不同样本点的正负误差相互抵消)
            seq_len (int): 滑动窗口大小
            inference_patch_size (int): 推理阶段 patch 大小 (NOTE: patch 就是对 window 再做切片)
            inference_patch_stride (int): 推理阶段 patch 步长
        """
        super().__init__()
        self.metric = FrequencyLoss(
            fft_mode=fft_mode,
            complex_error_type=complex_error_type,
            loss_type=loss_type,
            module_first=module_first,
            mean_dim=1,  # NOTE: 在时间维度上计算均值
            keep_mean_dim=True,  # NOTE: 保留均值维度 shape(batch_size, 1, num_features)
        )
        self.patch_size = inference_patch_size
        self.patch_stride = inference_patch_stride
        self.window_size = seq_len
        self.patch_num = int((self.window_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.window_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            z_hat (torch.Tensor): 重构值 (batch_size, seq_len, num_features)
            z (torch.Tensor): 真实值 (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: _description_
        """
        z_hat_patch = z_hat.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)

        b, n, c, p = z_hat_patch.shape
        z_hat_patch = rearrange(z_hat_patch, "b n c p -> (b n) p c")
        z_patch = z.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        z_patch = rearrange(z_patch, "b n c p -> (b n) p c")

        main_part_loss = self.metric(z_hat_patch, z_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, "(b n) p c -> b n p c", b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = (
            torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)])
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.window_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(z_hat[:, -self.padding_length :, :], z[:, -self.padding_length :, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss
