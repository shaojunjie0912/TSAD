import copy
import logging
import os
from typing import Literal, Optional

import torch


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
            state_dict = (
                model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            )
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
