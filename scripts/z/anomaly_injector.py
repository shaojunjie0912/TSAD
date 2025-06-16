from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class RandomAnomConfig:
    """
    随机异常注入的可调参数（全部都有默认值，可按需覆盖）

    —— 命名规则
    *_range    : Tuple(lo, hi)  表示均匀随机区间
    *_coeff_*  : 与列自身 std() 的比例因子
    *_ratio_*  : 0~1 比例
    *_len      : 整数 (采样点数)
    """

    # 全局
    anomaly_rate: float = 0.05  # * 100 %
    seed: int = 1037
    target_cols: Optional[List[str]] = None  # None: 所有列

    # 区间长度范围
    interval_len_range: Tuple[int, int] = (4, 12)

    # point-spike
    n_spikes: int = 10
    spike_factor_range: Tuple[float, float] = (1.5, 3.0)

    # level-shift
    shift_coeff_range: Tuple[float, float] = (-3.0, 3.0)

    # sensor-drift
    drift_rate_coeff_range: Tuple[float, float] = (-0.1, 0.1)  # per minute

    # fixed-value
    # "hold" | "min" | "max" | ("percentile", p) | ("constant", val)
    fixed_value_strategy: Union[str, Tuple[str, float]] = ("percentile", 0.95)

    # shock
    peak_coeff_range: Tuple[float, float] = (0.5, 2.0)
    rise_ratio_range: Tuple[float, float] = (0.2, 0.6)
    decay_rate_range: Tuple[float, float] = (0.01, 0.05)

    # 允许的异常类型
    anomaly_types: Tuple[str, ...] = (
        "point_spike",
        "level_shift",
        "sensor_drift",
        "sensor_fixed_value",
        "shock_pattern",
    )


# NOTE: 会把数据类型都转为 float64
class AnomalyInjector:
    def __init__(
        self,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        track_labels: bool = True,
    ):
        self.clamp_min = clamp_min if clamp_min is not None else -np.inf
        self.clamp_max = clamp_max if clamp_max is not None else np.inf
        self.track_labels = track_labels
        self.labels: Optional[pd.DataFrame] = None

    def _ensure_float_dtype(self, df: pd.DataFrame, cols: List[str]) -> None:
        """
        将指定列原地转成 float64 (若本来就为浮点则保持不变)
        """
        need_cast = [c for c in cols if not pd.api.types.is_float_dtype(df[c].dtype)]
        if need_cast:
            df[need_cast] = df[need_cast].astype(float, copy=False)

    def get_labels_detailed(self) -> pd.DataFrame:
        """
        返回累计得到的多列标签 (完整具体到每个时间点每个变量是否异常 1/0)
        """
        if self.labels is None:
            raise RuntimeError("没有记录标签, 请先注入异常!")
        return self.labels.copy()

    def get_labels_simple(self) -> pd.Series:
        """
        返回累计得到的单列标签 (每个时间点有一个变量异常则当前时间点异常 1/0)
        """
        if self.labels is None:
            raise RuntimeError("没有记录标签, 请先注入异常!")
        return self.labels.copy().any(axis=1).astype(np.int8)

    def reset_labels(self) -> None:
        """把累计的标签清零"""
        if self.labels is not None:
            self.labels.iloc[:, :] = 0

    def _ensure_labels(self, df: pd.DataFrame) -> None:
        """
        若 self.labels 为空，则初始化与 df 等形状的 0/1 DataFrame。
        """
        if not self.track_labels:
            return
        if self.labels is None:
            self.labels = pd.DataFrame(
                0, index=df.index.copy(), columns=df.columns.copy(), dtype=np.int8
            )
        else:
            # 若同一次会话里注入了不同 index/columns 的 df，可在此加 assert。
            if not self.labels.index.equals(df.index) or not self.labels.columns.equals(df.columns):
                raise ValueError(
                    "标签 DataFrame 形状不匹配. 请创建一个新的 AnomalyInjector 用于新的数据集."
                )

    def _mark_label(self, rows: pd.Index, cols: Union[str, List[str]]) -> None:
        """
        rows: 需要标 1 的 DatetimeIndex
        cols: 单列名或列名列表
        """
        if not self.track_labels:
            return
        self._ensure_labels(self.labels)  # type: ignore[arg-type]
        col_list: List[str] = [cols] if isinstance(cols, str) else cols
        self.labels.loc[rows, col_list] = 1  # type: ignore[index]

    def _check_columns(self, df: pd.DataFrame, cols: Union[str, List[str]]) -> List[str]:
        """检查无 NaN 且 index 为DatetimeIndex 且列名存在

        Args:
            df (pd.DataFrame): 输入的 DataFrame
            cols (Union[str, List[str]]): 输入的列名

        Raises:
            ValueError: 输入的 DataFrame 包含 NaN 值
            ValueError: 输入的 DataFrame 的 index 不是 pd.DatetimeIndex
            ValueError: 输入的列名不存在于 DataFrame 中

        Returns:
            List[str]: 输入的列名列表
        """
        if df.isna().any().any():
            raise ValueError(
                "DataFrame contains NaN values. Please handle NaN values before injecting anomalies."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be `pd.DatetimeIndex`.")

        col_list = [cols] if isinstance(cols, str) else cols
        for col in col_list:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        return col_list

    def _check_interval(
        self,
        interval: Union[
            Tuple[str, str],
            List[Tuple[str, str]],
            Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
        ],
    ) -> None:
        """检查输入的一个/多个 interval 是否合法 (start_time <= end_time)

        Args:
            interval (Union[ Tuple[str, str], List[Tuple[str, str]], Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]], ]): 输入的 interval

        Raises:
            ValueError: 输入的 interval 不合法 (start_time > end_time)
        """

        def _check_interval_tuple(interval: Tuple[str, str]) -> None:
            """检查输入的 interval 是否合法"""
            start_time = pd.to_datetime(interval[0])
            end_time = pd.to_datetime(interval[1])
            if start_time > end_time:
                raise ValueError(f"start_time {start_time} must be before end_time {end_time}")

        if isinstance(interval, tuple):
            _check_interval_tuple(interval)
        elif isinstance(interval, list):
            for item in interval:
                _check_interval_tuple(item)
        elif isinstance(interval, dict):
            for key, value in interval.items():
                if isinstance(value, tuple):
                    _check_interval_tuple(value)
                else:  # List[Tuple[str, str]]
                    for item in value:
                        _check_interval_tuple(item)

    def inject_point_spike(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        time_points: Union[str, List[str], Dict[str, Union[str, List[str]]]],
        factors: Optional[Union[float, Dict[str, float]]] = None,
        fixed_values: Optional[Union[float, Dict[str, float]]] = None,
    ) -> pd.DataFrame:

        col_list = self._check_columns(df, columns)

        df_copy = df.copy()
        self._ensure_float_dtype(df_copy, col_list)

        # 模式选择和操作值确定
        mode: str
        op_val: Union[float, Dict[str, float]]  # 操作值 (因子或固定值)

        if factors is not None and fixed_values is not None:
            raise ValueError("Cannot specify both 'factors' and 'fixed_values'. Choose one mode.")
        elif factors is not None:
            mode = "factor"
            op_val = factors
        elif fixed_values is not None:
            mode = "fixed"
            op_val = fixed_values
        else:
            raise ValueError(
                "Must specify either 'factors' (for spike anomaly) or 'fixed_values' (to set a fixed value)."
            )

        for col in col_list:
            # 1. 确定当前列的操作值 (因子或固定值)
            curr_op_val: float
            if isinstance(op_val, dict):
                if col not in op_val:
                    value_type_str = "Factor" if mode == "factor" else "Fixed value"
                    dict_name_str = "factors" if mode == "factor" else "fixed_values"
                    raise ValueError(
                        f"{value_type_str} for column '{col}' not found in '{dict_name_str}' dictionary. "
                        f"Available keys: {list(op_val.keys())}"
                    )
                curr_op_val = op_val[col]
            else:
                curr_op_val = op_val

            # 2. 确定当前列的异常时间点 (字符串列表)
            timestamps_str_list: List[str]

            if isinstance(time_points, str):
                timestamps_str_list = [time_points]
            elif isinstance(time_points, list):
                timestamps_str_list = time_points
            else:  # Dict
                if not isinstance(columns, list):
                    raise ValueError("columns must be a list when time_indices is a dict")

                if col not in time_points:
                    raise ValueError(f"Column '{col}' not found in 'time_indices' dictionary.")

                col_specific_times = time_points[col]
                if isinstance(col_specific_times, str):
                    timestamps_str_list = [col_specific_times]
                elif isinstance(col_specific_times, list):
                    timestamps_str_list = col_specific_times
                else:
                    raise TypeError(
                        f"Time specification for column '{col}' in 'time_indices' "
                        f"dictionary must be a string or a list of strings."
                    )

            # 3. 转换时间字符串为 pd.DatetimeIndex
            timestamps: pd.DatetimeIndex = pd.to_datetime(
                timestamps_str_list, format="%Y-%m-%d %H:%M:%S"
            )

            # 如果时间点不在 df_copy 的索引中, 则报错并打印出不在索引中的时间点
            mask = timestamps.isin(df_copy.index)
            if not mask.all():
                missing_time_points = timestamps[~mask]
                raise ValueError(f"时间点 {missing_time_points.tolist()} 不在 DataFrame 索引中。")

            # 应用异常
            original_values = df_copy.loc[timestamps, col]
            if mode == "factor":
                df_copy.loc[timestamps, col] = original_values * curr_op_val
            elif mode == "fixed":
                df_copy.loc[timestamps, col] = curr_op_val

            df_copy.loc[timestamps, col] = df_copy.loc[timestamps, col].clip(
                lower=self.clamp_min, upper=self.clamp_max
            )

            # 标记标签
            self._mark_label(timestamps, col)

        return df_copy

    def inject_level_shift(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        interval: Union[
            Tuple[str, str],  # Single interval for all specified columns
            List[Tuple[str, str]],  # Multiple intervals for all specified columns
            Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],  # Column-specific interval(s)
        ],
        shift_value: Union[float, Dict[str, float]],
    ) -> pd.DataFrame:

        col_list = self._check_columns(df, columns)

        self._check_interval(interval)

        df_copy = df.copy()
        self._ensure_float_dtype(df_copy, col_list)

        for col in col_list:
            # 1. 确定当前列的移位值
            curr_shift_val: float
            if isinstance(shift_value, dict):
                if col not in shift_value:
                    raise ValueError(
                        f"Shift value for column '{col}' not found in 'shift_value' dictionary. "
                        f"Available keys: {list(shift_value.keys())}"
                    )
                curr_shift_val = shift_value[col]
            else:  # shift_value 是单个 float
                curr_shift_val = shift_value

            # 2. 确定当前列的异常时间段列表 List[Tuple[str, str]]
            intervals_str: List[Tuple[str, str]] = []
            if isinstance(interval, tuple):  # 单个全局时间段 Tuple[str, str]
                intervals_str.append(interval)
            elif isinstance(interval, list):  # 全局时间段列表 List[Tuple[str, str]]
                intervals_str.extend(interval)
            elif isinstance(interval, dict):  # 列特定的时间段配置
                if col not in interval:
                    raise ValueError(f"Column '{col}' not found in 'interval' dictionary.")
                col_interval = interval[col]
                if isinstance(col_interval, tuple):  # Dict[str, Tuple[str, str]]
                    intervals_str.append(col_interval)
                elif isinstance(col_interval, list):  # Dict[str, List[Tuple[str, str]]]
                    intervals_str.extend(col_interval)
                else:
                    raise TypeError(
                        f"Interval configuration for column '{col}' in 'interval' dictionary "
                        f"must be Tuple[str, str] or List[Tuple[str, str]]."
                    )

            # 3. 对当前列的每个确定时间段应用移位
            for start_str, end_str in intervals_str:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)

                interval_indices = df_copy.loc[start_ts:end_ts].index

                if interval_indices.empty:
                    raise ValueError(f"列 '{col}' 在时间段 [{start_ts}, {end_ts}] 中没有数据。")

                original_values_slice = df_copy.loc[interval_indices, col]

                df_copy.loc[interval_indices, col] = original_values_slice + curr_shift_val

                # 钳位
                df_copy.loc[interval_indices, col] = df_copy.loc[interval_indices, col].clip(
                    lower=self.clamp_min, upper=self.clamp_max
                )

                # 标记标签
                self._mark_label(interval_indices, col)

        return df_copy

    def inject_sensor_drift(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        intervals: Union[
            Tuple[str, str],
            List[Tuple[str, str]],
            Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
        ],
        drift_rates: Union[float, Dict[str, float]],  # 漂移速率，例如：单位值/每分钟
    ) -> pd.DataFrame:

        col_list = self._check_columns(df, columns)

        self._check_interval(intervals)

        df_copy = df.copy()

        self._ensure_float_dtype(df_copy, col_list)

        for col in col_list:
            # 1. 确定当前列的漂移速率
            curr_drift_rate: float  # 现在的单位是 units/minute
            if isinstance(drift_rates, dict):
                if col not in drift_rates:
                    raise ValueError(
                        f"Drift rate for column '{col}' not found in 'drift_rate' dictionary. "
                        f"Available keys: {list(drift_rates.keys())}"
                    )
                curr_drift_rate = drift_rates[col]
            else:  # drift_rate 是单个 float
                curr_drift_rate = drift_rates

            # 2. 确定当前列的异常时间段列表 List[Tuple[str, str]]
            intervals_for_col_str: List[Tuple[str, str]] = []
            if isinstance(intervals, tuple):
                intervals_for_col_str.append(intervals)
            elif isinstance(intervals, list):
                intervals_for_col_str.extend(intervals)
            elif isinstance(intervals, dict):
                if col not in intervals:
                    raise ValueError(f"Column '{col}' not found in 'interval' dictionary.")
                col_specific_interval_config = intervals[col]
                if isinstance(col_specific_interval_config, tuple):
                    intervals_for_col_str.append(col_specific_interval_config)
                elif isinstance(col_specific_interval_config, list):
                    intervals_for_col_str.extend(col_specific_interval_config)
                else:
                    raise TypeError(
                        f"Interval configuration for column '{col}' in 'interval' dictionary "
                        f"must be Tuple[str, str] or List[Tuple[str, str]]."
                    )

            # 3. 对当前列的每个确定时间段应用漂移
            for start_str, end_str in intervals_for_col_str:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)

                interval_indices = df_copy.loc[start_ts:end_ts].index

                if interval_indices.empty:
                    raise ValueError(f"列 '{col}' 在时间段 [{start_ts}, {end_ts}] 中没有数据。")

                original_values_in_slice = df_copy.loc[interval_indices, col]

                # 计算从时间段开始到每个点的已流逝时间
                time_deltas_from_start = interval_indices - interval_indices[0]
                time_elapsed_minutes = time_deltas_from_start.total_seconds() / 60.0

                # 计算每个点的漂移量 (drift_rate 单位是 units/minute)
                drift_offset_values = curr_drift_rate * time_elapsed_minutes

                drift_series = pd.Series(drift_offset_values, index=interval_indices)

                df_copy.loc[interval_indices, col] = original_values_in_slice + drift_series

                # 钳位
                df_copy.loc[interval_indices, col] = df_copy.loc[interval_indices, col].clip(
                    lower=self.clamp_min, upper=self.clamp_max
                )

                # 标记标签
                self._mark_label(interval_indices, col)

        return df_copy

    def inject_sensor_fixed_value(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        intervals: Union[
            Tuple[str, str],  # 所有指定列的单个时间段
            List[Tuple[str, str]],  # 所有指定列的多个时间段
            Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],  # 列特定的时间段
        ],
        fixed_values: Union[float, Dict[str, float]],  # 传感器输出的固定值
    ) -> pd.DataFrame:

        col_list = self._check_columns(df, columns)

        self._check_interval(intervals)

        df_copy = df.copy()

        self._ensure_float_dtype(df_copy, col_list)

        for col in col_list:
            # 1. 确定当前列的固定值
            current_col_fixed_value: Union[int, float]
            if isinstance(fixed_values, dict):
                if col not in fixed_values:
                    raise ValueError(
                        f"Fixed value for column '{col}' not found in 'fixed_value' dictionary. "
                        f"Available keys: {list(fixed_values.keys())}"
                    )
                current_col_fixed_value = fixed_values[col]
            else:  # fixed_value 是单个 int 或 float
                current_col_fixed_value = fixed_values

            # 2. 确定当前列的异常时间段列表 List[Tuple[str, str]]
            intervals_for_col_str: List[Tuple[str, str]] = []
            if isinstance(intervals, tuple):
                intervals_for_col_str.append(intervals)
            elif isinstance(intervals, list):
                intervals_for_col_str.extend(intervals)
            else:  # Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]]
                if col not in intervals:
                    raise ValueError(f"Column '{col}' not found in 'interval' dictionary.")
                col_specific_interval_config = intervals[col]
                if isinstance(col_specific_interval_config, tuple):
                    intervals_for_col_str.append(col_specific_interval_config)
                elif isinstance(col_specific_interval_config, list):
                    intervals_for_col_str.extend(col_specific_interval_config)
                else:
                    raise TypeError(
                        f"Interval configuration for column '{col}' in 'interval' dictionary "
                        f"must be Tuple[str, str] or List[Tuple[str, str]]."
                    )

            # 3. 对当前列的每个确定时间段应用固定值
            for start_str, end_str in intervals_for_col_str:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)

                interval_indices = df_copy.loc[start_ts:end_ts].index

                if interval_indices.empty:
                    raise ValueError(f"列 '{col}' 在时间段 [{start_ts}, {end_ts}] 中没有数据。")

                df_copy.loc[interval_indices, col] = current_col_fixed_value

                # 钳位
                df_copy.loc[interval_indices, col] = df_copy.loc[interval_indices, col].clip(
                    lower=self.clamp_min, upper=self.clamp_max
                )

                # 标记标签
                self._mark_label(interval_indices, col)

        return df_copy

    def _generate_custom_shock_pattern(
        self,
        duration_points: int,
        peak_value: float,
        rise_time_ratio: float,
        decay_rate: float,
    ) -> np.ndarray:
        """
        生成一个自定义的冲击模式（快速上升然后指数衰减）。
        """
        if duration_points <= 0:
            raise ValueError(f"Duration points must be positive, got {duration_points}.")

        t = np.arange(duration_points)
        # 确保 rise_points 在 [0, duration_points] 范围内
        rise_points = min(max(0, int(duration_points * rise_time_ratio)), duration_points)

        pattern = np.zeros(duration_points, dtype=float)

        if rise_points > 0:
            # np.linspace(start, stop, num=N) 中，如果 N=1，结果是 array([stop])
            # 如果 N > 1，最后一个点 (索引 N-1) 是 stop
            pattern[:rise_points] = np.linspace(0, peak_value, rise_points, endpoint=True)

        if rise_points < duration_points:  # 只有当有空间进行衰减时
            # 衰减计算的相对时间调整
            # 如果 rise_points = 0, 峰值在索引0, 相对时间从0开始 (t[0:] - 0)
            # 如果 rise_points > 0, 峰值在索引 rise_points-1, 衰减从索引 rise_points 开始，其相对时间从1开始
            # (t_current - t_peak_index) = (t_current - (rise_points-1))
            time_offset_for_decay_calc = rise_points - 1 if rise_points > 0 else 0
            effective_time_for_decay = t[rise_points:] - time_offset_for_decay_calc
            pattern[rise_points:] = peak_value * np.exp(-decay_rate * effective_time_for_decay)
        # 如果 rise_points == duration_points，则整个模式都是上升阶段，已由linspace处理。

        return pattern

    def inject_shock_pattern(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]],
        intervals: Union[
            Tuple[str, str],
            List[Tuple[str, str]],
            Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]],
        ],
        peak_values: Union[float, Dict[str, float]],
        rise_time_ratios: Union[float, Dict[str, float]],
        decay_rates: Union[float, Dict[str, float]],
    ) -> pd.DataFrame:
        col_list = self._check_columns(df, columns)

        self._check_interval(intervals)

        df_copy = df.copy()

        self._ensure_float_dtype(df_copy, col_list)

        for col in col_list:
            current_peak: float = peak_values[col] if isinstance(peak_values, dict) else peak_values
            current_rise_ratio: float = (
                rise_time_ratios[col] if isinstance(rise_time_ratios, dict) else rise_time_ratios
            )
            current_decay_rate: float = (
                decay_rates[col] if isinstance(decay_rates, dict) else decay_rates
            )

            if not (0.0 <= current_rise_ratio <= 1.0):
                raise ValueError(
                    f"Rise time ratio for column '{col}' must be between 0.0 and 1.0, got {current_rise_ratio}."
                )
            if current_decay_rate < 0.0:
                raise ValueError(
                    f"Decay rate for column '{col}' must be non-negative, got {current_decay_rate}."
                )

            intervals_for_col_str: List[Tuple[str, str]] = []
            if isinstance(intervals, tuple):
                intervals_for_col_str.append(intervals)
            elif isinstance(intervals, list):
                for item_interval_tuple in intervals:
                    intervals_for_col_str.append(item_interval_tuple)
            else:  # Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]]
                if col not in intervals:
                    raise ValueError(f"Column '{col}' not found in 'interval' dictionary.")
                col_specific_interval_config = intervals[col]
                if isinstance(col_specific_interval_config, tuple):
                    intervals_for_col_str.append(col_specific_interval_config)
                else:  # List[Tuple[str, str]]
                    intervals_for_col_str.extend(col_specific_interval_config)

            for start_str, end_str in intervals_for_col_str:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)

                interval_indices = df_copy.loc[start_ts:end_ts].index

                if interval_indices.empty:
                    raise ValueError(f"列 '{col}' 在时间段 [{start_ts}, {end_ts}] 中没有数据。")

                duration_points = len(interval_indices)

                shock_pattern_values = self._generate_custom_shock_pattern(
                    duration_points, current_peak, current_rise_ratio, current_decay_rate
                )

                if shock_pattern_values.size == 0:
                    raise ValueError("冲击模式长度为 0")

                original_values_in_slice = df_copy.loc[interval_indices, col]

                if not pd.api.types.is_numeric_dtype(original_values_in_slice.dtype):
                    print(
                        f"Warning: Data in column '{col}' for interval [{start_ts}, {end_ts}] (shock pattern) "
                        f"is of non-numeric type '{original_values_in_slice.dtype}'. Skipping shock for this interval."
                    )
                    continue

                shock_series = pd.Series(shock_pattern_values, index=interval_indices)
                df_copy.loc[interval_indices, col] = original_values_in_slice + shock_series

                # 钳位
                df_copy.loc[interval_indices, col] = df_copy.loc[interval_indices, col].clip(
                    lower=self.clamp_min, upper=self.clamp_max
                )

                # 标记标签
                self._mark_label(interval_indices, col)

        return df_copy

    def inject_random_anomalies(
        self,
        df: pd.DataFrame,
        config: Optional[RandomAnomConfig] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        自动按 `config` 注入随机异常并返回 (新 df, labels deep-copy)
        """
        cfg = config if config is not None else RandomAnomConfig()
        rng = np.random.default_rng(cfg.seed)

        df_new = df.copy()
        self._ensure_labels(df_new)

        target_cols = df.columns.tolist() if cfg.target_cols is None else cfg.target_cols

        # 目标异常点数
        total_cells = len(df.index) * len(target_cols)
        target_count = max(1, int(total_cells * cfg.anomaly_rate))

        def _random_interval() -> Tuple[str, str]:
            """随机生成一个时间区间"""
            lo, hi = cfg.interval_len_range
            if hi > len(df.index):
                raise ValueError(
                    f"Interval length range ({lo}, {hi}) is too large for the given dataframe."
                )
            length = int(rng.integers(lo, hi + 1))
            start_idx = int(rng.integers(0, len(df) - length))
            start_ts = df.index[start_idx]
            end_ts = df.index[start_idx + length - 1]
            return str(start_ts), str(end_ts)

        def _pick_fixed_value(series: pd.Series) -> float:
            """随机选取一个固定值"""
            strategy = cfg.fixed_value_strategy
            if strategy == "hold":
                return float(series.iloc[-1])
            if strategy == "min":
                return float(series.min())
            if strategy == "max":
                return float(series.max())
            if isinstance(strategy, tuple) and strategy[0] == "percentile":
                p = float(strategy[1])
                return float(np.percentile(series, p * 100))
            if isinstance(strategy, tuple) and strategy[0] == "constant":
                return float(strategy[1])
            raise ValueError(f"Unsupported fixed_value_strategy: {strategy}")

        anomaly_types = cfg.anomaly_types

        while int(self.labels.values.sum()) < target_count:  # type: ignore[arg-type]
            col = rng.choice(target_cols)
            atype = rng.choice(anomaly_types)

            match atype:
                case "point_spike":
                    ts_list = rng.choice(df.index, size=cfg.n_spikes, replace=False)
                    factor = rng.uniform(*cfg.spike_factor_range)
                    df_new = self.inject_point_spike(
                        df=df_new,
                        columns=[col],
                        time_points=[
                            pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S") for ts in ts_list
                        ],
                        factors=factor,
                    )

                case "level_shift":
                    interval = _random_interval()
                    coeff = rng.uniform(*cfg.shift_coeff_range)
                    shift_val = coeff * df[col].std()
                    df_new = self.inject_level_shift(
                        df=df_new, columns=[col], interval=interval, shift_value=shift_val
                    )

                case "sensor_drift":
                    interval = _random_interval()
                    coeff = rng.uniform(*cfg.drift_rate_coeff_range)
                    rate = coeff * df[col].std()  # units per minute
                    df_new = self.inject_sensor_drift(
                        df=df_new, columns=[col], intervals=interval, drift_rates=rate
                    )

                case "sensor_fixed_value":
                    interval = _random_interval()
                    fixed_val = _pick_fixed_value(df[col])
                    df_new = self.inject_sensor_fixed_value(
                        df=df_new,
                        columns=[col],
                        intervals=interval,
                        fixed_values=fixed_val,
                    )

                case "shock_pattern":
                    interval = _random_interval()
                    peak_coeff = rng.uniform(*cfg.peak_coeff_range)
                    rise_ratio = rng.uniform(*cfg.rise_ratio_range)
                    decay_rate = rng.uniform(*cfg.decay_rate_range)
                    df_new = self.inject_shock_pattern(
                        df=df_new,
                        columns=[col],
                        intervals=interval,
                        peak_values=peak_coeff * df[col].std(),
                        rise_time_ratios=rise_ratio,
                        decay_rates=decay_rate,
                    )

                case _:
                    raise ValueError(f"Unknown anomaly type '{atype}'")

        return df_new, self.get_labels_detailed(), self.get_labels_simple()
