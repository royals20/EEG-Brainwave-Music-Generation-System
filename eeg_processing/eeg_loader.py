import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from eeg_processing.preprocess import ensure_microvolt_scale
from utils.config import EEG_SIMULATION_CONFIG

TIME_COLUMN_NAMES = {'time', 'timestamp'}
CSV_TIME_INTERVAL_TOLERANCE = 0.05


class EEGLoader:

    def __init__(self):
        self.raw = None
        self.data = None
        self.sample_rate = None
        self.channel_names = None
        self.duration = None
        self.file_path = None
        self.is_simulated = False
        self._csv_import_cache = {}

    def _get_csv_cache_key(self, file_path: str) -> Tuple[str, int, int]:
        normalized_path = os.path.abspath(file_path)
        stat_result = os.stat(normalized_path)
        return normalized_path, int(stat_result.st_size), int(stat_result.st_mtime_ns)

    def _get_cached_csv_import(self, file_path: str) -> Optional[Dict[str, Any]]:
        cache_key = self._get_csv_cache_key(file_path)
        return self._csv_import_cache.get(cache_key)

    def _cache_csv_import(self, file_path: str, details: Dict[str, Any]):
        cache_key = self._get_csv_cache_key(file_path)
        self._csv_import_cache = {cache_key: details}

    def _read_csv_frame(self, file_path: str) -> pd.DataFrame:
        frame = pd.read_csv(file_path)
        if frame.empty and not list(frame.columns):
            raise ValueError('CSV 文件为空，无法导入 EEG 数据。')
        return frame

    def _extract_time_column(
        self,
        frame: pd.DataFrame,
        valid_row_mask: np.ndarray,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        for column in frame.columns:
            if str(column).strip().lower() not in TIME_COLUMN_NAMES:
                continue

            time_series = pd.to_numeric(frame.loc[valid_row_mask, column], errors='coerce')
            return str(column), time_series.to_numpy(dtype=float)

        return None, None

    def _prepare_csv_import(self, file_path: str) -> Dict[str, Any]:
        cached_details = self._get_cached_csv_import(file_path)
        if cached_details is not None:
            return cached_details

        frame = self._read_csv_frame(file_path)

        channel_frame = frame.copy()
        ignored_columns = [
            column
            for column in channel_frame.columns
            if str(column).strip().lower() in TIME_COLUMN_NAMES
        ]
        if ignored_columns:
            channel_frame = channel_frame.drop(columns=ignored_columns)

        if channel_frame.empty:
            raise ValueError('CSV 中未找到 EEG 通道列。请保留通道列，时间列可命名为 time 或 timestamp。')

        valid_row_mask = ~channel_frame.isna().all(axis=1)
        channel_frame = channel_frame.loc[valid_row_mask]
        if channel_frame.empty:
            raise ValueError('CSV 中没有可用的 EEG 采样点。')

        numeric_columns = {}
        for column in channel_frame.columns:
            numeric_series = pd.to_numeric(channel_frame[column], errors='coerce')
            if numeric_series.isna().any():
                raise ValueError(
                    f"CSV 列 '{column}' 包含非数值或空白数据。"
                    "当前仅支持“列=通道、行=采样点”的数值 EEG 文件。"
                )
            numeric_columns[str(column)] = numeric_series.to_numpy(dtype=float)

        if not numeric_columns:
            raise ValueError('CSV 中没有可导入的 EEG 通道。')

        time_column_name, time_values = self._extract_time_column(frame, valid_row_mask)
        inferred_sample_rate, inference_reason = self._infer_sample_rate(time_values)

        details = {
            'file_path': file_path,
            'channel_names': list(numeric_columns.keys()),
            'channel_count': len(numeric_columns),
            'sample_count': len(channel_frame),
            'numeric_columns': numeric_columns,
            'time_column_name': time_column_name,
            'time_values': time_values,
            'inferred_sample_rate': inferred_sample_rate,
            'sample_rate_reason': inference_reason,
            'can_auto_infer_sample_rate': inferred_sample_rate is not None,
        }
        self._cache_csv_import(file_path, details)
        return details

    def _infer_sample_rate(self, time_values: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[str]]:
        if time_values is None:
            return None, '未检测到 time/timestamp 列。'

        if len(time_values) < 2:
            return None, '时间列样本点不足，无法推断采样率。'

        if not np.all(np.isfinite(time_values)):
            return None, '时间列包含非数值或空白数据，无法推断采样率。'

        time_diffs = np.diff(time_values)
        if np.any(time_diffs <= 0):
            return None, '时间列必须严格递增，无法推断采样率。'

        median_interval = float(np.median(time_diffs))
        if median_interval <= 0:
            return None, '时间列间隔无效，无法推断采样率。'

        relative_deviation = np.abs(time_diffs - median_interval) / median_interval
        if np.max(relative_deviation) > CSV_TIME_INTERVAL_TOLERANCE:
            return None, '时间列间隔不稳定，无法可靠推断采样率。'

        inferred_sample_rate = 1.0 / median_interval
        if not np.isfinite(inferred_sample_rate) or inferred_sample_rate <= 0:
            return None, '推断得到的采样率无效。'

        return inferred_sample_rate, None

    def inspect_csv(self, file_path: str) -> Dict[str, Any]:
        details = self._prepare_csv_import(file_path)
        return {
            'file_path': details['file_path'],
            'channel_names': details['channel_names'],
            'channel_count': details['channel_count'],
            'sample_count': details['sample_count'],
            'time_column_name': details['time_column_name'],
            'inferred_sample_rate': details['inferred_sample_rate'],
            'sample_rate_reason': details['sample_rate_reason'],
            'can_auto_infer_sample_rate': details['can_auto_infer_sample_rate'],
        }

    def load_edf(self, file_path: str) -> Dict[str, Any]:
        if not MNE_AVAILABLE:
            raise ImportError('未安装 MNE 库，无法导入 EDF/BDF。请先执行：pip install mne')

        self.file_path = file_path
        self.is_simulated = False

        extension = os.path.splitext(file_path)[1].lower()
        if extension == '.edf':
            self.raw = mne.io.read_raw_edf(file_path, preload=True)
        elif extension == '.bdf':
            self.raw = mne.io.read_raw_bdf(file_path, preload=True)
        else:
            raise ValueError(f'不支持的文件格式：{extension}')

        # MNE returns EEG in volts; the rest of the pipeline uses microvolt thresholds.
        self.data = self.raw.get_data() * 1e6
        self.sample_rate = self.raw.info['sfreq']
        self.channel_names = self.raw.ch_names
        self.duration = self.data.shape[1] / self.sample_rate

        return self.get_info()

    def load_csv(self, file_path: str, sample_rate: Optional[float] = None) -> Dict[str, Any]:
        self.file_path = file_path
        self.is_simulated = False

        details = self._prepare_csv_import(file_path)

        resolved_sample_rate = sample_rate
        if resolved_sample_rate is None:
            resolved_sample_rate = details['inferred_sample_rate']

        if resolved_sample_rate is None:
            raise ValueError(
                'CSV 缺少可推断的有效采样率。'
                '请在导入时根据设备配置手动输入采样率。'
            )

        resolved_sample_rate = float(resolved_sample_rate)
        if resolved_sample_rate <= 0:
            raise ValueError('采样率必须大于 0 Hz。')

        sample_matrix = np.column_stack(
            [details['numeric_columns'][column] for column in details['channel_names']]
        )
        scaled_data, _, _ = ensure_microvolt_scale(sample_matrix.T)

        self.data = scaled_data
        self.sample_rate = resolved_sample_rate
        self.channel_names = details['channel_names']
        self.duration = sample_matrix.shape[0] / resolved_sample_rate

        return self.get_info()

    def load_simulated(
        self,
        duration_minutes: float = None,
        sample_rate: int = None,
        sws_ratio: float = None,
    ) -> Dict[str, Any]:
        config = EEG_SIMULATION_CONFIG
        duration_minutes = duration_minutes or config['duration_minutes']
        sample_rate = sample_rate or config['sample_rate']
        sws_ratio = sws_ratio or config['sws_ratio']

        self.is_simulated = True
        self.sample_rate = sample_rate
        self.duration = duration_minutes * 60
        self.channel_names = ['Fz', 'Cz', 'Pz', 'Oz']

        n_samples = int(self.duration * sample_rate)
        n_channels = len(self.channel_names)
        self.data = np.zeros((n_channels, n_samples))

        time_axis = np.arange(n_samples, dtype=float) / sample_rate

        for channel_index in range(n_channels):
            noise = np.random.randn(n_samples) * 10
            background = np.sin(2 * np.pi * 0.5 * time_axis + np.random.rand() * 2 * np.pi) * 5

            n_sws_segments = int(self.duration * sws_ratio / 30)
            for _ in range(n_sws_segments):
                start_time = np.random.rand() * max(self.duration - 30, 1)
                start_idx = int(start_time * sample_rate)
                end_idx = min(int((start_time + 30) * sample_rate), n_samples)

                sws_freq = 0.75 + np.random.rand() * 0.5
                sws_amplitude = 75 + np.random.rand() * 50
                sws_time_axis = time_axis[start_idx:end_idx]
                sws_signal = np.sin(2 * np.pi * sws_freq * sws_time_axis) * sws_amplitude
                self.data[channel_index, start_idx:end_idx] += sws_signal

            self.data[channel_index, :] += noise + background

        return self.get_info()

    def get_info(self) -> Dict[str, Any]:
        return {
            'sample_rate': self.sample_rate,
            'channel_count': len(self.channel_names) if self.channel_names else 0,
            'duration': self.duration,
            'channel_names': self.channel_names,
            'file_path': self.file_path,
            'is_simulated': self.is_simulated,
        }

    def get_channel_data(self, channel_idx: int = 0) -> np.ndarray:
        if self.data is None:
            raise ValueError('当前尚未加载 EEG 数据，请先执行导入或生成模拟数据。')
        return self.data[channel_idx, :]

    def get_all_data(self) -> np.ndarray:
        if self.data is None:
            raise ValueError('当前尚未加载 EEG 数据，请先执行导入或生成模拟数据。')
        return self.data

    def get_time_vector(self) -> np.ndarray:
        if self.sample_rate is None or self.data is None:
            raise ValueError('当前尚未加载 EEG 数据。')
        sample_count = self.data.shape[1]
        return np.arange(sample_count, dtype=float) / self.sample_rate


def simulate_eeg(
    duration_minutes: float = 10,
    sample_rate: int = 256,
    sws_ratio: float = 0.3,
) -> Tuple[np.ndarray, float]:
    loader = EEGLoader()
    loader.load_simulated(duration_minutes, sample_rate, sws_ratio)
    return loader.get_channel_data(0), loader.sample_rate
