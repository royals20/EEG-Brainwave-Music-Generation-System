from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert, welch

from eeg_processing.preprocess import bandpass_filter, ensure_microvolt_scale


def classify_relative_level(values: Sequence[float], summary_value: float = None) -> str:
    if not values:
        return 'medium'

    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return 'medium'
    if np.allclose(array, array[0]):
        return 'medium'

    low_ref = float(np.percentile(array, 10))
    high_ref = float(np.percentile(array, 90))
    if high_ref <= low_ref:
        return 'medium'

    summary = float(np.median(array) if summary_value is None else summary_value)
    normalized = (summary - low_ref) / (high_ref - low_ref)
    if normalized <= 0.33:
        return 'low'
    if normalized >= 0.67:
        return 'high'
    return 'medium'


def _smooth_array(values: Sequence[float], window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    window = max(1, int(window_size))
    if window <= 1 or array.size < window:
        return array.copy()

    half_window = window // 2
    return _centered_moving_average(array, half_window)


def _centered_window_bounds(size: int, half_window: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(size, dtype=int)
    starts = np.maximum(indices - half_window, 0)
    ends = np.minimum(indices + half_window + 1, size)
    return starts, ends


def _centered_moving_average(array: np.ndarray, half_window: int) -> np.ndarray:
    if array.size == 0:
        return array.copy()

    starts, ends = _centered_window_bounds(array.shape[0], half_window)
    cumulative = np.concatenate(([0.0], np.cumsum(array, dtype=float)))
    totals = cumulative[ends] - cumulative[starts]
    counts = ends - starts
    return totals / counts


def _centered_moving_average_matrix(matrix: np.ndarray, half_window: int) -> np.ndarray:
    if matrix.size == 0:
        return matrix.copy()

    starts, ends = _centered_window_bounds(matrix.shape[0], half_window)
    cumulative = np.vstack(
        [
            np.zeros((1, matrix.shape[1]), dtype=float),
            np.cumsum(matrix, axis=0, dtype=float),
        ]
    )
    totals = cumulative[ends] - cumulative[starts]
    counts = (ends - starts).reshape(-1, 1)
    return totals / counts


def _safe_normalize(values: Sequence[float], neutral: float = 0.5) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return np.zeros_like(array, dtype=float)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if np.isclose(maximum, minimum):
        return np.full(array.shape, neutral, dtype=float)

    normalized = (array - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0)


def smooth_features(features: List[Dict[str, Any]], window_size: int = 5) -> List[Dict[str, Any]]:
    if len(features) < window_size:
        return features

    smoothed = [feature.copy() for feature in features]
    numeric_keys = [
        'delta_power',
        'theta_power',
        'alpha_power',
        'beta_power',
        'dominant_frequency',
        'mean_amplitude',
        'slow_wave_ratio',
        'avg_instantaneous_amplitude',
    ]

    keys_to_smooth = [key for key in numeric_keys if any(key in feature for feature in features)]
    if not keys_to_smooth:
        return smoothed

    half_window = window_size // 2
    value_matrix = np.asarray(
        [[feature.get(key, 0.0) for key in keys_to_smooth] for feature in features],
        dtype=float,
    )
    smoothed_matrix = _centered_moving_average_matrix(value_matrix, half_window)

    for row_index, smoothed_feature in enumerate(smoothed):
        feature = features[row_index]
        for column_index, key in enumerate(keys_to_smooth):
            if key in feature:
                smoothed_feature[key] = float(smoothed_matrix[row_index, column_index])

    return smoothed


def extract_epoch_features(epoch_data: np.ndarray, fs: float) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)

    nperseg = min(1024, len(epoch_data))
    if nperseg >= 16:
        freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg)

        for band_name, (low, high) in [
            ('delta', delta_band),
            ('theta', theta_band),
            ('alpha', alpha_band),
            ('beta', beta_band),
        ]:
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features[f'{band_name}_power'] = float(np.trapz(psd[mask], freqs[mask]))
            else:
                features[f'{band_name}_power'] = 0.0

        delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
        if np.any(delta_mask):
            delta_freqs = freqs[delta_mask]
            delta_psd = psd[delta_mask]
            features['dominant_frequency'] = float(delta_freqs[np.argmax(delta_psd)])
        else:
            features['dominant_frequency'] = 1.0
    else:
        features['delta_power'] = 0.0
        features['theta_power'] = 0.0
        features['alpha_power'] = 0.0
        features['beta_power'] = 0.0
        features['dominant_frequency'] = 1.0

    features['mean_amplitude'] = float(np.mean(np.abs(epoch_data)))
    features['rms_amplitude'] = float(np.sqrt(np.mean(epoch_data ** 2)))

    delta_filtered = bandpass_filter(epoch_data, 0.5, 2.0, fs)
    analytic_signal = hilbert(delta_filtered)
    instantaneous_amplitude = np.abs(analytic_signal)
    features['avg_instantaneous_amplitude'] = float(np.mean(instantaneous_amplitude))

    is_slow_wave = (instantaneous_amplitude * 2.0) >= 75.0
    features['slow_wave_ratio'] = float(np.sum(is_slow_wave) / len(is_slow_wave)) if len(is_slow_wave) else 0.0
    features['slow_wave_period'] = float(1.0 / (features['dominant_frequency'] + 0.001))

    return features


def extract_features_from_epochs(eeg: np.ndarray, fs: float, epoch_size: float = 30.0) -> List[Dict[str, Any]]:
    epoch_samples = int(epoch_size * fs)
    if epoch_samples <= 0:
        return []

    n_epochs = len(eeg) // epoch_samples
    all_features = []

    for index in range(n_epochs):
        start_idx = index * epoch_samples
        end_idx = start_idx + epoch_samples
        epoch_data = eeg[start_idx:end_idx]

        features = extract_epoch_features(epoch_data, fs)
        features['epoch_index'] = index
        features['start_time'] = index * epoch_size
        features['end_time'] = (index + 1) * epoch_size
        features['duration'] = epoch_size
        all_features.append(features)

    return all_features


def finalize_music_window_features(
    window_features: List[Dict[str, Any]],
    smoothing_window: int = 5,
) -> List[Dict[str, Any]]:
    if not window_features:
        return []

    delta = np.asarray([feature.get('delta_power', 0.0) for feature in window_features], dtype=float)
    theta = np.asarray([feature.get('theta_power', 0.0) for feature in window_features], dtype=float)
    alpha = np.asarray([feature.get('alpha_power', 0.0) for feature in window_features], dtype=float)
    beta = np.asarray([feature.get('beta_power', 0.0) for feature in window_features], dtype=float)
    amplitude = np.asarray([feature.get('mean_amplitude', 0.0) for feature in window_features], dtype=float)
    dominant_frequency = np.asarray(
        [feature.get('dominant_frequency', 1.0) for feature in window_features],
        dtype=float,
    )
    slow_wave_ratio = np.asarray([feature.get('slow_wave_ratio', 0.0) for feature in window_features], dtype=float)

    energy_change = np.zeros(len(window_features), dtype=float)
    if len(window_features) > 1:
        previous_amplitude = np.maximum(amplitude[:-1], 1.0)
        energy_change[1:] = np.abs(np.diff(amplitude)) / previous_amplitude

    total_power = delta + theta + alpha + beta + 1e-9
    delta_ratio = delta / total_power
    alpha_ratio = alpha / total_power
    beta_ratio = beta / total_power
    freq_norm = np.clip((dominant_frequency - 0.5) / 3.5, 0.0, 1.0)
    amplitude_norm = _safe_normalize(amplitude)
    energy_norm = _safe_normalize(energy_change)
    slow_wave_norm = _safe_normalize(slow_wave_ratio)

    calmness_raw = (
        0.45 * delta_ratio
        + 0.25 * (1.0 - beta_ratio)
        + 0.15 * (1.0 - energy_norm)
        + 0.15 * slow_wave_norm
    )
    density_raw = (
        0.35 * amplitude_norm
        + 0.30 * energy_norm
        + 0.20 * slow_wave_norm
        + 0.15 * freq_norm
    )
    brightness_raw = (
        0.40 * alpha_ratio
        + 0.25 * beta_ratio
        + 0.20 * freq_norm
        + 0.15 * (1.0 - delta_ratio)
    )

    calmness = _smooth_array(calmness_raw, smoothing_window)
    density = _smooth_array(density_raw, smoothing_window)
    brightness = _smooth_array(brightness_raw, smoothing_window)
    energy_smoothed = _smooth_array(energy_change, smoothing_window)

    finalized = []
    for index, feature in enumerate(window_features):
        enriched = feature.copy()
        enriched['window_index'] = index
        enriched['energy_change'] = float(energy_smoothed[index])
        enriched['calmness'] = float(np.clip(calmness[index], 0.0, 1.0))
        enriched['density'] = float(np.clip(density[index], 0.0, 1.0))
        enriched['brightness'] = float(np.clip(brightness[index], 0.0, 1.0))
        enriched['delta_ratio'] = float(delta_ratio[index])
        enriched['alpha_ratio'] = float(alpha_ratio[index])
        enriched['beta_ratio'] = float(beta_ratio[index])
        finalized.append(enriched)

    return finalized


def extract_music_windows_from_signal(
    eeg: np.ndarray,
    fs: float,
    window_size: float = 2.0,
    hop_size: float = 1.0,
    start_time: float = 0.0,
    smoothing_window: int = 5,
) -> List[Dict[str, Any]]:
    eeg_uv, _, _ = ensure_microvolt_scale(eeg)
    if len(eeg_uv) == 0:
        return []

    window_samples = max(16, int(round(window_size * fs)))
    hop_samples = max(1, int(round(hop_size * fs)))
    signal_duration = len(eeg_uv) / fs

    if len(eeg_uv) <= window_samples:
        start_indices = [0]
    else:
        last_start = max(0, len(eeg_uv) - window_samples)
        start_indices = list(range(0, last_start + 1, hop_samples))
        if start_indices[-1] != last_start and (len(eeg_uv) - (start_indices[-1] + window_samples)) >= (window_samples // 2):
            start_indices.append(last_start)

    window_features = []
    for window_index, start_idx in enumerate(start_indices):
        end_idx = min(len(eeg_uv), start_idx + window_samples)
        if end_idx - start_idx < 16:
            continue

        window = eeg_uv[start_idx:end_idx]
        feature = extract_epoch_features(window, fs)
        feature['window_index'] = window_index
        feature['start_time'] = float(start_time + (start_idx / fs))
        feature['end_time'] = float(start_time + (end_idx / fs))
        feature['duration'] = float((end_idx - start_idx) / fs)
        feature['signal_duration'] = float(signal_duration)
        window_features.append(feature)

    return finalize_music_window_features(window_features, smoothing_window=smoothing_window)


class FeatureExtractor:

    def __init__(self, fs: float, epoch_size: float = 30.0, smoothing_window: int = 5):
        self.fs = fs
        self.epoch_size = epoch_size
        self.smoothing_window = smoothing_window
        self.features_list: List[Dict[str, Any]] = []
        self.raw_features: List[Dict[str, Any]] = []
        self.music_windows: List[Dict[str, Any]] = []

    def extract(self, eeg: np.ndarray, smooth: bool = True) -> List[Dict[str, Any]]:
        eeg_uv, _, _ = ensure_microvolt_scale(eeg)
        self.raw_features = extract_features_from_epochs(eeg_uv, self.fs, self.epoch_size)

        if smooth and len(self.raw_features) >= self.smoothing_window:
            self.features_list = smooth_features(self.raw_features, self.smoothing_window)
        else:
            self.features_list = self.raw_features

        return self.features_list

    def extract_from_segment(self, segment: np.ndarray) -> Dict[str, Any]:
        segment_uv, _, _ = ensure_microvolt_scale(segment)
        features = extract_epoch_features(segment_uv, self.fs)
        features['duration'] = len(segment_uv) / self.fs if self.fs else 0.0
        return features

    def extract_music_windows(
        self,
        eeg: np.ndarray,
        window_size: float = 2.0,
        hop_size: float = 1.0,
        start_time: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self.music_windows = extract_music_windows_from_signal(
            eeg,
            self.fs,
            window_size=window_size,
            hop_size=hop_size,
            start_time=start_time,
            smoothing_window=self.smoothing_window,
        )
        return self.music_windows

    def get_music_features(self) -> List[Dict[str, Any]]:
        music_features = []
        for features in self.features_list:
            music_features.append(
                {
                    'delta_power': features.get('delta_power', 0),
                    'dominant_frequency': features.get('dominant_frequency', 1.0),
                    'mean_amplitude': features.get('mean_amplitude', 0),
                    'avg_instantaneous_amplitude': features.get('avg_instantaneous_amplitude', 0),
                    'slow_wave_ratio': features.get('slow_wave_ratio', 0),
                    'slow_wave_period': features.get('slow_wave_period', 1.0),
                    'start_time': features.get('start_time', 0),
                    'duration': features.get('duration', 30),
                }
            )
        return music_features

    def get_individualized_stats(self) -> Dict[str, Any]:
        if not self.features_list:
            return {
                'avg_delta_power': 0,
                'avg_slow_wave_frequency': 1.0,
                'avg_amplitude': 0,
                'delta_power_level': 'medium',
                'amplitude_level': 'medium',
                'frequency_level': 'medium',
            }

        delta_powers = [feature.get('delta_power', 0) for feature in self.features_list]
        frequencies = [feature.get('dominant_frequency', 1.0) for feature in self.features_list]
        amplitudes = [feature.get('mean_amplitude', 0) for feature in self.features_list]

        avg_delta_power = float(np.mean(delta_powers))
        avg_frequency = float(np.mean(frequencies))
        avg_amplitude = float(np.mean(amplitudes))

        median_delta_power = float(np.median(delta_powers))
        delta_level = classify_relative_level(delta_powers, median_delta_power)

        if avg_amplitude < 50:
            amp_level = 'low'
        elif avg_amplitude > 100:
            amp_level = 'high'
        else:
            amp_level = 'medium'

        if avg_frequency < 0.8:
            freq_level = 'low'
        elif avg_frequency > 1.2:
            freq_level = 'high'
        else:
            freq_level = 'medium'

        return {
            'avg_delta_power': avg_delta_power,
            'avg_slow_wave_frequency': avg_frequency,
            'avg_amplitude': avg_amplitude,
            'delta_power_level': delta_level,
            'amplitude_level': amp_level,
            'frequency_level': freq_level,
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.features_list:
            return {}

        stats = self.get_individualized_stats()
        stats['epoch_count'] = len(self.features_list)
        return stats
