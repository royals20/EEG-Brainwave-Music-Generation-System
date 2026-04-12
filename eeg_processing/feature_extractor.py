import numpy as np
from scipy import signal
from scipy.signal import hilbert, welch
from typing import Dict, Any, List, Tuple

from eeg_processing.preprocess import bandpass_filter


def smooth_features(features: List[Dict[str, Any]], 
                    window_size: int = 5) -> List[Dict[str, Any]]:
    if len(features) < window_size:
        return features
    
    smoothed = []
    half_window = window_size // 2
    
    numeric_keys = ['delta_power', 'theta_power', 'alpha_power', 'beta_power',
                    'dominant_frequency', 'mean_amplitude', 'slow_wave_ratio',
                    'avg_instantaneous_amplitude']
    
    for i, feature in enumerate(features):
        start_idx = max(0, i - half_window)
        end_idx = min(len(features), i + half_window + 1)
        
        smoothed_feature = feature.copy()
        
        for key in numeric_keys:
            if key in feature:
                values = [features[j].get(key, 0) for j in range(start_idx, end_idx)]
                smoothed_feature[key] = np.mean(values)
        
        smoothed.append(smoothed_feature)
    
    return smoothed


def extract_epoch_features(epoch_data: np.ndarray, fs: float) -> Dict[str, Any]:
    features = {}
    
    delta_band = (0.5, 4.0)
    theta_band = (4.0, 8.0)
    alpha_band = (8.0, 13.0)
    beta_band = (13.0, 30.0)
    
    nperseg = min(1024, len(epoch_data))
    if nperseg >= 16:
        freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg)
        
        for band_name, (low, high) in [('delta', delta_band), ('theta', theta_band),
                                        ('alpha', alpha_band), ('beta', beta_band)]:
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features[f'{band_name}_power'] = np.trapz(psd[mask], freqs[mask])
            else:
                features[f'{band_name}_power'] = 0.0
        
        delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
        if np.any(delta_mask):
            delta_freqs = freqs[delta_mask]
            delta_psd = psd[delta_mask]
            features['dominant_frequency'] = delta_freqs[np.argmax(delta_psd)]
        else:
            features['dominant_frequency'] = 1.0
    else:
        features['delta_power'] = 0.0
        features['theta_power'] = 0.0
        features['alpha_power'] = 0.0
        features['beta_power'] = 0.0
        features['dominant_frequency'] = 1.0
    
    features['mean_amplitude'] = np.mean(np.abs(epoch_data))
    features['rms_amplitude'] = np.sqrt(np.mean(epoch_data ** 2))
    
    delta_filtered = bandpass_filter(epoch_data, 0.5, 2.0, fs)
    analytic_signal = hilbert(delta_filtered)
    instantaneous_amplitude = np.abs(analytic_signal)
    features['avg_instantaneous_amplitude'] = np.mean(instantaneous_amplitude)
    
    is_slow_wave = instantaneous_amplitude >= 75.0
    features['slow_wave_ratio'] = np.sum(is_slow_wave) / len(is_slow_wave)
    
    features['slow_wave_period'] = 1.0 / (features['dominant_frequency'] + 0.001)
    
    return features


def extract_features_from_epochs(eeg: np.ndarray, fs: float, 
                                  epoch_size: float = 30.0) -> List[Dict[str, Any]]:
    epoch_samples = int(epoch_size * fs)
    n_epochs = len(eeg) // epoch_samples
    
    all_features = []
    
    for i in range(n_epochs):
        start_idx = i * epoch_samples
        end_idx = start_idx + epoch_samples
        epoch_data = eeg[start_idx:end_idx]
        
        features = extract_epoch_features(epoch_data, fs)
        features['epoch_index'] = i
        features['start_time'] = i * epoch_size
        features['end_time'] = (i + 1) * epoch_size
        features['duration'] = epoch_size
        
        all_features.append(features)
    
    return all_features


class FeatureExtractor:
    
    def __init__(self, fs: float, epoch_size: float = 30.0, smoothing_window: int = 5):
        self.fs = fs
        self.epoch_size = epoch_size
        self.smoothing_window = smoothing_window
        self.features_list = []
        self.raw_features = []
    
    def extract(self, eeg: np.ndarray, smooth: bool = True) -> List[Dict[str, Any]]:
        self.raw_features = extract_features_from_epochs(eeg, self.fs, self.epoch_size)
        
        if smooth and len(self.raw_features) >= self.smoothing_window:
            self.features_list = smooth_features(self.raw_features, self.smoothing_window)
        else:
            self.features_list = self.raw_features
        
        return self.features_list
    
    def extract_from_segment(self, segment: np.ndarray) -> Dict[str, Any]:
        features = extract_epoch_features(segment, self.fs)
        features['duration'] = len(segment) / self.fs
        return features
    
    def get_music_features(self) -> List[Dict[str, Any]]:
        music_features = []
        
        for features in self.features_list:
            music_feature = {
                'delta_power': features.get('delta_power', 0),
                'dominant_frequency': features.get('dominant_frequency', 1.0),
                'mean_amplitude': features.get('mean_amplitude', 0),
                'avg_instantaneous_amplitude': features.get('avg_instantaneous_amplitude', 0),
                'slow_wave_ratio': features.get('slow_wave_ratio', 0),
                'slow_wave_period': features.get('slow_wave_period', 1.0),
                'start_time': features.get('start_time', 0),
                'duration': features.get('duration', 30)
            }
            music_features.append(music_feature)
        
        return music_features
    
    def get_individualized_stats(self) -> Dict[str, Any]:
        if not self.features_list:
            return {
                'avg_delta_power': 0,
                'avg_slow_wave_frequency': 1.0,
                'avg_amplitude': 0,
                'delta_power_level': 'medium',
                'amplitude_level': 'medium',
                'frequency_level': 'medium'
            }
        
        delta_powers = [f.get('delta_power', 0) for f in self.features_list]
        frequencies = [f.get('dominant_frequency', 1.0) for f in self.features_list]
        amplitudes = [f.get('mean_amplitude', 0) for f in self.features_list]
        
        avg_delta_power = np.mean(delta_powers)
        avg_frequency = np.mean(frequencies)
        avg_amplitude = np.mean(amplitudes)
        
        if avg_delta_power < np.percentile(delta_powers, 33) if len(delta_powers) > 2 else avg_delta_power < 1e-5:
            delta_level = 'low'
        elif avg_delta_power > np.percentile(delta_powers, 67) if len(delta_powers) > 2 else avg_delta_power > 1e-4:
            delta_level = 'high'
        else:
            delta_level = 'medium'
        
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
            'frequency_level': freq_level
        }
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.features_list:
            return {}
        
        stats = self.get_individualized_stats()
        stats['epoch_count'] = len(self.features_list)
        
        return stats
