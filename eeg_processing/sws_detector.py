import numpy as np
from typing import List, Dict, Tuple, Any
from scipy import signal
from scipy.signal import hilbert, welch

from eeg_processing.preprocess import bandpass_filter, ensure_microvolt_scale
from utils.config import SWS_DETECTION_CONFIG


def compute_delta_power_welch(eeg_segment: np.ndarray, fs: float, 
                               delta_band: Tuple[float, float] = (0.5, 4.0)) -> float:
    nperseg = min(1024, len(eeg_segment))
    if nperseg < 16:
        return 0.0
    
    freqs, psd = welch(eeg_segment, fs=fs, nperseg=nperseg)
    delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
    if not np.any(delta_mask):
        return 0.0
    delta_power = np.trapz(psd[delta_mask], freqs[delta_mask])
    return delta_power


def detect_slow_waves_hilbert(eeg_segment: np.ndarray, fs: float,
                               amplitude_threshold: float = 75.0,
                               freq_range: Tuple[float, float] = (0.5, 2.0)) -> Dict[str, Any]:
    delta_filtered = bandpass_filter(eeg_segment, freq_range[0], freq_range[1], fs)
    
    analytic_signal = hilbert(delta_filtered)
    instantaneous_amplitude = np.abs(analytic_signal)

    # AASM amplitude threshold is peak-to-peak. Hilbert amplitude approximates peak amplitude.
    peak_to_peak_amplitude = instantaneous_amplitude * 2.0
    is_slow_wave = peak_to_peak_amplitude >= amplitude_threshold
    
    slow_wave_count = 0
    in_wave = False
    wave_durations = []
    wave_start = 0
    
    for i, sw in enumerate(is_slow_wave):
        if sw and not in_wave:
            in_wave = True
            wave_start = i
        elif not sw and in_wave:
            in_wave = False
            duration = (i - wave_start) / fs
            if duration >= 0.5:
                slow_wave_count += 1
                wave_durations.append(duration)
    
    if in_wave:
        duration = (len(is_slow_wave) - wave_start) / fs
        if duration >= 0.5:
            slow_wave_count += 1
            wave_durations.append(duration)
    
    slow_wave_time = np.sum(is_slow_wave) / fs
    total_time = len(eeg_segment) / fs
    slow_wave_ratio = slow_wave_time / total_time if total_time > 0 else 0
    
    avg_amplitude = np.mean(instantaneous_amplitude)
    avg_peak_to_peak_amplitude = np.mean(peak_to_peak_amplitude)
    
    return {
        'slow_wave_count': slow_wave_count,
        'slow_wave_time': slow_wave_time,
        'slow_wave_ratio': slow_wave_ratio,
        'avg_instantaneous_amplitude': avg_amplitude,
        'avg_peak_to_peak_amplitude': avg_peak_to_peak_amplitude,
        'wave_durations': wave_durations
    }


def detect_sws_epoch(eeg_epoch: np.ndarray, fs: float, 
                     config: dict = None) -> Dict[str, Any]:
    if config is None:
        config = SWS_DETECTION_CONFIG
    
    delta_band = config.get('delta_band', (0.5, 4.0))
    amplitude_threshold = config.get('amplitude_threshold_uv', 75.0)
    
    delta_power = compute_delta_power_welch(eeg_epoch, fs, delta_band)
    
    slow_wave_result = detect_slow_waves_hilbert(
        eeg_epoch, fs, amplitude_threshold, (0.5, 2.0)
    )
    
    return {
        'delta_power': delta_power,
        'slow_wave_count': slow_wave_result['slow_wave_count'],
        'slow_wave_time': slow_wave_result['slow_wave_time'],
        'slow_wave_ratio': slow_wave_result['slow_wave_ratio'],
        'avg_instantaneous_amplitude': slow_wave_result['avg_instantaneous_amplitude'],
        'avg_peak_to_peak_amplitude': slow_wave_result['avg_peak_to_peak_amplitude'],
        'wave_durations': slow_wave_result['wave_durations']
    }


def classify_sws_aasm(epoch_result: Dict[str, Any], 
                      delta_power_threshold: float = None,
                      slow_wave_ratio_threshold: float = 0.2) -> bool:
    delta_power = epoch_result.get('delta_power', 0)
    slow_wave_ratio = epoch_result.get('slow_wave_ratio', 0)
    
    if delta_power_threshold is None:
        delta_power_threshold = SWS_DETECTION_CONFIG.get('delta_power_threshold', 0.1)
    
    is_sws = (delta_power > delta_power_threshold and 
              slow_wave_ratio >= slow_wave_ratio_threshold)
    
    return is_sws


class SWSDetector:
    
    def __init__(self, config: dict = None):
        self.config = config or SWS_DETECTION_CONFIG.copy()
        self.epochs = []
        self.sws_epochs = []
        self.total_sws_duration = 0
        self.avg_delta_power = 0
        self.slow_wave_density = 0
    
    def detect(self, eeg: np.ndarray, fs: float) -> Dict[str, Any]:
        eeg_uv, _, _ = ensure_microvolt_scale(eeg)
        window_size = self.config.get('window_size', 30)
        window_samples = int(window_size * fs)
        n_epochs = len(eeg_uv) // window_samples
        
        self.epochs = []
        sws_epoch_indices = []
        
        for i in range(n_epochs):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            epoch_data = eeg_uv[start_idx:end_idx]
            
            epoch_result = detect_sws_epoch(epoch_data, fs, self.config)
            epoch_result['epoch_index'] = i
            epoch_result['start_time'] = i * window_size
            epoch_result['end_time'] = (i + 1) * window_size
            epoch_result['is_sws'] = classify_sws_aasm(
                epoch_result,
                self.config.get('delta_power_threshold', 1e-6),
                self.config.get('slow_wave_ratio_threshold', 0.2)
            )
            
            self.epochs.append(epoch_result)
            
            if epoch_result['is_sws']:
                sws_epoch_indices.append(i)
        
        self.sws_epochs = [self.epochs[i] for i in sws_epoch_indices]
        
        if self.sws_epochs:
            self.total_sws_duration = len(self.sws_epochs) * window_size
            self.avg_delta_power = np.mean([e['delta_power'] for e in self.sws_epochs])
            
            total_slow_waves = sum(e['slow_wave_count'] for e in self.sws_epochs)
            self.slow_wave_density = total_slow_waves / (self.total_sws_duration / 60) if self.total_sws_duration > 0 else 0
        else:
            self.total_sws_duration = 0
            self.avg_delta_power = 0
            self.slow_wave_density = 0
        
        return {
            'total_sws_duration': self.total_sws_duration,
            'avg_delta_power': self.avg_delta_power,
            'slow_wave_density': self.slow_wave_density,
            'sws_epoch_count': len(self.sws_epochs),
            'total_epoch_count': n_epochs,
            'epochs': self.epochs,
            'sws_epochs': self.sws_epochs
        }
    
    def get_sws_eeg_segments(self, eeg: np.ndarray, fs: float) -> List[np.ndarray]:
        eeg_uv, _, _ = ensure_microvolt_scale(eeg)
        sws_segments = []
        window_size = self.config.get('window_size', 30)
        window_samples = int(window_size * fs)
        
        for epoch in self.sws_epochs:
            start_idx = int(epoch['start_time'] * fs)
            end_idx = int(epoch['end_time'] * fs)
            if end_idx <= len(eeg_uv):
                sws_segments.append(eeg_uv[start_idx:end_idx])
        
        return sws_segments
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_sws_duration': self.total_sws_duration,
            'avg_delta_power': self.avg_delta_power,
            'slow_wave_density': self.slow_wave_density,
            'sws_epoch_count': len(self.sws_epochs)
        }
