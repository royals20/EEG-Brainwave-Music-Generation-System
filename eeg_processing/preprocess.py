import numpy as np
from scipy import signal
from typing import Tuple, Optional


def bandpass_filter(eeg: np.ndarray, low: float, high: float, 
                    fs: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2
    low_normalized = low / nyquist
    high_normalized = high / nyquist
    
    low_normalized = max(0.001, min(low_normalized, 0.999))
    high_normalized = max(low_normalized + 0.001, min(high_normalized, 0.999))
    
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    filtered = signal.filtfilt(b, a, eeg)
    
    return filtered


def notch_filter(eeg: np.ndarray, freq: float, fs: float, 
                 quality_factor: float = 30) -> np.ndarray:
    b, a = signal.iirnotch(freq, quality_factor, fs)
    filtered = signal.filtfilt(b, a, eeg)
    return filtered


def normalize_signal(eeg: np.ndarray, method: str = 'zscore') -> np.ndarray:
    if method == 'zscore':
        return (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-10)
    elif method == 'minmax':
        return (eeg - np.min(eeg)) / (np.max(eeg) - np.min(eeg) + 1e-10)
    elif method == 'amplitude':
        return eeg / (np.max(np.abs(eeg)) + 1e-10)
    else:
        return eeg


def downsample_signal(eeg: np.ndarray, original_fs: float, 
                      target_fs: float) -> np.ndarray:
    if target_fs >= original_fs:
        return eeg
    
    ratio = int(original_fs / target_fs)
    return signal.resample(eeg, len(eeg) // ratio)


def remove_artifacts(eeg: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    z_scores = np.abs((eeg - np.mean(eeg)) / (np.std(eeg) + 1e-10))
    artifact_mask = z_scores > threshold
    
    cleaned = eeg.copy()
    artifact_indices = np.where(artifact_mask)[0]
    
    for idx in artifact_indices:
        start = max(0, idx - 10)
        end = min(len(eeg), idx + 10)
        cleaned[idx] = np.median(eeg[start:end])
    
    return cleaned


def compute_power_spectrum(eeg: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(eeg, fs=fs, nperseg=min(1024, len(eeg)))
    return freqs, psd


def compute_band_power(eeg: np.ndarray, fs: float, 
                       low: float, high: float) -> float:
    freqs, psd = compute_power_spectrum(eeg, fs)
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = np.trapz(psd[band_mask], freqs[band_mask])
    return band_power


class Preprocessor:
    
    def __init__(self, sample_rate: float):
        self.fs = sample_rate
        self.processing_steps = []
    
    def add_bandpass(self, low: float, high: float, order: int = 4):
        self.processing_steps.append(('bandpass', low, high, order))
    
    def add_notch(self, freq: float, quality_factor: float = 30):
        self.processing_steps.append(('notch', freq, quality_factor))
    
    def add_normalization(self, method: str = 'zscore'):
        self.processing_steps.append(('normalize', method))
    
    def add_artifact_removal(self, threshold: float = 3.0):
        self.processing_steps.append(('artifact', threshold))
    
    def process(self, eeg: np.ndarray) -> np.ndarray:
        result = eeg.copy()
        
        for step in self.processing_steps:
            if step[0] == 'bandpass':
                _, low, high, order = step
                result = bandpass_filter(result, low, high, self.fs, order)
            elif step[0] == 'notch':
                _, freq, qf = step
                result = notch_filter(result, freq, self.fs, qf)
            elif step[0] == 'normalize':
                _, method = step
                result = normalize_signal(result, method)
            elif step[0] == 'artifact':
                _, threshold = step
                result = remove_artifacts(result, threshold)
        
        return result
    
    def clear_steps(self):
        self.processing_steps = []
