import numpy as np
from scipy import signal
from typing import Tuple, Optional


def infer_eeg_unit(eeg: np.ndarray) -> Tuple[float, str]:
    if eeg is None:
        return 1.0, 'uV'

    eeg = np.asarray(eeg, dtype=float)
    if eeg.size == 0:
        return 1.0, 'uV'

    robust_abs = float(np.percentile(np.abs(eeg), 95))
    if robust_abs == 0:
        return 1.0, 'uV'

    if robust_abs < 1e-3:
        return 1e6, 'V'
    if robust_abs < 1e-1:
        return 1e3, 'mV'
    return 1.0, 'uV'


def ensure_microvolt_scale(eeg: np.ndarray) -> Tuple[np.ndarray, float, str]:
    eeg = np.asarray(eeg, dtype=float)
    scale_factor, source_unit = infer_eeg_unit(eeg)
    return eeg * scale_factor, scale_factor, source_unit


def _can_apply_zero_phase_filter(b: np.ndarray, a: np.ndarray, sample_count: int) -> bool:
    padlen = 3 * max(len(a), len(b))
    return sample_count > padlen


def bandpass_filter(eeg: np.ndarray, low: float, high: float, 
                    fs: float, order: int = 4) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=float)
    if eeg.size == 0:
        return eeg.copy()

    nyquist = fs / 2
    low_normalized = low / nyquist
    high_normalized = high / nyquist
    
    low_normalized = max(0.001, min(low_normalized, 0.999))
    high_normalized = max(low_normalized + 0.001, min(high_normalized, 0.999))
    
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    if not _can_apply_zero_phase_filter(b, a, eeg.size):
        return eeg.copy()

    filtered = signal.filtfilt(b, a, eeg)
    
    return filtered


def notch_filter(eeg: np.ndarray, freq: float, fs: float, 
                 quality_factor: float = 30) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=float)
    if eeg.size == 0:
        return eeg.copy()

    b, a = signal.iirnotch(freq, quality_factor, fs)
    if not _can_apply_zero_phase_filter(b, a, eeg.size):
        return eeg.copy()
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
