from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert


@dataclass
class SWSResult:
    filtered_signal: np.ndarray
    delta_band: np.ndarray
    slow_wave_indices: np.ndarray
    sws_duration: float
    avg_delta_power: float
    slow_wave_density: float


class SWSDetector:
    def __init__(self, lowcut: float = 0.5, highcut: float = 4.0) -> None:
        self.lowcut = lowcut
        self.highcut = highcut

    def _bandpass_filter(self, signal: np.ndarray, sfreq: float) -> np.ndarray:
        nyquist = 0.5 * sfreq
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def detect(self, signal: np.ndarray, sfreq: float) -> SWSResult:
        if signal.ndim > 1:
            signal = signal[0]
        filtered = self._bandpass_filter(signal, sfreq)
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        threshold = envelope.mean() + envelope.std()
        peaks, _ = find_peaks(envelope, height=threshold, distance=max(1, int(sfreq / 2)))

        sws_duration = float(len(peaks)) * 0.8
        avg_delta_power = float(np.mean(filtered**2))
        duration_minutes = max(len(filtered) / sfreq / 60.0, 1e-6)
        slow_wave_density = float(len(peaks) / duration_minutes)

        return SWSResult(
            filtered_signal=filtered,
            delta_band=filtered,
            slow_wave_indices=peaks,
            sws_duration=sws_duration,
            avg_delta_power=avg_delta_power,
            slow_wave_density=slow_wave_density,
        )
