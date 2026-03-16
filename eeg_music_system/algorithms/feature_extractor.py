from dataclasses import dataclass

import numpy as np


@dataclass
class EEGFeatures:
    amplitude: np.ndarray
    power: np.ndarray
    rhythm: np.ndarray


class FeatureExtractor:
    def extract(self, signal: np.ndarray, sfreq: float) -> EEGFeatures:
        amplitude = np.abs(signal)
        power = signal**2
        if sfreq <= 0:
            sfreq = 100.0
        rhythm = np.full_like(signal, fill_value=1.0 / sfreq, dtype=float)
        return EEGFeatures(amplitude=amplitude, power=power, rhythm=rhythm)
