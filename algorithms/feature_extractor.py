"""
Feature extraction from slow-wave events and EEG data.

Features:
  - delta_power
  - slow_wave_amplitude (mean)
  - slow_wave_frequency  (mean)
  - slow_wave_density    (events per minute of N3)
  - sws_duration         (total N3 duration in minutes)

All features are also returned in normalised (0–1) form via Min-Max scaling.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import welch

from algorithms.sws_detector import SlowWaveEvent
from utils.logger import logger


@dataclass
class SWSFeatures:
    """Extracted slow-wave sleep features."""
    delta_power: float
    slow_wave_amplitude: float
    slow_wave_frequency: float
    slow_wave_density: float
    sws_duration: float  # minutes

    # Normalised versions (0–1)
    delta_power_norm: float = 0.0
    slow_wave_amplitude_norm: float = 0.0
    slow_wave_frequency_norm: float = 0.0
    slow_wave_density_norm: float = 0.0


# Typical physiological ranges for Min-Max normalisation
_RANGES = {
    "delta_power": (0.0, 500.0),          # µV² / Hz (approximate)
    "slow_wave_amplitude": (75.0, 300.0),  # µV
    "slow_wave_frequency": (0.5, 4.0),     # Hz
    "slow_wave_density": (0.0, 20.0),      # events / min
}


def extract_features(
    data: np.ndarray,
    sfreq: float,
    hypnogram: np.ndarray,
    events: List[SlowWaveEvent],
) -> SWSFeatures:
    """Compute SWS features from EEG data and detected slow-wave events.

    Parameters
    ----------
    data : 1-D ndarray
        Single-channel EEG in **µV**.
    sfreq : float
        Sampling frequency.
    hypnogram : 1-D ndarray of int
        Stage labels per 30-s epoch.
    events : list of SlowWaveEvent
        Detected slow-wave events.
    """
    # SWS duration (N3)
    n3_epochs = int(np.sum(hypnogram == 3))
    sws_duration_min = (n3_epochs * 30.0) / 60.0

    # Delta power in N3 segments
    delta_power = _compute_delta_power(data, sfreq, hypnogram)

    # Mean slow-wave amplitude and frequency
    if events:
        sw_amp = float(np.mean([e.amplitude for e in events]))
        sw_freq = float(np.mean([e.frequency for e in events]))
        sw_density = len(events) / max(sws_duration_min, 1e-6)
    else:
        sw_amp = 0.0
        sw_freq = 0.0
        sw_density = 0.0

    features = SWSFeatures(
        delta_power=delta_power,
        slow_wave_amplitude=sw_amp,
        slow_wave_frequency=sw_freq,
        slow_wave_density=sw_density,
        sws_duration=sws_duration_min,
    )

    # Normalise
    features.delta_power_norm = _minmax(delta_power, *_RANGES["delta_power"])
    features.slow_wave_amplitude_norm = _minmax(sw_amp, *_RANGES["slow_wave_amplitude"])
    features.slow_wave_frequency_norm = _minmax(sw_freq, *_RANGES["slow_wave_frequency"])
    features.slow_wave_density_norm = _minmax(sw_density, *_RANGES["slow_wave_density"])

    logger.info(
        "Features: delta=%.2f, amp=%.2f, freq=%.2f, density=%.2f, SWS=%.1f min",
        delta_power, sw_amp, sw_freq, sw_density, sws_duration_min,
    )
    return features


def _compute_delta_power(data: np.ndarray, sfreq: float, hypnogram: np.ndarray) -> float:
    """Average delta-band power (0.5–4 Hz) across N3 epochs."""
    epoch_len = int(30 * sfreq)
    delta_powers: list[float] = []
    for i, stage in enumerate(hypnogram):
        if int(stage) != 3:
            continue
        start = i * epoch_len
        end = min(start + epoch_len, len(data))
        seg = data[start:end]
        if len(seg) < sfreq:
            continue
        freqs, psd = welch(seg, fs=sfreq, nperseg=min(len(seg), int(4 * sfreq)))
        mask = (freqs >= 0.5) & (freqs <= 4.0)
        delta_powers.append(float(np.mean(psd[mask])))
    return float(np.mean(delta_powers)) if delta_powers else 0.0


def _minmax(value: float, vmin: float, vmax: float) -> float:
    """Min-Max normalisation to [0, 1]."""
    if vmax <= vmin:
        return 0.0
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
