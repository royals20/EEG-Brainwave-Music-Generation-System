"""
Slow-wave sleep (SWS / N3) detector.

Detects individual slow waves in N3 epochs:
  - Delta band: 0.5 – 4 Hz
  - Negative peak amplitude > 75 µV
  - Duration: 0.25 – 1 s
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import butter, sosfiltfilt

from utils.logger import logger


@dataclass
class SlowWaveEvent:
    """Single detected slow wave."""
    peak_time: float       # seconds from recording start
    amplitude: float       # µV (positive value = magnitude)
    duration: float        # seconds
    frequency: float       # Hz (1 / duration)


def detect_slow_waves(
    data: np.ndarray,
    sfreq: float,
    hypnogram: np.ndarray,
    amplitude_threshold: float = 75.0,
    duration_range: tuple = (0.25, 1.0),
) -> List[SlowWaveEvent]:
    """Detect slow-wave events in N3 (stage 3) epochs.

    Parameters
    ----------
    data : 1-D ndarray
        Single-channel EEG data in **µV**.
    sfreq : float
        Sampling frequency (Hz).
    hypnogram : 1-D ndarray of int
        One label per 30-s epoch.
    amplitude_threshold : float
        Minimum negative-peak magnitude in µV.
    duration_range : tuple of float
        (min_dur, max_dur) in seconds for a valid slow wave.

    Returns
    -------
    events : list of SlowWaveEvent
    """
    # Build a mask of samples that belong to N3 epochs
    epoch_samples = int(30 * sfreq)
    n3_mask = np.zeros(len(data), dtype=bool)
    for i, stage in enumerate(hypnogram):
        if int(stage) == 3:
            start = i * epoch_samples
            end = min(start + epoch_samples, len(data))
            n3_mask[start:end] = True

    if not np.any(n3_mask):
        logger.info("No N3 epochs found – no slow waves to detect.")
        return []

    # Delta band-pass 0.5 – 4 Hz
    filtered = _bandpass(data, sfreq, low=0.5, high=4.0)

    # Detect negative peaks (troughs)
    events: List[SlowWaveEvent] = []
    min_dur, max_dur = duration_range
    min_samples = int(min_dur * sfreq)
    max_samples = int(max_dur * sfreq)

    # Find zero-crossings (negative-going and positive-going)
    sign = np.sign(filtered)
    # Indices where sign changes from positive to negative
    neg_crossings = np.where((sign[:-1] >= 0) & (sign[1:] < 0))[0]
    pos_crossings = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]

    for nc in neg_crossings:
        # Find the next positive crossing after this negative crossing
        future_pos = pos_crossings[pos_crossings > nc]
        if len(future_pos) == 0:
            continue
        pc = future_pos[0]
        duration_samples = pc - nc
        if duration_samples < min_samples or duration_samples > max_samples:
            continue

        # Check the negative peak in between
        segment = filtered[nc:pc]
        trough_idx = nc + np.argmin(segment)

        # Must be within N3
        if not n3_mask[trough_idx]:
            continue

        amp = abs(filtered[trough_idx])  # µV magnitude
        if amp < amplitude_threshold:
            continue

        peak_time = trough_idx / sfreq
        duration = duration_samples / sfreq
        freq = 1.0 / duration

        events.append(SlowWaveEvent(
            peak_time=peak_time,
            amplitude=amp,
            duration=duration,
            frequency=freq,
        ))

    logger.info("Detected %d slow-wave events in N3.", len(events))
    return events


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter."""
    sos = butter(order, [low, high], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data)
