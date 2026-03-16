"""
Automatic sleep staging using YASA.

Produces a hypnogram with 30-second epochs:
  W, N1, N2, N3 (SWS), R
"""

from typing import Optional

import numpy as np
import mne

from utils.logger import logger

# Stage label mapping used throughout the system
STAGE_LABELS = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
STAGE_NAMES = {v: k for k, v in STAGE_LABELS.items()}


def run_sleep_staging(raw: mne.io.BaseRaw, eeg_name: Optional[str] = None,
                      eog_name: Optional[str] = None,
                      emg_name: Optional[str] = None) -> np.ndarray:
    """Perform automatic sleep staging and return an integer hypnogram.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Pre-processed EEG.
    eeg_name : str, optional
        Name of the EEG channel to use. If *None*, the first channel is used.
    eog_name : str, optional
        Name of EOG channel (improves staging accuracy).
    emg_name : str, optional
        Name of EMG channel (improves staging accuracy).

    Returns
    -------
    hypnogram : np.ndarray of int
        One value per 30-second epoch. Uses STAGE_LABELS encoding.
    """
    try:
        import yasa
    except ImportError:
        logger.warning("YASA not available – falling back to simple delta-power staging.")
        return _simple_staging(raw, eeg_name)

    if eeg_name is None:
        eeg_name = raw.ch_names[0]

    sls = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name)
    hyp_str = sls.predict()  # array of strings like 'W', 'N1', ...

    mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    hypnogram = np.array([mapping.get(s, 0) for s in hyp_str], dtype=int)
    logger.info("Sleep staging complete – %d epochs, stages: %s",
                len(hypnogram), dict(zip(*np.unique(hypnogram, return_counts=True))))
    return hypnogram


def _simple_staging(raw: mne.io.BaseRaw, eeg_name: Optional[str] = None) -> np.ndarray:
    """Fallback staging based on delta-power thresholding.

    This is a simplified heuristic when YASA is not installed.
    """
    from scipy.signal import welch

    if eeg_name is None:
        eeg_name = raw.ch_names[0]

    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=[eeg_name])[0]
    epoch_len = int(30 * sfreq)
    n_epochs = len(data) // epoch_len
    hypnogram = np.zeros(n_epochs, dtype=int)

    for i in range(n_epochs):
        segment = data[i * epoch_len:(i + 1) * epoch_len]
        freqs, psd = welch(segment, fs=sfreq, nperseg=min(len(segment), int(4 * sfreq)))
        delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
        total_mask = (freqs >= 0.5) & (freqs <= 30.0)
        delta_ratio = np.sum(psd[delta_mask]) / (np.sum(psd[total_mask]) + 1e-12)

        if delta_ratio > 0.6:
            hypnogram[i] = 3  # N3
        elif delta_ratio > 0.4:
            hypnogram[i] = 2  # N2
        elif delta_ratio > 0.25:
            hypnogram[i] = 1  # N1
        else:
            hypnogram[i] = 0  # Wake

    logger.info("Simple staging: %d epochs classified.", n_epochs)
    return hypnogram


def hypnogram_to_labels(hypnogram: np.ndarray) -> list[str]:
    """Convert integer hypnogram to string labels."""
    return [STAGE_NAMES.get(int(v), "W") for v in hypnogram]
