"""
EEG preprocessing pipeline for sleep analysis.

Steps:
  1. Re-reference to average
  2. Band-pass filter 0.3 – 35 Hz
  3. Artifact detection (amplitude > 200 µV)
"""

import numpy as np
import mne

from utils.logger import logger


def load_eeg(file_path: str) -> mne.io.BaseRaw:
    """Load EEG from EDF, BDF, or CSV.

    Returns an MNE Raw object.
    """
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "edf":
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif ext == "bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
    elif ext == "csv":
        raw = _load_csv_as_raw(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    logger.info(
        "Loaded EEG: %s | sfreq=%.1f Hz | %d channels | %.1f s",
        file_path, raw.info["sfreq"], len(raw.ch_names), raw.times[-1],
    )
    return raw


def _load_csv_as_raw(file_path: str) -> mne.io.RawArray:
    """Load a CSV file as MNE RawArray.

    Expected CSV layout: first column is time (seconds), remaining columns are
    EEG channels. The first row is treated as a header.
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    ch_names = list(df.columns[1:])
    data = df.iloc[:, 1:].values.T  # (n_channels, n_samples)
    # Estimate sampling rate from time column
    times = df.iloc[:, 0].values
    sfreq = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 256.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    # CSV data assumed to be in µV; MNE expects V
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    return raw


def preprocess(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Run the standard sleep-EEG preprocessing pipeline.

    1. Average re-reference
    2. Band-pass filter 0.3 – 35 Hz
    3. Mark artefacts (amplitude > 200 µV)
    """
    # 1. Re-reference to average
    raw.set_eeg_reference("average", projection=False, verbose=False)
    logger.info("Applied average reference.")

    # 2. Band-pass filter
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)
    logger.info("Band-pass filtered 0.3 – 35 Hz.")

    # 3. Artifact annotation (epochs with amplitude > 200 µV)
    raw = _annotate_artifacts(raw, threshold=200e-6)
    return raw


def _annotate_artifacts(raw: mne.io.BaseRaw, threshold: float = 200e-6) -> mne.io.BaseRaw:
    """Mark segments where any EEG channel exceeds *threshold* (in V)."""
    data = raw.get_data()  # (n_channels, n_samples)
    bad_mask = np.any(np.abs(data) > threshold, axis=0)

    if not np.any(bad_mask):
        logger.info("No artefacts detected.")
        return raw

    # Find contiguous bad segments
    onsets: list[float] = []
    durations: list[float] = []
    sfreq = raw.info["sfreq"]
    in_bad = False
    start = 0
    for i, is_bad in enumerate(bad_mask):
        if is_bad and not in_bad:
            start = i
            in_bad = True
        elif not is_bad and in_bad:
            onsets.append(start / sfreq)
            durations.append((i - start) / sfreq)
            in_bad = False
    if in_bad:
        onsets.append(start / sfreq)
        durations.append((len(bad_mask) - start) / sfreq)

    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=["bad_artifact"] * len(onsets),
    )
    raw.set_annotations(raw.annotations + annotations)
    logger.info("Marked %d artefact segments.", len(onsets))
    return raw


def get_eeg_info(raw: mne.io.BaseRaw) -> dict:
    """Return a summary dict of the loaded EEG data."""
    return {
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names,
        "duration_s": raw.times[-1],
    }
