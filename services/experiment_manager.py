"""
Experiment manager: orchestrates the analysis pipeline for a subject.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import mne

from algorithms.eeg_preprocess import load_eeg, preprocess, get_eeg_info
from algorithms.sleep_staging import run_sleep_staging
from algorithms.sws_detector import detect_slow_waves, SlowWaveEvent
from algorithms.feature_extractor import extract_features, SWSFeatures
from algorithms.music_mapper import map_features_to_notes, NoteEvent
from algorithms.midi_generator import generate_midi
from algorithms.audio_synthesizer import midi_to_wav, synthesise_from_notes
from database.db_manager import DatabaseManager
from utils.file_manager import get_subject_directory, copy_file_to_subject
from utils.logger import logger


class ExperimentManager:
    """High-level service that drives the full analysis workflow."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.raw: Optional[mne.io.BaseRaw] = None
        self.hypnogram: Optional[np.ndarray] = None
        self.events: list[SlowWaveEvent] = []
        self.features: Optional[SWSFeatures] = None
        self.notes: list[NoteEvent] = []
        self.current_subject_id: Optional[str] = None
        self.eeg_path: Optional[str] = None
        self.midi_path: Optional[str] = None
        self.wav_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Step 1: Import EEG
    # ------------------------------------------------------------------
    def import_eeg(self, file_path: str, subject_id: str) -> dict:
        """Import and copy EEG file into the subject directory."""
        self.current_subject_id = subject_id
        dest = copy_file_to_subject(file_path, subject_id)
        self.eeg_path = dest
        self.raw = load_eeg(dest)
        info = get_eeg_info(self.raw)
        logger.info("EEG imported for subject %s: %s", subject_id, info)
        return info

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------
    def preprocess_eeg(self) -> None:
        """Run the preprocessing pipeline."""
        if self.raw is None:
            raise RuntimeError("No EEG loaded. Call import_eeg first.")
        self.raw = preprocess(self.raw)
        logger.info("EEG preprocessing complete.")

    # ------------------------------------------------------------------
    # Step 3: Sleep staging
    # ------------------------------------------------------------------
    def run_staging(self, eeg_name: Optional[str] = None) -> np.ndarray:
        """Perform automatic sleep staging."""
        if self.raw is None:
            raise RuntimeError("No EEG loaded.")
        self.hypnogram = run_sleep_staging(self.raw, eeg_name=eeg_name)
        return self.hypnogram

    # ------------------------------------------------------------------
    # Step 4: SWS detection
    # ------------------------------------------------------------------
    def detect_sws(self, channel: Optional[str] = None) -> list[SlowWaveEvent]:
        """Detect slow waves in N3 epochs."""
        if self.raw is None or self.hypnogram is None:
            raise RuntimeError("Run staging first.")
        ch = channel or self.raw.ch_names[0]
        # Get data in µV
        data = self.raw.get_data(picks=[ch])[0] * 1e6
        self.events = detect_slow_waves(data, self.raw.info["sfreq"], self.hypnogram)
        return self.events

    # ------------------------------------------------------------------
    # Step 5: Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, channel: Optional[str] = None) -> SWSFeatures:
        """Extract SWS features."""
        if self.raw is None or self.hypnogram is None:
            raise RuntimeError("Run staging first.")
        ch = channel or self.raw.ch_names[0]
        data = self.raw.get_data(picks=[ch])[0] * 1e6
        self.features = extract_features(data, self.raw.info["sfreq"], self.hypnogram, self.events)
        return self.features

    # ------------------------------------------------------------------
    # Step 6: Music generation
    # ------------------------------------------------------------------
    def generate_music(self) -> tuple[str, str]:
        """Generate MIDI and WAV from features and slow-wave events."""
        if self.features is None:
            raise RuntimeError("Extract features first.")
        if self.current_subject_id is None:
            raise RuntimeError("No subject set.")

        self.notes = map_features_to_notes(self.features, self.events)
        subj_dir = get_subject_directory(self.current_subject_id)

        self.midi_path = os.path.join(subj_dir, "brainwave_music.mid")
        generate_midi(self.notes, self.midi_path)

        self.wav_path = os.path.join(subj_dir, "brainwave_music.wav")
        try:
            midi_to_wav(self.midi_path, self.wav_path)
        except Exception:
            synthesise_from_notes(self.notes, self.wav_path)

        return self.midi_path, self.wav_path

    # ------------------------------------------------------------------
    # Step 7: Save experiment
    # ------------------------------------------------------------------
    def save_experiment(self) -> str:
        """Persist the experiment record to the database."""
        exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        self.db.add_experiment(
            experiment_id=exp_id,
            subject_id=self.current_subject_id or "",
            eeg_path=self.eeg_path or "",
            midi_path=self.midi_path or "",
            wav_path=self.wav_path or "",
            sws_duration=self.features.sws_duration if self.features else 0,
            delta_power=self.features.delta_power if self.features else 0,
            slow_wave_density=self.features.slow_wave_density if self.features else 0,
            slow_wave_amplitude=self.features.slow_wave_amplitude if self.features else 0,
        )
        logger.info("Experiment %s saved.", exp_id)
        return exp_id

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self, file_path: str, subject_id: str,
                          channel: Optional[str] = None) -> str:
        """Run the entire analysis pipeline end-to-end."""
        self.import_eeg(file_path, subject_id)
        self.preprocess_eeg()
        self.run_staging(eeg_name=channel)
        self.detect_sws(channel=channel)
        self.extract_features(channel=channel)
        self.generate_music()
        return self.save_experiment()
