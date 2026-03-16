from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from eeg_music_system.algorithms.feature_extractor import FeatureExtractor
from eeg_music_system.algorithms.music_mapper import MusicMapper
from eeg_music_system.algorithms.sws_detector import SWSDetector, SWSResult
from eeg_music_system.database.db_manager import DBManager
from eeg_music_system.utils.file_manager import FileManager
from eeg_music_system.utils.logger import setup_logger

try:
    import mne
except Exception:  # pragma: no cover - optional runtime dependency
    mne = None


class ExperimentManager:
    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self.logger = setup_logger(__name__)
        self.db = db_manager or DBManager()
        self.db.initialize()
        self.file_manager = FileManager()
        self.sws_detector = SWSDetector()
        self.feature_extractor = FeatureExtractor()
        self.music_mapper = MusicMapper()

    def load_eeg_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".edf", ".bdf"}:
            if mne is None:
                raise RuntimeError("MNE-Python is required to read EDF/BDF files.")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            signal, _times = raw[0, :]
            sfreq = float(raw.info["sfreq"])
            ch_count = int(raw.info["nchan"])
            duration = float(raw.times[-1]) if len(raw.times) else 0.0
            eeg = signal[0]
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("CSV file is empty.")
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                raise ValueError("CSV must include numeric EEG columns.")
            eeg = numeric.iloc[:, 0].to_numpy(dtype=float)
            sfreq = 100.0
            ch_count = numeric.shape[1]
            duration = len(eeg) / sfreq
        else:
            raise ValueError("Unsupported EEG file format. Use EDF/BDF/CSV.")

        self.logger.info("EEG imported: %s", file_path)
        return {
            "signal": eeg,
            "sfreq": sfreq,
            "channel_count": ch_count,
            "duration": duration,
        }

    def run_analysis_and_music(self, subject_id: int, eeg_source_path: str) -> dict[str, Any]:
        eeg_copied_path = self.file_manager.save_eeg_file(subject_id, eeg_source_path)
        eeg_data = self.load_eeg_file(str(eeg_copied_path))
        sws: SWSResult = self.sws_detector.detect(eeg_data["signal"], eeg_data["sfreq"])
        features = self.feature_extractor.extract(sws.filtered_signal, eeg_data["sfreq"])

        subject_dir = self.file_manager.ensure_subject_dir(subject_id)
        midi_path = subject_dir / "brain_music.mid"
        wav_path = subject_dir / "brain_music.wav"
        self.music_mapper.generate_midi(features, midi_path)
        self.music_mapper.midi_to_wav(midi_path, wav_path)

        exp = self.db.add_experiment(
            subject_id=subject_id,
            eeg_path=str(eeg_copied_path),
            midi_path=str(midi_path),
            wav_path=str(wav_path),
            sws_duration=sws.sws_duration,
            avg_delta_power=sws.avg_delta_power,
            slow_wave_density=sws.slow_wave_density,
        )
        self.logger.info("Experiment saved: subject=%s exp=%s", subject_id, exp.id)
        return {
            "experiment_id": exp.id,
            "eeg": eeg_data,
            "sws": asdict(sws),
            "midi_path": str(midi_path),
            "wav_path": str(wav_path),
        }

    def export_experiments_csv(self, out_path: str, subject_id: Optional[int] = None) -> str:
        experiments = self.db.list_experiments(subject_id=subject_id)
        rows = [
            {
                "subject_id": exp.subject_id,
                "experiment_time": exp.created_time.isoformat(),
                "sws_duration": exp.sws_duration,
                "avg_delta_power": exp.avg_delta_power,
                "slow_wave_density": exp.slow_wave_density,
            }
            for exp in experiments
        ]
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path
