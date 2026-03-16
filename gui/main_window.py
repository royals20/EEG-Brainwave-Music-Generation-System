"""
Main application window.

Layout:
  - Left: Subject panel
  - Right: Tabbed panels (EEG Import, EEG/Hypnogram, SWS Analysis,
    Music Generation, Experiment Records)
"""

from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QStatusBar, QMessageBox, QApplication, QSplitter, QLabel, QProgressBar,
)

from database.db_manager import DatabaseManager
from services.experiment_manager import ExperimentManager
from gui.subject_panel import SubjectPanel
from gui.eeg_panel import EEGPanel
from gui.sleep_stage_panel import SleepStagePanel
from gui.analysis_panel import AnalysisPanel
from gui.music_panel import MusicPanel
from gui.experiment_panel import ExperimentPanel
from utils.file_manager import ensure_directories
from utils.logger import logger

import numpy as np
from scipy.signal import butter, sosfiltfilt


class _Worker(QThread):
    """Generic worker thread for long-running tasks."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Worker error")
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    """Top-level window for the EEG Sleep Music system."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEG Sleep Analysis & Brainwave Music Generation")
        self.setMinimumSize(1280, 800)

        ensure_directories()

        # Core services
        self.db = DatabaseManager()
        self.experiment = ExperimentManager(self.db)
        self._current_subject: Optional[str] = None
        self._worker: Optional[_Worker] = None

        self._build_ui()
        self._connect_signals()
        self._apply_style()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left: subjects
        self.subject_panel = SubjectPanel(self.db)
        self.subject_panel.setMinimumWidth(260)
        self.subject_panel.setMaximumWidth(400)
        splitter.addWidget(self.subject_panel)

        # Right: tabs
        self.tabs = QTabWidget()
        self.eeg_panel = EEGPanel()
        self.stage_panel = SleepStagePanel()
        self.analysis_panel = AnalysisPanel()
        self.music_panel = MusicPanel()
        self.experiment_panel = ExperimentPanel(self.db)

        self.tabs.addTab(self.eeg_panel, "EEG Import")
        self.tabs.addTab(self.stage_panel, "EEG / Hypnogram")
        self.tabs.addTab(self.analysis_panel, "SWS Analysis")
        self.tabs.addTab(self.music_panel, "Music Generation")
        self.tabs.addTab(self.experiment_panel, "Experiments")
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Progress bar in status bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.subject_panel.subject_selected.connect(self._on_subject_selected)
        self.eeg_panel.eeg_loaded.connect(self._on_eeg_loaded)

        # Run preprocessing + staging when user clicks a (future) button, or
        # after EEG load we kick it off automatically
        self.music_panel.gen_btn.clicked.connect(self._on_generate_music)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_subject_selected(self, subject_id: str) -> None:
        self._current_subject = subject_id
        self.experiment_panel.set_subject(subject_id)
        self.statusBar().showMessage(f"Subject: {subject_id}")

    @Slot(str)
    def _on_eeg_loaded(self, path: str) -> None:
        if self._current_subject is None:
            QMessageBox.warning(self, "Warning", "Please select a subject first.")
            return
        self._set_busy(True, "Importing EEG…")
        self._worker = _Worker(self._import_and_stage, path)
        self._worker.finished.connect(self._on_pipeline_done)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _import_and_stage(self, path: str) -> dict:
        """Run import → preprocess → staging → detect → extract (in worker thread)."""
        info = self.experiment.import_eeg(path, self._current_subject)
        self.experiment.preprocess_eeg()
        channel = self.eeg_panel.selected_channel()
        self.experiment.run_staging(eeg_name=channel)
        self.experiment.detect_sws(channel=channel)
        self.experiment.extract_features(channel=channel)
        return info

    @Slot(object)
    def _on_pipeline_done(self, info: dict) -> None:
        self._set_busy(False)
        # Update EEG panel info
        self.eeg_panel.update_info(info)

        # Plot EEG and hypnogram
        raw = self.experiment.raw
        if raw is not None:
            ch = self.eeg_panel.selected_channel() or raw.ch_names[0]
            data_uv = raw.get_data(picks=[ch])[0] * 1e6
            self.stage_panel.plot_eeg(data_uv, raw.info["sfreq"])

            # Delta band overlay
            delta = self._bandpass(data_uv, raw.info["sfreq"], 0.5, 4.0)
            self.stage_panel.plot_delta_band(delta, raw.info["sfreq"])

        if self.experiment.hypnogram is not None:
            self.stage_panel.plot_hypnogram(self.experiment.hypnogram)

        # Slow-wave markers
        if self.experiment.events:
            times = [e.peak_time for e in self.experiment.events]
            amps = [e.amplitude for e in self.experiment.events]
            self.stage_panel.plot_slow_waves(times, amps)

        # Analysis panel
        if self.experiment.features:
            self.analysis_panel.update_features(self.experiment.features)
        if self.experiment.events:
            self.analysis_panel.update_events(self.experiment.events)

        self.tabs.setCurrentIndex(1)  # switch to EEG/Hypnogram tab
        self.statusBar().showMessage("Analysis complete.")

    @Slot(str)
    def _on_pipeline_error(self, msg: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Error", msg)

    def _on_generate_music(self) -> None:
        if self.experiment.features is None:
            QMessageBox.warning(self, "Warning", "Run EEG analysis first.")
            return
        self._set_busy(True, "Generating music…")
        self._worker = _Worker(self.experiment.generate_music)
        self._worker.finished.connect(self._on_music_done)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    @Slot(object)
    def _on_music_done(self, result) -> None:
        self._set_busy(False)
        midi_path, wav_path = result
        n_notes = len(self.experiment.notes)
        self.music_panel.update_info(midi_path, wav_path, n_notes)

        # Save experiment record
        exp_id = self.experiment.save_experiment()
        self.experiment_panel.refresh()
        self.tabs.setCurrentIndex(3)  # Music tab
        self.statusBar().showMessage(f"Music generated – Experiment {exp_id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)  # indeterminate
            self.statusBar().showMessage(msg)
        else:
            self.progress.setRange(0, 1)

    @staticmethod
    def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        sos = butter(4, [low, high], btype="band", fs=sfreq, output="sos")
        return sosfiltfilt(sos, data)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #F5F7FA; }
            QTabWidget::pane { border: 1px solid #B0BEC5; background: white; border-radius: 4px; }
            QTabBar::tab {
                background: #E3F2FD; color: #1565C0; padding: 8px 18px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: white; font-weight: bold; }
            QGroupBox {
                background: white; border: 1px solid #E0E0E0; border-radius: 6px;
                margin-top: 12px; padding-top: 18px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QTableWidget { gridline-color: #E0E0E0; }
            QHeaderView::section { background: #E3F2FD; color: #1565C0; padding: 4px; }
            QStatusBar { background: #E3F2FD; color: #1565C0; }
        """)
