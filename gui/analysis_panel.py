"""
SWS analysis panel – shows slow-wave features.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from algorithms.feature_extractor import SWSFeatures
from algorithms.sws_detector import SlowWaveEvent


class AnalysisPanel(QWidget):
    """Displays extracted SWS features and detected slow-wave events."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Feature summary ---
        feat_grp = QGroupBox("SWS Features")
        feat_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        feat_form = QFormLayout(feat_grp)

        self.lbl_sws_dur = QLabel("-")
        self.lbl_delta = QLabel("-")
        self.lbl_amp = QLabel("-")
        self.lbl_freq = QLabel("-")
        self.lbl_density = QLabel("-")

        feat_form.addRow("SWS Duration (min):", self.lbl_sws_dur)
        feat_form.addRow("Delta Power (µV²/Hz):", self.lbl_delta)
        feat_form.addRow("SW Amplitude (µV):", self.lbl_amp)
        feat_form.addRow("SW Frequency (Hz):", self.lbl_freq)
        feat_form.addRow("SW Density (events/min):", self.lbl_density)
        layout.addWidget(feat_grp)

        # --- Event table ---
        ev_grp = QGroupBox("Slow-Wave Events")
        ev_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        ev_layout = QVBoxLayout(ev_grp)

        self.event_table = QTableWidget()
        self.event_table.setColumnCount(4)
        self.event_table.setHorizontalHeaderLabels(["Peak Time (s)", "Amplitude (µV)", "Duration (s)", "Frequency (Hz)"])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ev_layout.addWidget(self.event_table)
        layout.addWidget(ev_grp)

    def update_features(self, features: SWSFeatures) -> None:
        self.lbl_sws_dur.setText(f"{features.sws_duration:.2f}")
        self.lbl_delta.setText(f"{features.delta_power:.4f}")
        self.lbl_amp.setText(f"{features.slow_wave_amplitude:.2f}")
        self.lbl_freq.setText(f"{features.slow_wave_frequency:.2f}")
        self.lbl_density.setText(f"{features.slow_wave_density:.2f}")

    def update_events(self, events: list[SlowWaveEvent]) -> None:
        self.event_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            self.event_table.setItem(i, 0, QTableWidgetItem(f"{ev.peak_time:.3f}"))
            self.event_table.setItem(i, 1, QTableWidgetItem(f"{ev.amplitude:.2f}"))
            self.event_table.setItem(i, 2, QTableWidgetItem(f"{ev.duration:.3f}"))
            self.event_table.setItem(i, 3, QTableWidgetItem(f"{ev.frequency:.2f}"))
