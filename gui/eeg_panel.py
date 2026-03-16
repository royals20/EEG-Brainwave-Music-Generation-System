"""
EEG data import panel – load EDF / BDF / CSV and show basic info.
"""

import os
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QFormLayout, QMessageBox, QComboBox,
)

from utils.logger import logger


class EEGPanel(QWidget):
    """Panel for importing EEG data and displaying summary info."""

    eeg_loaded = Signal(str)  # emits file path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Import group ---
        grp = QGroupBox("EEG Data Import")
        grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        grp_layout = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        self.import_btn = QPushButton("Import EEG File")
        self.import_btn.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; border-radius: 4px; padding: 8px 16px; }"
            "QPushButton:hover { background: #1565C0; }"
        )
        self.import_btn.clicked.connect(self._import)
        btn_row.addWidget(self.import_btn)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)

        # Info labels
        info_form = QFormLayout()
        self.lbl_file = QLabel("-")
        self.lbl_sfreq = QLabel("-")
        self.lbl_channels = QLabel("-")
        self.lbl_duration = QLabel("-")
        info_form.addRow("File:", self.lbl_file)
        info_form.addRow("Sample Rate:", self.lbl_sfreq)
        info_form.addRow("Channels:", self.lbl_channels)
        info_form.addRow("Duration:", self.lbl_duration)
        grp_layout.addLayout(info_form)

        # Channel selector
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Analysis Channel:"))
        self.channel_combo = QComboBox()
        ch_row.addWidget(self.channel_combo)
        ch_row.addStretch()
        grp_layout.addLayout(ch_row)

        layout.addWidget(grp)
        layout.addStretch()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG File", "",
            "EEG Files (*.edf *.bdf *.csv);;All Files (*)",
        )
        if path:
            self.eeg_loaded.emit(path)

    def update_info(self, info: dict) -> None:
        """Populate info labels from a dict returned by get_eeg_info()."""
        self.lbl_file.setText(os.path.basename(info.get("file", self.lbl_file.text())))
        self.lbl_sfreq.setText(f"{info.get('sfreq', 0):.1f} Hz")
        self.lbl_channels.setText(str(info.get("n_channels", 0)))
        dur = info.get("duration_s", 0)
        m, s = divmod(int(dur), 60)
        h, m = divmod(m, 60)
        self.lbl_duration.setText(f"{h:02d}:{m:02d}:{s:02d}")

        self.channel_combo.clear()
        self.channel_combo.addItems(info.get("ch_names", []))

    def selected_channel(self) -> Optional[str]:
        return self.channel_combo.currentText() or None
