"""
Music generation panel – trigger MIDI / WAV generation and playback.
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QMessageBox,
)

from utils.logger import logger


class MusicPanel(QWidget):
    """Panel for generating and previewing brainwave music."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        grp = QGroupBox("Brainwave Music Generation")
        grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        grp_layout = QVBoxLayout(grp)

        # Generate button
        self.gen_btn = QPushButton("Generate Music")
        self.gen_btn.setStyleSheet(
            "QPushButton { background: #2E7D32; color: white; border-radius: 4px; padding: 10px 20px; font-size: 14px; }"
            "QPushButton:hover { background: #1B5E20; }"
        )
        grp_layout.addWidget(self.gen_btn)

        # Info
        info_form = QFormLayout()
        self.lbl_midi = QLabel("-")
        self.lbl_wav = QLabel("-")
        self.lbl_notes = QLabel("-")
        info_form.addRow("MIDI File:", self.lbl_midi)
        info_form.addRow("WAV File:", self.lbl_wav)
        info_form.addRow("Notes Generated:", self.lbl_notes)
        grp_layout.addLayout(info_form)

        layout.addWidget(grp)
        layout.addStretch()

    def update_info(self, midi_path: str, wav_path: str, n_notes: int) -> None:
        self.lbl_midi.setText(os.path.basename(midi_path))
        self.lbl_wav.setText(os.path.basename(wav_path))
        self.lbl_notes.setText(str(n_notes))
