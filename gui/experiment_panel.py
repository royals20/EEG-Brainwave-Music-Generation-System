"""
Experiment records panel – view past experiments and export CSV.
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox,
)

from typing import Optional

from database.db_manager import DatabaseManager
from utils.logger import logger


class ExperimentPanel(QWidget):
    """Panel listing experiment records with CSV export."""

    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self._subject_id: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        grp = QGroupBox("Experiment Records")
        grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        grp_layout = QVBoxLayout(grp)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Experiment ID", "Subject", "Time",
            "SWS Dur (min)", "Delta Power", "SW Density", "SW Amplitude",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        grp_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        for b in [self.refresh_btn, self.export_btn]:
            b.setStyleSheet(
                "QPushButton { background: #1976D2; color: white; border-radius: 4px; padding: 6px 14px; }"
                "QPushButton:hover { background: #1565C0; }"
            )
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp)

    def set_subject(self, subject_id: Optional[str]) -> None:
        self._subject_id = subject_id
        self.refresh()

    def refresh(self) -> None:
        exps = self.db.list_experiments(self._subject_id)
        self.table.setRowCount(len(exps))
        for row, e in enumerate(exps):
            self.table.setItem(row, 0, QTableWidgetItem(e.experiment_id))
            self.table.setItem(row, 1, QTableWidgetItem(e.subject_id))
            t = e.created_time.strftime("%Y-%m-%d %H:%M") if e.created_time else ""
            self.table.setItem(row, 2, QTableWidgetItem(t))
            self.table.setItem(row, 3, QTableWidgetItem(f"{e.sws_duration or 0:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{e.delta_power or 0:.4f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{e.slow_wave_density or 0:.2f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{e.slow_wave_amplitude or 0:.2f}"))

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "experiments.csv", "CSV Files (*.csv)",
        )
        if path:
            try:
                self.db.export_experiments_csv(path, self._subject_id)
                QMessageBox.information(self, "Success", f"Exported to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
