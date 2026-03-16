from typing import Iterable

from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from eeg_music_system.database.models import Experiment


class ExperimentPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.export_btn = QPushButton("导出 CSV")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["实验时间", "SWS时长", "平均Delta功率", "慢波密度", "WAV路径"]
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("实验记录"))
        layout.addWidget(self.export_btn)
        layout.addWidget(self.table)

    def set_experiments(self, experiments: Iterable[Experiment]) -> None:
        rows = list(experiments)
        self.table.setRowCount(len(rows))
        for i, exp in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(exp.created_time.strftime("%Y-%m-%d %H:%M:%S")))
            self.table.setItem(i, 1, QTableWidgetItem(f"{exp.sws_duration:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{exp.avg_delta_power:.4f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{exp.slow_wave_density:.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(exp.wav_path))

    def ask_export_path(self) -> str:
        file_path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "", "CSV files (*.csv)")
        return file_path
