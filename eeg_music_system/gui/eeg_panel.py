from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional runtime dependency
    pg = None


class EEGPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.subject_id: Optional[int] = None
        self.eeg_path: Optional[str] = None
        self.info_label = QLabel("请先选择受试者并导入EEG文件")
        self.import_btn = QPushButton("导入 EEG (EDF/BDF/CSV)")
        self.plot_widget = pg.PlotWidget() if pg else QLabel("pyqtgraph 未安装，无法显示波形。")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("EEG数据"))
        self.import_btn.clicked.connect(self.select_eeg_file)
        layout.addWidget(self.import_btn)
        layout.addWidget(self.info_label)
        layout.addWidget(self.plot_widget)

    def set_subject(self, subject_id: int) -> None:
        self.subject_id = subject_id
        self.info_label.setText(f"当前受试者: {subject_id}")

    def select_eeg_file(self) -> Optional[str]:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 EEG 文件", "", "EEG Files (*.edf *.bdf *.csv)"
        )
        if not file_path:
            return None
        self.eeg_path = file_path
        self.info_label.setText(f"已选择: {Path(file_path).name}")
        return file_path

    def show_eeg(
        self, signal: np.ndarray, delta_band: np.ndarray, sfreq: float, slow_wave_indices: np.ndarray
    ) -> None:
        if pg is None:
            return
        self.plot_widget.clear()
        time = np.arange(len(signal)) / sfreq
        self.plot_widget.plot(time, signal, pen=pg.mkPen("#4A90E2", width=1), name="Raw EEG")
        self.plot_widget.plot(
            time, delta_band, pen=pg.mkPen("#50E3C2", width=1), name="Delta Band"
        )
        if len(slow_wave_indices):
            self.plot_widget.plot(
                time[slow_wave_indices],
                delta_band[slow_wave_indices],
                pen=None,
                symbol="o",
                symbolBrush="#D0021B",
                symbolSize=5,
            )
