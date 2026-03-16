import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from eeg_music_system.database.db_manager import DBManager
from eeg_music_system.gui.eeg_panel import EEGPanel
from eeg_music_system.gui.experiment_panel import ExperimentPanel
from eeg_music_system.gui.music_panel import MusicPanel
from eeg_music_system.gui.subject_panel import SubjectPanel
from eeg_music_system.services.experiment_manager import ExperimentManager
from eeg_music_system.utils.logger import setup_logger


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.logger = setup_logger(__name__)
        self.db = DBManager()
        self.db.initialize()
        self.manager = ExperimentManager(self.db)
        self.current_subject_id: Optional[int] = None
        self.current_eeg_path: Optional[str] = None
        self.last_result: Optional[dict] = None
        self._build_ui()
        self._bind_events()
        self._apply_style()

    def _build_ui(self) -> None:
        self.setWindowTitle("EEG Brainwave Music Generation System")
        self.resize(1300, 800)
        central = QWidget()
        layout = QHBoxLayout(central)
        splitter = QSplitter()
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.subject_panel = SubjectPanel(self.db)
        splitter.addWidget(self.subject_panel)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("EEG Brainwave Music Generation System"))
        self.eeg_panel = EEGPanel()
        self.music_panel = MusicPanel()
        self.experiment_panel = ExperimentPanel()
        right_layout.addWidget(self.eeg_panel)
        right_layout.addWidget(self.music_panel)
        right_layout.addWidget(self.experiment_panel)
        splitter.addWidget(right)
        splitter.setSizes([300, 1000])

    def _bind_events(self) -> None:
        self.subject_panel.subject_changed.connect(self.on_subject_changed)
        self.eeg_panel.import_btn.clicked.connect(self.on_import_eeg)
        self.music_panel.generate_clicked.connect(self.on_generate_music)
        self.experiment_panel.export_btn.clicked.connect(self.on_export_csv)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #F2F7FF; color: #1F2D3D; font-size: 13px; }
            QPushButton {
                background-color: #3D8BFF; color: white; border-radius: 8px; padding: 8px;
            }
            QPushButton:hover { background-color: #2369CF; }
            QTableWidget, QListWidget, QLabel {
                background-color: #FFFFFF; border: 1px solid #D8E5FF; border-radius: 8px; padding: 6px;
            }
            """
        )

    def on_subject_changed(self, subject_id: int) -> None:
        self.current_subject_id = subject_id
        self.eeg_panel.set_subject(subject_id)
        self._refresh_experiments()

    def on_import_eeg(self) -> None:
        if not self.current_subject_id:
            QMessageBox.warning(self, "提示", "请先选择受试者。")
            return
        file_path = self.eeg_panel.select_eeg_file()
        if not file_path:
            return
        self.current_eeg_path = file_path
        try:
            eeg = self.manager.load_eeg_file(file_path)
            self.eeg_panel.info_label.setText(
                f"采样率: {eeg['sfreq']} Hz | 通道数: {eeg['channel_count']} | 时长: {eeg['duration']:.2f} s"
            )
            self.music_panel.set_status("EEG已导入，可进行SWS分析与音乐生成")
        except Exception as exc:
            self.logger.exception("EEG import failed")
            QMessageBox.critical(self, "错误", str(exc))

    def on_generate_music(self) -> None:
        if not self.current_subject_id or not self.current_eeg_path:
            QMessageBox.warning(self, "提示", "请先选择受试者并导入EEG。")
            return
        try:
            result = self.manager.run_analysis_and_music(
                self.current_subject_id, self.current_eeg_path
            )
            sws = result["sws"]
            eeg = result["eeg"]
            self.eeg_panel.show_eeg(
                eeg["signal"], sws["delta_band"], eeg["sfreq"], sws["slow_wave_indices"]
            )
            self.music_panel.set_status(
                "SWS时长: %.2fs | 平均Delta功率: %.4f | 慢波密度: %.2f"
                % (sws["sws_duration"], sws["avg_delta_power"], sws["slow_wave_density"])
            )
            self.last_result = result
            self._refresh_experiments()
        except Exception as exc:
            self.logger.exception("Generation failed")
            QMessageBox.critical(self, "错误", str(exc))

    def _refresh_experiments(self) -> None:
        if not self.current_subject_id:
            return
        experiments = self.db.list_experiments(self.current_subject_id)
        self.experiment_panel.set_experiments(experiments)

    def on_export_csv(self) -> None:
        if not self.current_subject_id:
            QMessageBox.warning(self, "提示", "请先选择受试者。")
            return
        out_path = self.experiment_panel.ask_export_path()
        if not out_path:
            return
        self.manager.export_experiments_csv(out_path, self.current_subject_id)
        QMessageBox.information(self, "导出成功", f"数据已导出到: {Path(out_path).name}")


def run_app() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
