import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from eeg_processing.eeg_loader import EEGLoader, MNE_AVAILABLE
from eeg_processing.preprocess import bandpass_filter
from utils.config import EEG_SIMULATION_CONFIG

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EEGCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_eeg(self, time_axis, eeg_data, title='EEG 信号'):
        self.fig.clear()
        axis = self.fig.add_subplot(111)
        axis.plot(time_axis, eeg_data, 'b-', linewidth=0.5)
        axis.set_xlabel('时间（秒）')
        axis.set_ylabel('幅值（uV）')
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()

    def plot_eeg_and_delta(self, raw_time, eeg_data, delta_time, delta_data):
        self.fig.clear()

        raw_axis = self.fig.add_subplot(211)
        raw_axis.plot(raw_time, eeg_data, 'b-', linewidth=0.5)
        raw_axis.set_xlabel('时间（秒）')
        raw_axis.set_ylabel('幅值（uV）')
        raw_axis.set_title('原始 EEG 信号')
        raw_axis.grid(True, alpha=0.3)

        delta_axis = self.fig.add_subplot(212)
        delta_axis.plot(delta_time, delta_data, 'r-', linewidth=0.5)
        delta_axis.set_xlabel('时间（秒）')
        delta_axis.set_ylabel('幅值（uV）')
        delta_axis.set_title('Delta 频段（0.5-4 Hz）')
        delta_axis.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    def clear(self):
        self.fig.clear()
        self.draw()


class EEGPanel(QWidget):

    eeg_loaded = pyqtSignal(dict)
    channel_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loader = EEGLoader()
        self.current_file = None
        self._syncing_channel_combo = False
        self._delta_view_cache = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        import_group = QGroupBox('EEG 数据导入')
        import_layout = QVBoxLayout(import_group)

        button_layout = QHBoxLayout()

        self.import_edf_btn = QPushButton('导入 EDF/BDF')
        self.import_edf_btn.clicked.connect(self.import_edf)
        if not MNE_AVAILABLE:
            self.import_edf_btn.setEnabled(False)
            self.import_edf_btn.setToolTip('需要安装 MNE：pip install mne')
        button_layout.addWidget(self.import_edf_btn)

        self.import_csv_btn = QPushButton('导入 CSV')
        self.import_csv_btn.clicked.connect(self.import_csv)
        button_layout.addWidget(self.import_csv_btn)

        self.simulate_btn = QPushButton('生成模拟数据')
        self.simulate_btn.clicked.connect(self.load_simulated)
        button_layout.addWidget(self.simulate_btn)

        import_layout.addLayout(button_layout)

        info_layout = QGridLayout()
        info_layout.addWidget(QLabel('采样率:'), 0, 0)
        self.sample_rate_label = QLabel('-')
        self.sample_rate_label.setStyleSheet('font-weight: bold;')
        info_layout.addWidget(self.sample_rate_label, 0, 1)

        info_layout.addWidget(QLabel('通道数:'), 0, 2)
        self.channel_count_label = QLabel('-')
        self.channel_count_label.setStyleSheet('font-weight: bold;')
        info_layout.addWidget(self.channel_count_label, 0, 3)

        info_layout.addWidget(QLabel('记录时长:'), 0, 4)
        self.duration_label = QLabel('-')
        self.duration_label.setStyleSheet('font-weight: bold;')
        info_layout.addWidget(self.duration_label, 0, 5)

        info_layout.addWidget(QLabel('分析通道:'), 1, 0)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        info_layout.addWidget(self.channel_combo, 1, 1, 1, 2)

        import_layout.addLayout(info_layout)
        layout.addWidget(import_group)

        visualization_group = QGroupBox('EEG 可视化')
        visualization_layout = QVBoxLayout(visualization_group)
        self.canvas = EEGCanvas(self)
        visualization_layout.addWidget(self.canvas)

        view_layout = QHBoxLayout()
        self.view_raw_btn = QPushButton('显示原始信号')
        self.view_raw_btn.clicked.connect(self.view_raw)
        self.view_raw_btn.setEnabled(False)
        view_layout.addWidget(self.view_raw_btn)

        self.view_delta_btn = QPushButton('显示 Delta 频段')
        self.view_delta_btn.clicked.connect(self.view_delta)
        self.view_delta_btn.setEnabled(False)
        view_layout.addWidget(self.view_delta_btn)

        visualization_layout.addLayout(view_layout)
        layout.addWidget(visualization_group)

    def _downsample_for_plot(self, values, max_points: int):
        sample_count = len(values)
        if sample_count == 0:
            return np.array([], dtype=int)
        if sample_count <= max_points:
            return np.arange(sample_count, dtype=int)

        step = max(1, int(np.ceil(sample_count / max_points)))
        indices = np.arange(0, sample_count, step, dtype=int)
        if indices[-1] != sample_count - 1 and len(indices) < max_points:
            indices = np.append(indices, sample_count - 1)
        return indices

    def _prepare_signal_for_plot(self, values, sample_rate: float, max_points: int):
        if len(values) <= max_points:
            return np.asarray(values, dtype=float), sample_rate
        source_indices = np.arange(len(values), dtype=float)
        target_indices = np.linspace(0, len(values) - 1, max_points)
        resampled = np.interp(target_indices, source_indices, values)
        duration = (len(values) - 1) / sample_rate if sample_rate > 0 and len(values) > 1 else 0
        preview_fs = sample_rate
        if duration > 0:
            preview_fs = (len(resampled) - 1) / duration
        return resampled, preview_fs

    def _format_sample_rate(self, sample_rate: float) -> str:
        if sample_rate >= 10:
            return f'{sample_rate:.2f} Hz'
        return f'{sample_rate:.4f} Hz'

    def _prompt_csv_sample_rate(self, csv_info: dict):
        inferred_sample_rate = csv_info.get('inferred_sample_rate')
        if inferred_sample_rate is not None:
            message = (
                f"检测到时间列：{csv_info.get('time_column_name')}。\n"
                f"已推断采样率约为 {self._format_sample_rate(inferred_sample_rate)}。\n"
                '可直接确认，也可以手动覆写。'
            )
            default_value = float(inferred_sample_rate)
        else:
            reason = csv_info.get('sample_rate_reason') or '未检测到可用于推断的时间列。'
            default_value = float(EEG_SIMULATION_CONFIG.get('sample_rate', 256))
            message = (
                f'{reason}\n'
                '请根据设备配置手动输入采样率（Hz）。'
            )

        sample_rate, accepted = QInputDialog.getDouble(
            self,
            '确认 CSV 采样率',
            message,
            default_value,
            0.001,
            100000.0,
            6,
        )
        if not accepted:
            return None
        return sample_rate

    def import_edf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择 EDF/BDF 文件',
            '',
            'EDF/BDF 文件 (*.edf *.bdf);;所有文件 (*)',
        )

        if not file_path:
            return

        try:
            info = self.loader.load_edf(file_path)
            self.current_file = file_path
            self.update_info(info)
            self.eeg_loaded.emit(info)
        except Exception as exc:
            QMessageBox.critical(self, '导入失败', str(exc))

    def import_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择 CSV 文件',
            '',
            'CSV 文件 (*.csv);;所有文件 (*)',
        )

        if not file_path:
            return

        try:
            csv_info = self.loader.inspect_csv(file_path)
            sample_rate = self._prompt_csv_sample_rate(csv_info)
            if sample_rate is None:
                return

            info = self.loader.load_csv(file_path, sample_rate=sample_rate)
            self.current_file = file_path
            self.update_info(info)
            self.eeg_loaded.emit(info)
        except Exception as exc:
            QMessageBox.critical(self, '导入失败', str(exc))

    def load_simulated(self):
        try:
            info = self.loader.load_simulated()
            self.current_file = None
            self.update_info(info)
            self.eeg_loaded.emit(info)
            QMessageBox.information(self, '成功', '模拟 EEG 数据已生成。')
        except Exception as exc:
            QMessageBox.critical(self, '生成失败', str(exc))

    def update_info(self, info):
        self._delta_view_cache.clear()
        self.sample_rate_label.setText(f"{info.get('sample_rate', 0):.3f} Hz")
        self.channel_count_label.setText(str(info.get('channel_count', 0)))

        duration = float(info.get('duration', 0) or 0)
        minutes = int(duration // 60)
        seconds = duration % 60
        self.duration_label.setText(f'{minutes}分{seconds:.1f}秒')

        channel_names = info.get('channel_names', []) or ['通道 1']
        self._syncing_channel_combo = True
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(channel_names)
        self.channel_combo.setCurrentIndex(0)
        self.channel_combo.blockSignals(False)
        self._syncing_channel_combo = False

        self.view_raw_btn.setEnabled(True)
        self.view_delta_btn.setEnabled(True)
        self.view_raw()

    def reset(self):
        self.loader = EEGLoader()
        self.current_file = None
        self._delta_view_cache.clear()
        self.sample_rate_label.setText('-')
        self.channel_count_label.setText('-')
        self.duration_label.setText('-')

        self._syncing_channel_combo = True
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.blockSignals(False)
        self._syncing_channel_combo = False

        self.channel_combo.setEnabled(False)
        self.view_raw_btn.setEnabled(False)
        self.view_delta_btn.setEnabled(False)
        self.canvas.clear()

    def on_channel_changed(self, index):
        if self.loader.data is None or index < 0:
            return

        self.view_raw()
        if not self._syncing_channel_combo:
            self.channel_selected.emit(
                {
                    'channel_index': index,
                    'channel_name': self.get_selected_channel_name(),
                }
            )

    def view_raw(self):
        eeg_data = self.get_selected_channel_data()
        if eeg_data is None:
            return

        sample_rate = self.loader.sample_rate
        display_indices = self._downsample_for_plot(eeg_data, max_points=50000)
        display_time = display_indices / sample_rate
        display_data = np.asarray(eeg_data, dtype=float)[display_indices]
        self.canvas.plot_eeg(display_time, display_data, f'原始 EEG - {self.get_selected_channel_name()}')

    def view_delta(self):
        eeg_data = self.get_selected_channel_data()
        if eeg_data is None:
            return

        fs = self.loader.sample_rate
        channel_index = self.get_selected_channel_index()
        cached_preview = self._delta_view_cache.get(channel_index)
        if cached_preview is None:
            preview_signal, preview_fs = self._prepare_signal_for_plot(eeg_data, fs, max_points=30000)
            if len(preview_signal) == 0 or preview_fs <= 0:
                self.canvas.clear()
                return
            raw_time = np.arange(len(preview_signal), dtype=float) / preview_fs
            delta_data = bandpass_filter(preview_signal, 0.5, 4.0, preview_fs)
            delta_time = np.arange(len(delta_data), dtype=float) / preview_fs
            cached_preview = (raw_time, preview_signal, delta_time, delta_data)
            self._delta_view_cache[channel_index] = cached_preview
        self.canvas.plot_eeg_and_delta(*cached_preview)

    def get_loader(self):
        return self.loader

    def get_selected_channel_index(self):
        return max(0, self.channel_combo.currentIndex())

    def get_selected_channel_name(self):
        return self.channel_combo.currentText() or f'通道 {self.get_selected_channel_index() + 1}'

    def get_selected_channel_data(self):
        if self.loader.data is None:
            return None
        return self.loader.get_channel_data(self.get_selected_channel_index())

    def set_enabled(self, enabled):
        self.import_edf_btn.setEnabled(enabled and MNE_AVAILABLE)
        self.import_csv_btn.setEnabled(enabled)
        self.simulate_btn.setEnabled(enabled)
        self.channel_combo.setEnabled(enabled and self.loader.data is not None)
        self.view_raw_btn.setEnabled(enabled and self.loader.data is not None)
        self.view_delta_btn.setEnabled(enabled and self.loader.data is not None)
