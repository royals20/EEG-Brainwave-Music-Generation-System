import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLabel, QFileDialog, QGridLayout,
                              QComboBox, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from eeg_processing.eeg_loader import EEGLoader, MNE_AVAILABLE
from eeg_processing.preprocess import bandpass_filter


class EEGCanvas(FigureCanvas):
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.axes = []
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_eeg(self, time, eeg_data, title='EEG信号'):
        self.fig.clear()
        
        ax = self.fig.add_subplot(111)
        ax.plot(time, eeg_data, 'b-', linewidth=0.5)
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度 (μV)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_eeg_and_delta(self, time, eeg_data, delta_data, fs):
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(211)
        ax1.plot(time, eeg_data, 'b-', linewidth=0.5)
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅度 (μV)')
        ax1.set_title('原始EEG信号')
        ax1.grid(True, alpha=0.3)
        
        ax2 = self.fig.add_subplot(212)
        delta_time = np.linspace(0, len(delta_data) / fs, len(delta_data))
        ax2.plot(delta_time, delta_data, 'r-', linewidth=0.5)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('幅度 (μV)')
        ax2.set_title('Delta频段 (0.5-4 Hz)')
        ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def clear(self):
        self.fig.clear()
        self.draw()


class EEGPanel(QWidget):
    
    eeg_loaded = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loader = EEGLoader()
        self.current_file = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        import_group = QGroupBox('EEG数据导入')
        import_layout = QVBoxLayout(import_group)
        
        button_layout = QHBoxLayout()
        
        self.import_edf_btn = QPushButton('导入EDF/BDF')
        self.import_edf_btn.clicked.connect(self.import_edf)
        if not MNE_AVAILABLE:
            self.import_edf_btn.setEnabled(False)
            self.import_edf_btn.setToolTip('需要安装MNE库: pip install mne')
        button_layout.addWidget(self.import_edf_btn)
        
        self.import_csv_btn = QPushButton('导入CSV')
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
        
        info_layout.addWidget(QLabel('选择通道:'), 1, 0)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        info_layout.addWidget(self.channel_combo, 1, 1, 1, 2)
        
        import_layout.addLayout(info_layout)
        
        layout.addWidget(import_group)
        
        viz_group = QGroupBox('EEG可视化')
        viz_layout = QVBoxLayout(viz_group)
        
        self.canvas = EEGCanvas(self)
        viz_layout.addWidget(self.canvas)
        
        view_layout = QHBoxLayout()
        
        self.view_raw_btn = QPushButton('显示原始信号')
        self.view_raw_btn.clicked.connect(self.view_raw)
        self.view_raw_btn.setEnabled(False)
        view_layout.addWidget(self.view_raw_btn)
        
        self.view_delta_btn = QPushButton('显示Delta频段')
        self.view_delta_btn.clicked.connect(self.view_delta)
        self.view_delta_btn.setEnabled(False)
        view_layout.addWidget(self.view_delta_btn)
        
        viz_layout.addLayout(view_layout)
        
        layout.addWidget(viz_group)
    
    def import_edf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择EDF/BDF文件',
            '',
            'EDF/BDF Files (*.edf *.bdf);;All Files (*)'
        )
        
        if file_path:
            try:
                info = self.loader.load_edf(file_path)
                self.current_file = file_path
                self.update_info(info)
                self.eeg_loaded.emit(info)
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导入失败: {str(e)}')
    
    def import_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择CSV文件',
            '',
            'CSV Files (*.csv);;All Files (*)'
        )
        
        if file_path:
            try:
                info = self.loader.load_csv(file_path)
                self.current_file = file_path
                self.update_info(info)
                self.eeg_loaded.emit(info)
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导入失败: {str(e)}')
    
    def load_simulated(self):
        try:
            info = self.loader.load_simulated()
            self.current_file = None
            self.update_info(info)
            self.eeg_loaded.emit(info)
            QMessageBox.information(self, '成功', '模拟EEG数据已生成')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'生成失败: {str(e)}')
    
    def update_info(self, info):
        self.sample_rate_label.setText(f"{info.get('sample_rate', 0):.1f} Hz")
        self.channel_count_label.setText(str(info.get('channel_count', 0)))
        
        duration = info.get('duration', 0)
        minutes = int(duration // 60)
        seconds = duration % 60
        self.duration_label.setText(f"{minutes}分{seconds:.1f}秒")
        
        self.channel_combo.clear()
        channel_names = info.get('channel_names', [])
        self.channel_combo.addItems(channel_names if channel_names else ['Channel 1'])
        
        self.view_raw_btn.setEnabled(True)
        self.view_delta_btn.setEnabled(True)
    
    def on_channel_changed(self, index):
        if self.loader.data is not None:
            self.view_raw()
    
    def view_raw(self):
        if self.loader.data is None:
            return
        
        channel_idx = self.channel_combo.currentIndex()
        eeg_data = self.loader.get_channel_data(channel_idx)
        time = self.loader.get_time_vector()
        
        max_points = 50000
        if len(eeg_data) > max_points:
            step = len(eeg_data) // max_points
            eeg_data = eeg_data[::step]
            time = time[::step]
        
        self.canvas.plot_eeg(time, eeg_data, '原始EEG信号')
    
    def view_delta(self):
        if self.loader.data is None:
            return
        
        channel_idx = self.channel_combo.currentIndex()
        eeg_data = self.loader.get_channel_data(channel_idx)
        fs = self.loader.sample_rate
        
        delta_data = bandpass_filter(eeg_data, 0.5, 4.0, fs)
        
        time = self.loader.get_time_vector()
        
        max_points = 30000
        if len(eeg_data) > max_points:
            step = len(eeg_data) // max_points
            eeg_data = eeg_data[::step]
            time = time[::step]
        
        delta_step = len(delta_data) // max_points if len(delta_data) > max_points else 1
        delta_data = delta_data[::delta_step]
        
        self.canvas.plot_eeg_and_delta(time, eeg_data, delta_data, fs)
    
    def get_loader(self):
        return self.loader
    
    def set_enabled(self, enabled):
        self.import_edf_btn.setEnabled(enabled)
        self.import_csv_btn.setEnabled(enabled)
        self.simulate_btn.setEnabled(enabled)
