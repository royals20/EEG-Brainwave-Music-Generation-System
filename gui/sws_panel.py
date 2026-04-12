from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLabel, QGridLayout, QDoubleSpinBox,
                              QProgressBar)
from PyQt5.QtCore import pyqtSignal, Qt

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SWSCanvas(FigureCanvas):
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_sws_detection(self, epochs, total_duration):
        self.fig.clear()
        
        ax = self.fig.add_subplot(111)
        
        times = []
        delta_powers = []
        colors = []
        
        for epoch in epochs:
            times.append(epoch['start_time'])
            delta_powers.append(epoch.get('delta_power', 0))
            if epoch.get('is_sws', False):
                colors.append('steelblue')
            else:
                colors.append('lightgray')
        
        ax.bar(times, delta_powers, width=28, color=colors, alpha=0.7, edgecolor='white')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('Delta Power')
        ax.set_title('SWS检测结果 (蓝色=SWS, 灰色=非SWS)')
        ax.grid(True, alpha=0.3, axis='y')
        
        self.fig.tight_layout()
        self.draw()
    
    def clear(self):
        self.fig.clear()
        self.draw()


class SWSPanel(QWidget):
    
    detect_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sws_results = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        config_group = QGroupBox('SWS检测参数 (AASM标准)')
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel('Delta Power阈值:'), 0, 0)
        self.delta_threshold_spin = QDoubleSpinBox()
        self.delta_threshold_spin.setRange(1e-8, 1e-3)
        self.delta_threshold_spin.setValue(1e-6)
        self.delta_threshold_spin.setDecimals(8)
        self.delta_threshold_spin.setSingleStep(1e-7)
        config_layout.addWidget(self.delta_threshold_spin, 0, 1)
        
        config_layout.addWidget(QLabel('慢波振幅阈值 (μV):'), 0, 2)
        self.amplitude_threshold_spin = QDoubleSpinBox()
        self.amplitude_threshold_spin.setRange(10, 200)
        self.amplitude_threshold_spin.setValue(75)
        config_layout.addWidget(self.amplitude_threshold_spin, 0, 3)
        
        config_layout.addWidget(QLabel('慢波占比阈值 (%):'), 1, 0)
        self.ratio_threshold_spin = QDoubleSpinBox()
        self.ratio_threshold_spin.setRange(5, 50)
        self.ratio_threshold_spin.setValue(20)
        config_layout.addWidget(self.ratio_threshold_spin, 1, 1)
        
        layout.addWidget(config_group)
        
        detect_group = QGroupBox('SWS检测')
        detect_layout = QVBoxLayout(detect_group)
        
        self.detect_btn = QPushButton('检测SWS')
        self.detect_btn.clicked.connect(self.on_detect)
        detect_layout.addWidget(self.detect_btn)
        
        results_layout = QGridLayout()
        
        results_layout.addWidget(QLabel('SWS总时长:'), 0, 0)
        self.sws_duration_label = QLabel('-')
        self.sws_duration_label.setStyleSheet('font-weight: bold; font-size: 14px; color: #2196F3;')
        results_layout.addWidget(self.sws_duration_label, 0, 1)
        
        results_layout.addWidget(QLabel('平均Delta Power:'), 0, 2)
        self.avg_delta_label = QLabel('-')
        self.avg_delta_label.setStyleSheet('font-weight: bold; font-size: 14px;')
        results_layout.addWidget(self.avg_delta_label, 0, 3)
        
        results_layout.addWidget(QLabel('SWS Epoch数:'), 1, 0)
        self.epoch_count_label = QLabel('-')
        self.epoch_count_label.setStyleSheet('font-weight: bold; font-size: 14px;')
        results_layout.addWidget(self.epoch_count_label, 1, 1)
        
        results_layout.addWidget(QLabel('慢波密度 (个/分钟):'), 1, 2)
        self.sw_density_label = QLabel('-')
        self.sw_density_label.setStyleSheet('font-weight: bold; font-size: 14px; color: #4CAF50;')
        results_layout.addWidget(self.sw_density_label, 1, 3)
        
        detect_layout.addLayout(results_layout)
        
        layout.addWidget(detect_group)
        
        viz_group = QGroupBox('SWS可视化')
        viz_layout = QVBoxLayout(viz_group)
        
        self.canvas = SWSCanvas(self)
        viz_layout.addWidget(self.canvas)
        
        layout.addWidget(viz_group)
    
    def on_detect(self):
        self.detect_clicked.emit()
    
    def update_results(self, results):
        self.sws_results = results
        
        duration = results.get('total_sws_duration', 0)
        minutes = int(duration // 60)
        seconds = duration % 60
        self.sws_duration_label.setText(f"{minutes}分{seconds:.1f}秒")
        
        delta_power = results.get('avg_delta_power', 0)
        if delta_power > 0:
            self.avg_delta_label.setText(f"{delta_power:.2e}")
        else:
            self.avg_delta_label.setText("-")
        
        self.epoch_count_label.setText(str(results.get('sws_epoch_count', 0)))
        
        sw_density = results.get('slow_wave_density', 0)
        self.sw_density_label.setText(f"{sw_density:.2f}")
        
        epochs = results.get('epochs', [])
        if epochs:
            total_duration = epochs[-1]['end_time'] if epochs else 0
            self.canvas.plot_sws_detection(epochs, total_duration)
    
    def get_thresholds(self):
        return {
            'delta_power_threshold': self.delta_threshold_spin.value(),
            'amplitude_threshold': self.amplitude_threshold_spin.value(),
            'slow_wave_ratio_threshold': self.ratio_threshold_spin.value() / 100.0
        }
    
    def set_enabled(self, enabled):
        self.detect_btn.setEnabled(enabled)
