import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from utils.config import MUSIC_MAPPING_CONFIG

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MusicCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 3), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_music_features(self, features):
        self.fig.clear()

        if not features:
            self.draw()
            return

        ax1 = self.fig.add_subplot(131)
        pitches = [f['pitch'] for f in features]
        ax1.hist(pitches, bins=20, color='steelblue', edgecolor='white')
        ax1.set_xlabel('音高（MIDI）')
        ax1.set_ylabel('数量')
        ax1.set_title('音高分布')

        ax2 = self.fig.add_subplot(132)
        velocities = [f['velocity'] for f in features]
        ax2.hist(velocities, bins=20, color='coral', edgecolor='white')
        ax2.set_xlabel('力度')
        ax2.set_ylabel('数量')
        ax2.set_title('力度分布')

        ax3 = self.fig.add_subplot(133)
        times = [f['start_time'] for f in features]
        ax3.scatter(times, pitches, c=velocities, cmap='viridis', alpha=0.6, s=20)
        ax3.set_xlabel('时间（秒）')
        ax3.set_ylabel('音高（MIDI）')
        ax3.set_title('音符时序图')

        self.fig.tight_layout()
        self.draw()

    def clear(self):
        self.fig.clear()
        self.draw()


class MusicPanel(QWidget):

    generate_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    seek_clicked = pyqtSignal(float)
    volume_changed = pyqtSignal(float)
    export_music_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.music_results = None
        self.music_duration = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_progress)
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0
        self.slider_dragging = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        config_group = QGroupBox('音乐生成参数')
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel('基础音高:'), 0, 0)
        self.base_pitch_spin = QSpinBox()
        self.base_pitch_spin.setRange(20, 80)
        self.base_pitch_spin.setValue(MUSIC_MAPPING_CONFIG.get('base_pitch', 60))
        config_layout.addWidget(self.base_pitch_spin, 0, 1)

        config_layout.addWidget(QLabel('音高范围:'), 0, 2)
        self.pitch_range_spin = QSpinBox()
        self.pitch_range_spin.setRange(10, 60)
        self.pitch_range_spin.setValue(MUSIC_MAPPING_CONFIG.get('pitch_range', 12))
        config_layout.addWidget(self.pitch_range_spin, 0, 3)

        config_layout.addWidget(QLabel('基础节奏 (BPM):'), 1, 0)
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(40, 120)
        self.tempo_spin.setValue(MUSIC_MAPPING_CONFIG.get('base_tempo', 60))
        config_layout.addWidget(self.tempo_spin, 1, 1)

        layout.addWidget(config_group)

        generate_group = QGroupBox('脑波音乐生成')
        generate_layout = QVBoxLayout(generate_group)

        self.generate_btn = QPushButton('生成音乐')
        self.generate_btn.clicked.connect(self.on_generate)
        generate_layout.addWidget(self.generate_btn)

        results_layout = QGridLayout()

        results_layout.addWidget(QLabel('平均音高:'), 0, 0)
        self.avg_pitch_label = QLabel('-')
        self.avg_pitch_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.avg_pitch_label, 0, 1)

        results_layout.addWidget(QLabel('平均节奏:'), 0, 2)
        self.avg_tempo_label = QLabel('-')
        self.avg_tempo_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.avg_tempo_label, 0, 3)

        results_layout.addWidget(QLabel('音符数量:'), 1, 0)
        self.note_count_label = QLabel('-')
        self.note_count_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.note_count_label, 1, 1)

        results_layout.addWidget(QLabel('MIDI 文件:'), 1, 2)
        self.midi_path_label = QLabel('-')
        self.midi_path_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.midi_path_label, 1, 3)

        generate_layout.addLayout(results_layout)
        layout.addWidget(generate_group)

        player_group = QGroupBox('音乐播放')
        player_layout = QVBoxLayout(player_group)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel('音乐时长:'))
        self.duration_label = QLabel('0:00')
        self.duration_label.setStyleSheet('font-weight: bold; font-size: 14px;')
        duration_layout.addWidget(self.duration_label)
        duration_layout.addStretch()
        player_layout.addLayout(duration_layout)

        progress_layout = QHBoxLayout()
        self.position_label = QLabel('0:00')
        progress_layout.addWidget(self.position_label)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        progress_layout.addWidget(self.progress_slider)
        self.total_label = QLabel('0:00')
        progress_layout.addWidget(self.total_label)
        player_layout.addLayout(progress_layout)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel('音量:'))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        self.volume_label = QLabel('80%')
        volume_layout.addWidget(self.volume_label)
        player_layout.addLayout(volume_layout)

        button_layout = QHBoxLayout()

        self.play_btn = QPushButton('播放')
        self.play_btn.clicked.connect(self.on_play)
        self.play_btn.setEnabled(False)
        button_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton('暂停')
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton('停止')
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        player_layout.addLayout(button_layout)
        layout.addWidget(player_group)

        viz_group = QGroupBox('音乐特征可视化')
        viz_layout = QVBoxLayout(viz_group)
        self.canvas = MusicCanvas(self)
        viz_layout.addWidget(self.canvas)
        layout.addWidget(viz_group)

        export_group = QGroupBox('数据导出')
        export_layout = QHBoxLayout(export_group)

        self.export_csv_btn = QPushButton('导出 CSV 数据')
        self.export_csv_btn.clicked.connect(self.on_export)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)

        self.export_music_btn = QPushButton('导出音乐文件')
        self.export_music_btn.clicked.connect(self.on_export_music)
        self.export_music_btn.setEnabled(False)
        export_layout.addWidget(self.export_music_btn)

        layout.addWidget(export_group)

    def on_generate(self):
        self.generate_clicked.emit()

    def on_play(self):
        self.play_clicked.emit()
        self.is_playing = True
        self.is_paused = False
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_slider.setEnabled(True)
        self.playback_timer.start(100)

    def on_pause(self):
        self.pause_clicked.emit()
        if self.is_paused:
            self.is_paused = False
            self.playback_timer.start(100)
        else:
            self.is_paused = True
            self.playback_timer.stop()

    def on_stop(self):
        self.stop_clicked.emit()
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_slider.setValue(0)
        self.position_label.setText('0:00')

    def update_progress(self):
        if self.music_duration > 0 and not self.slider_dragging:
            self.playback_position += 0.1
            if self.playback_position >= self.music_duration:
                self.on_stop()
                return

            progress = int((self.playback_position / self.music_duration) * 1000)
            self.progress_slider.setValue(progress)

            mins = int(self.playback_position // 60)
            secs = int(self.playback_position % 60)
            self.position_label.setText(f'{mins}:{secs:02d}')

    def format_duration(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{mins}:{secs:02d}'

    def on_slider_pressed(self):
        self.slider_dragging = True

    def on_slider_released(self):
        if self.music_duration > 0:
            position = (self.progress_slider.value() / 1000.0) * self.music_duration
            self.playback_position = position
            self.seek_clicked.emit(position)

            mins = int(position // 60)
            secs = int(position % 60)
            self.position_label.setText(f'{mins}:{secs:02d}')

        self.slider_dragging = False

    def on_export(self):
        self.export_clicked.emit()

    def on_export_music(self):
        self.export_music_clicked.emit()

    def on_volume_changed(self, value):
        self.volume_label.setText(f'{value}%')
        self.volume_changed.emit(value / 100.0)

    def get_volume(self):
        return self.volume_slider.value() / 100.0

    def get_generation_params(self):
        return {
            'base_pitch': self.base_pitch_spin.value(),
            'pitch_range': self.pitch_range_spin.value(),
            'base_tempo': self.tempo_spin.value(),
        }

    def reset_results(self):
        self.music_results = None
        self.music_duration = 0
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0
        self.slider_dragging = False
        self.playback_timer.stop()

        self.avg_pitch_label.setText('-')
        self.avg_tempo_label.setText('-')
        self.note_count_label.setText('-')
        self.midi_path_label.setText('-')
        self.duration_label.setText('0:00')
        self.total_label.setText('0:00')
        self.position_label.setText('0:00')
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)

        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_music_btn.setEnabled(False)

        self.canvas.clear()

    def update_results(self, results):
        self.music_results = results
        self.music_duration = results.get('music_duration', 0)
        self.playback_position = 0

        avg_pitch = results.get('avg_pitch')
        avg_tempo = results.get('avg_tempo')
        note_count = results.get('note_count')

        self.avg_pitch_label.setText(f"{avg_pitch:.1f}" if avg_pitch is not None else '-')
        self.avg_tempo_label.setText(f"{avg_tempo:.0f} BPM" if avg_tempo is not None else '-')
        self.note_count_label.setText(str(note_count) if note_count is not None else '-')

        midi_path = results.get('midi_path', '')
        self.midi_path_label.setText(os.path.basename(midi_path) if midi_path else '-')

        self.duration_label.setText(self.format_duration(self.music_duration or 0))
        self.total_label.setText(self.format_duration(self.music_duration or 0))
        self.position_label.setText('0:00')
        self.progress_slider.setValue(0)

        self.play_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_music_btn.setEnabled(True)

    def set_generate_enabled(self, enabled):
        self.generate_btn.setEnabled(enabled)

    def set_play_enabled(self, enabled):
        self.play_btn.setEnabled(enabled)
        if enabled:
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

    def set_export_enabled(self, enabled):
        self.export_csv_btn.setEnabled(enabled)
        self.export_music_btn.setEnabled(enabled)

    def update_visualization(self, features):
        self.canvas.plot_music_features(features)
