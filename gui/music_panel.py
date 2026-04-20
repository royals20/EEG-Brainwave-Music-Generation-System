import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        axis_pitch = self.fig.add_subplot(131)
        pitches = [feature['pitch'] for feature in features]
        axis_pitch.hist(pitches, bins=20, color='steelblue', edgecolor='white')
        axis_pitch.set_xlabel('音高')
        axis_pitch.set_ylabel('数量')
        axis_pitch.set_title('音高分布')

        axis_velocity = self.fig.add_subplot(132)
        velocities = [feature['velocity'] for feature in features]
        axis_velocity.hist(velocities, bins=20, color='coral', edgecolor='white')
        axis_velocity.set_xlabel('力度')
        axis_velocity.set_ylabel('数量')
        axis_velocity.set_title('力度分布')

        axis_sequence = self.fig.add_subplot(133)
        times = [feature['start_time'] for feature in features]
        axis_sequence.scatter(times, pitches, c=velocities, cmap='viridis', alpha=0.6, s=20)
        axis_sequence.set_xlabel('时间（秒）')
        axis_sequence.set_ylabel('音高')
        axis_sequence.set_title('音符时序')

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
    progress_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.music_results = None
        self.music_duration = 0.0
        self.is_playing = False
        self.is_paused = False
        self.slider_dragging = False
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.request_progress_update)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        config_group = QGroupBox('音乐生成设置')
        config_layout = QGridLayout(config_group)
        config_layout.setHorizontalSpacing(14)
        config_layout.setVerticalSpacing(10)

        config_layout.addWidget(QLabel('基础音高：'), 0, 0)
        self.base_pitch_spin = QSpinBox()
        self.base_pitch_spin.setRange(20, 80)
        self.base_pitch_spin.setValue(MUSIC_MAPPING_CONFIG.get('base_pitch', 60))
        self.base_pitch_spin.setSuffix(' MIDI')
        config_layout.addWidget(self.base_pitch_spin, 0, 1)

        config_layout.addWidget(QLabel('音域范围：'), 0, 2)
        self.pitch_range_spin = QSpinBox()
        self.pitch_range_spin.setRange(10, 60)
        self.pitch_range_spin.setValue(MUSIC_MAPPING_CONFIG.get('pitch_range', 14))
        self.pitch_range_spin.setSuffix(' 半音')
        config_layout.addWidget(self.pitch_range_spin, 0, 3)

        config_layout.addWidget(QLabel('基础速度：'), 1, 0)
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(40, 120)
        self.tempo_spin.setValue(MUSIC_MAPPING_CONFIG.get('base_tempo', 60))
        self.tempo_spin.setSuffix(' BPM')
        config_layout.addWidget(self.tempo_spin, 1, 1)

        config_layout.addWidget(QLabel('旋律密度：'), 1, 2)
        self.melody_density_spin = QSpinBox()
        self.melody_density_spin.setRange(1, 5)
        self.melody_density_spin.setValue(int(MUSIC_MAPPING_CONFIG.get('melody_density', 2)))
        config_layout.addWidget(self.melody_density_spin, 1, 3)

        config_layout.addWidget(QLabel('和声丰富度：'), 2, 0)
        self.harmony_richness_spin = QSpinBox()
        self.harmony_richness_spin.setRange(1, 5)
        self.harmony_richness_spin.setValue(int(MUSIC_MAPPING_CONFIG.get('harmony_richness', 4)))
        config_layout.addWidget(self.harmony_richness_spin, 2, 1)

        config_layout.addWidget(QLabel('目标时长：'), 2, 2)
        self.target_duration_spin = QDoubleSpinBox()
        self.target_duration_spin.setRange(0.0, 240.0)
        self.target_duration_spin.setDecimals(1)
        self.target_duration_spin.setSingleStep(5.0)
        self.target_duration_spin.setValue(
            float(MUSIC_MAPPING_CONFIG.get('target_music_duration_minutes', 0.0) or 0.0)
        )
        self.target_duration_spin.setSuffix(' 分钟')
        config_layout.addWidget(self.target_duration_spin, 2, 3)

        layout.addWidget(config_group)

        generate_group = QGroupBox('生成结果')
        generate_layout = QVBoxLayout(generate_group)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.generate_btn = QPushButton('生成音乐')
        self.generate_btn.setMinimumWidth(220)
        self.generate_btn.clicked.connect(self.generate_clicked.emit)
        action_row.addWidget(self.generate_btn)
        action_row.addStretch()
        generate_layout.addLayout(action_row)

        results_layout = QGridLayout()
        results_layout.setHorizontalSpacing(20)
        results_layout.setVerticalSpacing(8)
        results_layout.addWidget(QLabel('平均音高：'), 0, 0)
        self.avg_pitch_label = QLabel('-')
        self.avg_pitch_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.avg_pitch_label, 0, 1)

        results_layout.addWidget(QLabel('平均速度：'), 0, 2)
        self.avg_tempo_label = QLabel('-')
        self.avg_tempo_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.avg_tempo_label, 0, 3)

        results_layout.addWidget(QLabel('音符数量：'), 1, 0)
        self.note_count_label = QLabel('-')
        self.note_count_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.note_count_label, 1, 1)

        results_layout.addWidget(QLabel('MIDI 文件：'), 1, 2)
        self.midi_path_label = QLabel('-')
        self.midi_path_label.setStyleSheet('font-weight: bold;')
        results_layout.addWidget(self.midi_path_label, 1, 3)

        generate_layout.addLayout(results_layout)
        layout.addWidget(generate_group)

        player_group = QGroupBox('音乐播放')
        player_layout = QVBoxLayout(player_group)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel('音乐时长：'))
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
        volume_layout.addWidget(QLabel('音量：'))
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
        self.play_btn.clicked.connect(self.play_clicked.emit)
        self.play_btn.setEnabled(False)
        button_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton('暂停')
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton('停止')
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        player_layout.addLayout(button_layout)
        layout.addWidget(player_group)

        visualization_group = QGroupBox('音乐特征')
        visualization_layout = QVBoxLayout(visualization_group)
        self.canvas = MusicCanvas(self)
        visualization_layout.addWidget(self.canvas)
        layout.addWidget(visualization_group)

        export_group = QGroupBox('导出')
        export_layout = QHBoxLayout(export_group)

        self.export_csv_btn = QPushButton('导出 CSV')
        self.export_csv_btn.clicked.connect(self.export_clicked.emit)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)

        self.export_music_btn = QPushButton('导出音乐文件')
        self.export_music_btn.clicked.connect(self.export_music_clicked.emit)
        self.export_music_btn.setEnabled(False)
        export_layout.addWidget(self.export_music_btn)

        layout.addWidget(export_group)

    def request_progress_update(self):
        if self.is_playing and not self.slider_dragging:
            self.progress_requested.emit()

    def format_duration(self, seconds):
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f'{minutes}:{remaining_seconds:02d}'

    def on_slider_pressed(self):
        self.slider_dragging = True

    def on_slider_released(self):
        if self.music_duration > 0:
            position = (self.progress_slider.value() / 1000.0) * self.music_duration
            self.position_label.setText(self.format_duration(position))
            self.seek_clicked.emit(position)
        self.slider_dragging = False

    def on_volume_changed(self, value):
        self.volume_label.setText(f'{value}%')
        self.volume_changed.emit(value / 100.0)

    def get_volume(self):
        return self.volume_slider.value() / 100.0

    def get_generation_params(self):
        target_duration_minutes = float(self.target_duration_spin.value())
        keep_original_duration = target_duration_minutes <= 0
        if keep_original_duration:
            target_duration_minutes = None

        return {
            'base_pitch': self.base_pitch_spin.value(),
            'pitch_range': self.pitch_range_spin.value(),
            'base_tempo': self.tempo_spin.value(),
            'target_duration_minutes': target_duration_minutes,
            'keep_original_duration': keep_original_duration,
            'melody_density': self.melody_density_spin.value(),
            'harmony_richness': self.harmony_richness_spin.value(),
            'phrase_length_bars': int(MUSIC_MAPPING_CONFIG.get('phrase_length_bars', 2)),
            'section_length_bars': int(MUSIC_MAPPING_CONFIG.get('section_length_bars', 8)),
            'humanization_ms': int(MUSIC_MAPPING_CONFIG.get('humanization_ms', 20)),
            'enable_pad': bool(MUSIC_MAPPING_CONFIG.get('enable_pad', True)),
            'enable_bass': bool(MUSIC_MAPPING_CONFIG.get('enable_bass', True)),
            'enable_motif': bool(MUSIC_MAPPING_CONFIG.get('enable_motif', True)),
            'render_backend': MUSIC_MAPPING_CONFIG.get('render_backend', 'auto'),
            'soundfont_path': str(MUSIC_MAPPING_CONFIG.get('soundfont_path', '') or ''),
            'music_window_seconds': float(MUSIC_MAPPING_CONFIG.get('music_window_seconds', 2.0)),
            'music_hop_seconds': float(MUSIC_MAPPING_CONFIG.get('music_hop_seconds', 1.0)),
            'random_seed': int(MUSIC_MAPPING_CONFIG.get('random_seed', 17)),
        }

    def get_target_duration_minutes(self):
        return self.get_generation_params().get('target_duration_minutes')

    def _has_audio_output(self):
        if not self.music_results:
            return False
        if 'audio_exists' in self.music_results:
            return bool(self.music_results.get('audio_exists'))
        return bool(self.music_results.get('audio_path'))

    def _has_midi_output(self):
        if not self.music_results:
            return False
        if 'midi_exists' in self.music_results:
            return bool(self.music_results.get('midi_exists'))
        return bool(self.music_results.get('midi_path'))

    def can_export_music_bundle(self):
        return self._has_midi_output() and self._has_audio_output()

    def on_playback_started(self):
        self.is_playing = True
        self.is_paused = False
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText('暂停')
        self.stop_btn.setEnabled(True)
        self.progress_slider.setEnabled(self.music_duration > 0)
        self.playback_timer.start(200)

    def on_playback_paused(self, paused: bool):
        self.is_playing = not paused
        self.is_paused = paused
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText('继续' if paused else '暂停')
        if paused:
            self.playback_timer.stop()
        else:
            self.playback_timer.start(200)

    def on_playback_stopped(self, reset_position: bool = True):
        self.is_playing = False
        self.is_paused = False
        self.playback_timer.stop()
        self.play_btn.setEnabled(bool(self.music_results and self.music_results.get('audio_path')))
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText('暂停')
        self.stop_btn.setEnabled(False)
        if reset_position:
            self.position_label.setText('0:00')
            self.progress_slider.setValue(0)

    def sync_playback_position(self, position: float):
        if self.music_duration <= 0 or self.slider_dragging:
            return
        clamped_position = max(0.0, min(position, self.music_duration))
        progress = int((clamped_position / self.music_duration) * 1000)
        self.progress_slider.setValue(progress)
        self.position_label.setText(self.format_duration(clamped_position))

    def reset_results(self):
        self.music_results = None
        self.music_duration = 0.0
        self.slider_dragging = False
        self.on_playback_stopped(reset_position=True)

        self.avg_pitch_label.setText('-')
        self.avg_tempo_label.setText('-')
        self.note_count_label.setText('-')
        self.midi_path_label.setText('-')
        self.duration_label.setText('0:00')
        self.total_label.setText('0:00')
        self.progress_slider.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_music_btn.setEnabled(False)
        self.canvas.clear()

    def update_results(self, results):
        self.music_results = results
        self.music_duration = float(results.get('music_duration', 0) or 0)
        self.on_playback_stopped(reset_position=True)

        avg_pitch = results.get('avg_pitch')
        avg_tempo = results.get('avg_tempo')
        note_count = results.get('note_count')

        self.avg_pitch_label.setText(f'{avg_pitch:.1f}' if avg_pitch is not None else '-')
        self.avg_tempo_label.setText(f'{avg_tempo:.0f} BPM' if avg_tempo is not None else '-')
        self.note_count_label.setText(str(note_count) if note_count is not None else '-')

        midi_path = results.get('midi_path', '')
        self.midi_path_label.setText(os.path.basename(midi_path) if midi_path else '-')

        formatted_duration = self.format_duration(self.music_duration)
        self.duration_label.setText(formatted_duration)
        self.total_label.setText(formatted_duration)
        self.position_label.setText('0:00')

        self.play_btn.setEnabled(self._has_audio_output())
        self.export_csv_btn.setEnabled(True)
        self.export_music_btn.setEnabled(self.can_export_music_bundle())
        self.progress_slider.setEnabled(self._has_audio_output() and self.music_duration > 0)

    def set_generate_enabled(self, enabled):
        self.generate_btn.setEnabled(enabled)

    def set_play_enabled(self, enabled):
        if not self.is_playing and not self.is_paused:
            self.play_btn.setEnabled(enabled)
            if not enabled:
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)

    def set_export_enabled(self, enabled, can_export_music_bundle=None):
        self.export_csv_btn.setEnabled(enabled)
        if can_export_music_bundle is None:
            can_export_music_bundle = self.can_export_music_bundle()
        self.export_music_btn.setEnabled(enabled and can_export_music_bundle)

    def update_visualization(self, features):
        self.canvas.plot_music_features(features)
