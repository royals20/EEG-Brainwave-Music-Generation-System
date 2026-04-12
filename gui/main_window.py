import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QTabWidget, QLabel, QMessageBox, QStatusBar,
                              QSplitter, QFrame, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from gui.subject_panel import SubjectPanel
from gui.eeg_panel import EEGPanel
from gui.sws_panel import SWSPanel
from gui.music_panel import MusicPanel

from database.db_manager import DatabaseManager
from eeg_processing.eeg_loader import EEGLoader
from eeg_processing.sws_detector import SWSDetector
from eeg_processing.feature_extractor import FeatureExtractor
from music.mapper import EEGMusicMapper
from music.midi_generator import MIDIGenerator
from music.audio_synth import AudioSynthesizer, AudioPlayer
from reports.csv_exporter import CSVExporter
from utils.config import ensure_subject_directories, get_subject_dirs, MIDI_OUTPUT_DIR, AUDIO_OUTPUT_DIR


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.db = DatabaseManager()
        self.eeg_loader = None
        self.sws_detector = None
        self.feature_extractor = None
        self.music_mapper = None
        self.midi_generator = None
        self.audio_synthesizer = None
        self.audio_player = None
        
        self.current_subject = None
        self.current_eeg_data = None
        self.current_fs = None
        self.sws_results = None
        self.music_features = None
        self.midi_path = None
        self.audio_path = None
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('EEG脑波音乐生成系统')
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5f8f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #cccccc;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        header_label = QLabel('EEG脑波音乐生成系统')
        header_label.setFont(QFont('Microsoft YaHei', 18, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet('color: #333333; padding: 10px;')
        main_layout.addWidget(header_label)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_frame = QFrame()
        left_frame.setFrameStyle(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        
        self.subject_panel = SubjectPanel(self.db)
        self.subject_panel.subject_selected.connect(self.on_subject_selected)
        left_layout.addWidget(self.subject_panel)
        
        right_frame = QFrame()
        right_frame.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        
        tabs = QTabWidget()
        tabs.setFont(QFont('Microsoft YaHei', 10))
        
        self.eeg_panel = EEGPanel()
        self.eeg_panel.eeg_loaded.connect(self.on_eeg_loaded)
        tabs.addTab(self.eeg_panel, 'EEG数据')
        
        self.sws_panel = SWSPanel()
        self.sws_panel.detect_clicked.connect(self.on_detect_sws)
        tabs.addTab(self.sws_panel, 'SWS检测')
        
        self.music_panel = MusicPanel()
        self.music_panel.generate_clicked.connect(self.on_generate_music)
        self.music_panel.play_clicked.connect(self.on_play_music)
        self.music_panel.pause_clicked.connect(self.on_pause_music)
        self.music_panel.stop_clicked.connect(self.on_stop_music)
        self.music_panel.export_clicked.connect(self.on_export_csv)
        self.music_panel.seek_clicked.connect(self.on_seek_music)
        self.music_panel.volume_changed.connect(self.on_volume_changed)
        self.music_panel.export_music_clicked.connect(self.on_export_music)
        tabs.addTab(self.music_panel, '音乐生成')
        
        right_layout.addWidget(tabs)
        
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')
        
        self.update_ui_state()
    
    def on_subject_selected(self, subject_data):
        subject_id = subject_data.get('subject_id', '')
        
        ensure_subject_directories(subject_id)
        
        self.current_subject = subject_data
        
        self._load_subject_history(subject_id)
        
        self.status_bar.showMessage(f"已选择受试者: {subject_id} - {subject_data.get('name', '')}")
        self.update_ui_state()
    
    def _load_subject_history(self, subject_id):
        try:
            sessions = self.db.get_subject_sessions(subject_id)
            if sessions:
                latest_result = self.db.get_latest_results(subject_id)
                if latest_result:
                    self.status_bar.showMessage(
                        f"加载历史数据: 共{len(sessions)}次会话, "
                        f"最新会话: {latest_result.get('import_time', '')}"
                    )
        except Exception as e:
            self.status_bar.showMessage(f"加载历史数据时出错: {str(e)}")
    
    def on_eeg_loaded(self, eeg_info):
        self.eeg_loader = self.eeg_panel.get_loader()
        if self.eeg_loader and self.eeg_loader.data is not None:
            self.current_eeg_data = self.eeg_loader.get_channel_data(0)
            self.current_fs = self.eeg_loader.sample_rate
            self.status_bar.showMessage(f"EEG数据已加载: 采样率={self.current_fs}Hz, 时长={self.eeg_loader.duration:.1f}秒")
        self.update_ui_state()
    
    def on_detect_sws(self):
        if self.current_eeg_data is None:
            QMessageBox.warning(self, '警告', '请先加载EEG数据')
            return
        
        self.status_bar.showMessage('正在检测SWS (使用Hilbert变换和AASM标准)...')
        QApplication.processEvents()
        
        thresholds = self.sws_panel.get_thresholds()
        self.sws_detector = SWSDetector({
            'delta_power_threshold': thresholds['delta_power_threshold'],
            'amplitude_threshold_uv': thresholds['amplitude_threshold'],
            'slow_wave_ratio_threshold': thresholds['slow_wave_ratio_threshold']
        })
        
        self.sws_results = self.sws_detector.detect(self.current_eeg_data, self.current_fs)
        
        self.sws_panel.update_results(self.sws_results)
        
        mins = int(self.sws_results['total_sws_duration'] // 60)
        secs = self.sws_results['total_sws_duration'] % 60
        self.status_bar.showMessage(
            f"SWS检测完成: 时长={mins}分{secs:.1f}秒, "
            f"Epoch数={self.sws_results['sws_epoch_count']}, "
            f"慢波密度={self.sws_results['slow_wave_density']:.2f}/分钟"
        )
        self.update_ui_state()
    
    def on_generate_music(self):
        if self.current_eeg_data is None:
            QMessageBox.warning(self, '警告', '请先加载EEG数据')
            return
        
        if self.sws_results is None:
            QMessageBox.warning(self, '警告', '请先进行SWS检测')
            return
        
        try:
            self.status_bar.showMessage('正在提取EEG特征...')
            QApplication.processEvents()
            
            self.feature_extractor = FeatureExtractor(self.current_fs)
            all_features = self.feature_extractor.extract(self.current_eeg_data)
            
            if not all_features:
                QMessageBox.warning(self, '警告', '无法提取EEG特征')
                return
            
            stats = self.feature_extractor.get_individualized_stats()
            slow_wave_density = self.sws_results.get('slow_wave_density', 3.0)
            
            self.status_bar.showMessage(
                f'个体化参数: Delta={stats["delta_power_level"]}, '
                f'振幅={stats["amplitude_level"]}, 频率={stats["avg_slow_wave_frequency"]:.2f}Hz'
            )
            QApplication.processEvents()
            
            self.music_mapper = EEGMusicMapper()
            self.music_mapper.configure_individualized(stats, slow_wave_density)
            
            individual_params = self.music_mapper.get_individual_params()
            
            self.status_bar.showMessage(
                f'音乐配置: 音阶={individual_params["scale_name"]}, '
                f'节奏={individual_params["tempo"]}BPM, 音色={individual_params["instrument_name"]}'
            )
            QApplication.processEvents()
            
            self.music_features = self.music_mapper.generate_ambient_structure(all_features)
            
            if not self.music_features:
                QMessageBox.warning(self, '警告', '无法生成音乐特征')
                return
            
            self.status_bar.showMessage(f'生成了 {len(self.music_features)} 个音符，正在创建MIDI...')
            QApplication.processEvents()
            
            subject_id = self.current_subject.get('subject_id', 'default')
            subject_dirs = get_subject_dirs(subject_id)
            
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            self.midi_generator = MIDIGenerator(
                bpm=individual_params['tempo'],
                instrument_program=individual_params.get('instrument_program', 0)
            )
            self.midi_path = self.midi_generator.generate(
                self.music_features, 
                target_duration_minutes=30,
                output_dir=subject_dirs['midi'],
                filename=f"{subject_id}_{timestamp}.mid"
            )
            
            self.status_bar.showMessage('正在合成30分钟音频...')
            QApplication.processEvents()
            
            self.audio_synthesizer = AudioSynthesizer()
            self.audio_path = self.audio_synthesizer.synthesize(
                self.midi_path,
                output_dir=subject_dirs['audio'],
                filename=f"{subject_id}_{timestamp}.wav"
            )
            
            audio_info = self.audio_synthesizer.get_audio_info()
            music_duration = audio_info.get('duration', 0) if audio_info else 0
            
            music_params = self.music_mapper.get_music_parameters()
            
            self.music_panel.update_results({
                'midi_path': self.midi_path,
                'audio_path': self.audio_path,
                'avg_pitch': music_params.get('avg_pitch', 0),
                'avg_tempo': music_params.get('avg_tempo', 0),
                'note_count': music_params.get('note_count', 0),
                'music_duration': music_duration,
                'scale': music_params.get('scale', 'Major'),
                'instrument': music_params.get('instrument', 'Soft Piano')
            })
            
            self.music_panel.update_visualization(self.music_features)
            
            mins = int(music_duration // 60)
            secs = int(music_duration % 60)
            self.status_bar.showMessage(
                f'个体化音乐生成完成！时长: {mins}分{secs}秒, '
                f'音阶: {music_params.get("scale", "Major")}, '
                f'节奏: {music_params.get("avg_tempo", 60)}BPM'
            )
            self.update_ui_state()
            
        except Exception as e:
            import traceback
            error_msg = f'音乐生成失败: {str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.critical(self, '错误', error_msg)
            self.status_bar.showMessage('音乐生成失败')
    
    def on_play_music(self):
        if self.audio_path is None:
            QMessageBox.warning(self, '警告', '请先生成音乐')
            return
        
        if self.audio_player is None:
            self.audio_player = AudioPlayer()
        
        self.audio_player.play(self.audio_path)
        self.status_bar.showMessage('正在播放音乐...')
    
    def on_pause_music(self):
        if self.audio_player:
            self.audio_player.pause()
            self.status_bar.showMessage('音乐已暂停')
    
    def on_stop_music(self):
        if self.audio_player:
            self.audio_player.stop()
            self.status_bar.showMessage('音乐已停止')
    
    def on_seek_music(self, position_seconds):
        if self.audio_player and self.audio_path:
            self.audio_player.seek(position_seconds)
            self.status_bar.showMessage(f'跳转到 {position_seconds:.1f} 秒')
    
    def on_export_csv(self):
        if self.current_subject is None:
            QMessageBox.warning(self, '警告', '请先选择受试者')
            return
        
        if self.sws_results is None:
            QMessageBox.warning(self, '警告', '请先进行SWS检测')
            return
        
        exporter = CSVExporter()
        
        music_params = {}
        if self.music_mapper:
            music_params = self.music_mapper.get_music_parameters()
        
        exporter.add_result(
            subject_id=self.current_subject.get('subject_id', ''),
            sws_duration=self.sws_results.get('total_sws_duration', 0),
            avg_delta_power=self.sws_results.get('avg_delta_power', 0),
            avg_pitch=music_params.get('avg_pitch', 0),
            avg_tempo=music_params.get('avg_tempo', 0)
        )
        
        csv_path = exporter.export()
        
        QMessageBox.information(self, '导出成功', f'数据已导出到:\n{csv_path}')
        self.status_bar.showMessage(f'数据已导出: {csv_path}')
    
    def on_volume_changed(self, volume):
        if self.audio_player:
            self.audio_player.set_volume(volume)
    
    def on_export_music(self):
        if self.audio_path is None or not os.path.exists(self.audio_path):
            QMessageBox.warning(self, '警告', '请先生成音乐')
            return
        
        if self.midi_path is None or not os.path.exists(self.midi_path):
            QMessageBox.warning(self, '警告', '请先生成音乐')
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, '选择导出目录')
        
        if not save_dir:
            return
        
        import shutil
        
        midi_dest = os.path.join(save_dir, 'brain_music.mid')
        wav_dest = os.path.join(save_dir, 'brain_music.wav')
        
        try:
            shutil.copy2(self.midi_path, midi_dest)
            shutil.copy2(self.audio_path, wav_dest)
            
            QMessageBox.information(
                self, '导出成功', 
                f'音乐文件已导出:\n\nMIDI: {midi_dest}\nWAV: {wav_dest}'
            )
            self.status_bar.showMessage(f'音乐已导出到: {save_dir}')
        except Exception as e:
            QMessageBox.critical(self, '导出失败', f'导出失败: {str(e)}')
    
    def update_ui_state(self):
        has_subject = self.current_subject is not None
        has_eeg = self.current_eeg_data is not None
        has_sws = self.sws_results is not None
        has_music = self.music_features is not None
        
        self.eeg_panel.set_enabled(has_subject)
        self.sws_panel.set_enabled(has_eeg)
        self.music_panel.set_generate_enabled(has_sws)
        self.music_panel.set_play_enabled(has_music)
        self.music_panel.set_export_enabled(has_sws)


from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 9))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
