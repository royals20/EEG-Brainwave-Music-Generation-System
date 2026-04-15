import json
import os
import shutil
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from database.db_manager import DatabaseManager
from eeg_processing.feature_extractor import FeatureExtractor
from eeg_processing.sws_detector import SWSDetector
from gui.eeg_panel import EEGPanel
from gui.music_panel import MusicPanel
from gui.subject_panel import SubjectPanel
from gui.sws_panel import SWSPanel
from music.audio_synth import AudioPlayer, AudioSynthesizer
from music.mapper import EEGMusicMapper
from music.midi_generator import MIDIGenerator
from reports.csv_exporter import CSVExporter
from utils.config import ensure_subject_directories, get_subject_dirs


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
        self.current_channel_index = 0
        self.current_channel_name = None
        self.current_session_id = None
        self.sws_results = None
        self.music_features = None
        self.midi_path = None
        self.audio_path = None

        self.init_ui()

    @staticmethod
    def _translate_scale_name(scale_name):
        scale_mapping = {
            'Pentatonic': '五声音阶',
            'Pentatonic (浜斿０闊抽樁)': '五声音阶',
            'Major': '大调',
            'Major (澶ц皟)': '大调',
            'Minor': '小调',
            'Minor (灏忚皟)': '小调',
        }
        return scale_mapping.get(scale_name, scale_name)

    @staticmethod
    def _translate_instrument_name(instrument_name):
        instrument_mapping = {
            'Ambient Pad': '氛围音垫',
            'Soft Piano': '柔和钢琴',
            'Warm Strings': '温暖弦乐',
        }
        return instrument_mapping.get(instrument_name, instrument_name)

    def init_ui(self):
        self.setWindowTitle('EEG 脑波音乐生成系统')
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

        header_label = QLabel('EEG 脑波音乐生成系统')
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
        self.eeg_panel.channel_selected.connect(self.on_eeg_channel_selected)
        tabs.addTab(self.eeg_panel, 'EEG 数据')

        self.sws_panel = SWSPanel()
        self.sws_panel.detect_clicked.connect(self.on_detect_sws)
        tabs.addTab(self.sws_panel, 'SWS 检测')

        self.music_panel = MusicPanel()
        self.music_panel.generate_clicked.connect(self.on_generate_music)
        self.music_panel.play_clicked.connect(self.on_play_music)
        self.music_panel.pause_clicked.connect(self.on_pause_music)
        self.music_panel.stop_clicked.connect(self.on_stop_music)
        self.music_panel.export_clicked.connect(self.on_export_csv)
        self.music_panel.seek_clicked.connect(self.on_seek_music)
        self.music_panel.volume_changed.connect(self.on_volume_changed)
        self.music_panel.export_music_clicked.connect(self.on_export_music)
        tabs.addTab(self.music_panel, '音乐')

        right_layout.addWidget(tabs)

        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([350, 1050])

        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')

        self.update_ui_state()

    def _reset_processing_state(self, clear_eeg: bool = False, preserve_session: bool = False):
        if self.audio_player:
            self.audio_player.stop()

        self.sws_detector = None
        self.feature_extractor = None
        self.music_mapper = None
        self.midi_generator = None
        self.audio_synthesizer = None

        self.sws_results = None
        self.music_features = None
        self.midi_path = None
        self.audio_path = None

        if not preserve_session:
            self.current_session_id = None

        self.sws_panel.reset_results()
        self.music_panel.reset_results()

        if clear_eeg:
            self.eeg_loader = None
            self.current_eeg_data = None
            self.current_fs = None
            self.current_channel_index = 0
            self.current_channel_name = None

    def _set_current_analysis_channel(self, channel_index: int = None):
        if not self.eeg_loader or self.eeg_loader.data is None:
            return

        if channel_index is None:
            channel_index = self.eeg_panel.get_selected_channel_index()

        self.current_channel_index = channel_index
        self.current_channel_name = self.eeg_panel.get_selected_channel_name()
        self.current_eeg_data = self.eeg_loader.get_channel_data(channel_index)
        self.current_fs = self.eeg_loader.sample_rate

    def _save_eeg_session(self, eeg_info: dict):
        if not self.current_subject:
            return

        try:
            self.current_session_id = self.db.add_eeg_session(
                subject_id=self.current_subject.get('subject_id', ''),
                file_path=eeg_info.get('file_path'),
                sample_rate=eeg_info.get('sample_rate', 0),
                channel_count=eeg_info.get('channel_count', 0),
                duration_seconds=eeg_info.get('duration', 0),
            )
        except Exception as e:
            self.current_session_id = None
            QMessageBox.warning(self, '保存失败', f'无法保存 EEG 会话: {str(e)}')

    def _save_sws_result(self):
        if not self.current_session_id or not self.sws_results:
            return

        try:
            self.db.upsert_sws_result(
                session_id=self.current_session_id,
                sws_duration=self.sws_results.get('total_sws_duration', 0),
                avg_delta_power=self.sws_results.get('avg_delta_power', 0),
                sws_segments=json.dumps(self.sws_results.get('sws_epochs', []), ensure_ascii=False),
            )
        except Exception as e:
            QMessageBox.warning(self, '保存失败', f'无法保存 SWS 结果: {str(e)}')

    def _save_music_result(self, music_params: dict, music_duration: float):
        if not self.current_session_id:
            return

        try:
            self.db.upsert_music_result(
                session_id=self.current_session_id,
                midi_path=self.midi_path,
                audio_path=self.audio_path,
                avg_pitch=music_params.get('avg_pitch', 0),
                avg_tempo=music_params.get('avg_tempo', 0),
                music_duration=music_duration,
            )
        except Exception as e:
            QMessageBox.warning(self, '保存失败', f'无法保存音乐结果: {str(e)}')

    def _restore_eeg_session(self, session_data):
        file_path = session_data.get('file_path')
        if not file_path:
            return False, '最新会话没有 EEG 文件'

        if not os.path.exists(file_path):
            return False, f'历史 EEG 文件缺失: {os.path.basename(file_path)}'

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.edf', '.bdf'):
                info = self.eeg_panel.get_loader().load_edf(file_path)
            elif ext == '.csv':
                info = self.eeg_panel.get_loader().load_csv(
                    file_path,
                    sample_rate=session_data.get('sample_rate') or 256,
                )
            else:
                return False, f'不支持的 EEG 文件类型: {ext or "未知"}'
        except Exception as e:
            return False, f'恢复 EEG 历史失败: {str(e)}'

        self.eeg_panel.current_file = file_path
        self.eeg_panel.update_info(info)
        self.eeg_loader = self.eeg_panel.get_loader()
        self._set_current_analysis_channel(0)
        return True, None

    def _restore_sws_history(self, session_data):
        if session_data.get('sws_duration') is None:
            return False

        sws_epochs = []
        sws_segments = session_data.get('sws_segments')
        if sws_segments:
            try:
                sws_epochs = json.loads(sws_segments)
            except (TypeError, json.JSONDecodeError):
                sws_epochs = []

        total_sws_duration = session_data.get('sws_duration') or 0
        total_slow_waves = sum(epoch.get('slow_wave_count', 0) for epoch in sws_epochs)
        slow_wave_density = (
            total_slow_waves / (total_sws_duration / 60.0)
            if total_sws_duration > 0 else 0
        )

        total_epoch_count = 0
        if self.current_eeg_data is not None and self.current_fs:
            epoch_seconds = 30
            total_epoch_count = int(len(self.current_eeg_data) // (epoch_seconds * self.current_fs))

        self.sws_results = {
            'total_sws_duration': total_sws_duration,
            'avg_delta_power': session_data.get('avg_delta_power') or 0,
            'slow_wave_density': slow_wave_density,
            'sws_epoch_count': len(sws_epochs),
            'total_epoch_count': total_epoch_count,
            'epochs': sws_epochs,
            'sws_epochs': sws_epochs,
        }
        self.sws_panel.update_results(self.sws_results)
        return True

    def _restore_music_history(self, session_data):
        if session_data.get('midi_path') is None and session_data.get('audio_path') is None:
            return False, None

        self.midi_path = session_data.get('midi_path')
        self.audio_path = session_data.get('audio_path')

        midi_exists = bool(self.midi_path and os.path.exists(self.midi_path))
        audio_exists = bool(self.audio_path and os.path.exists(self.audio_path))

        self.music_panel.update_results({
            'midi_path': self.midi_path,
            'audio_path': self.audio_path,
            'avg_pitch': session_data.get('avg_pitch'),
            'avg_tempo': session_data.get('avg_tempo'),
            'note_count': None,
            'music_duration': session_data.get('music_duration') or 0,
        })
        self.music_panel.play_btn.setEnabled(audio_exists)
        self.music_panel.export_music_btn.setEnabled(midi_exists or audio_exists)

        if not midi_exists and not audio_exists:
            return False, '历史音乐文件缺失'

        return True, None

    def _has_valid_sws_results(self):
        return bool(self.sws_results) and (self.sws_results.get('total_sws_duration', 0) > 0)

    def on_subject_selected(self, subject_data):
        subject_id = subject_data.get('subject_id', '')

        ensure_subject_directories(subject_id)
        self._reset_processing_state(clear_eeg=True)

        self.current_subject = subject_data
        history_message = self._load_subject_history(subject_id)

        self.status_bar.showMessage(
            history_message or f"已选择患者：{subject_id} - {subject_data.get('name', '')}"
        )
        self.update_ui_state()

    def _load_subject_history(self, subject_id):
        try:
            sessions = self.db.get_subject_sessions(subject_id)
            if not sessions:
                return None

            latest_result = self.db.get_latest_results(subject_id)
            if not latest_result:
                return None

            self.current_session_id = latest_result.get('session_id')

            restored_parts = []
            warnings = []

            eeg_restored, eeg_warning = self._restore_eeg_session(latest_result)
            if eeg_restored:
                restored_parts.append('EEG')
            elif eeg_warning:
                warnings.append(eeg_warning)

            if self._restore_sws_history(latest_result):
                restored_parts.append('SWS')

            music_restored, music_warning = self._restore_music_history(latest_result)
            if music_restored:
                restored_parts.append('music')
            elif music_warning:
                warnings.append(music_warning)

            restored_text = ', '.join(restored_parts) if restored_parts else '仅元数据'
            restored_text = restored_text.replace('EEG', 'EEG 数据').replace('SWS', 'SWS 结果').replace('music', '音乐结果')
            message = (
                f"已选择患者：{subject_id} - {self.current_subject.get('name', '')}；"
                f"已恢复历史：{restored_text}；会话数={len(sessions)}"
            )
            if warnings:
                message += f"；提示：{'; '.join(warnings)}"
            return message
        except Exception as e:
            return f"加载历史失败: {str(e)}"

    def on_eeg_loaded(self, eeg_info):
        self.eeg_loader = self.eeg_panel.get_loader()
        if self.eeg_loader and self.eeg_loader.data is not None:
            self._reset_processing_state(clear_eeg=False)
            self._set_current_analysis_channel()
            self._save_eeg_session(eeg_info)
            self.status_bar.showMessage(
                f"EEG 已加载：通道={self.current_channel_name}，采样率={self.current_fs}Hz，"
                f"时长={self.eeg_loader.duration:.1f}秒"
            )
        self.update_ui_state()

    def on_eeg_channel_selected(self, channel_info):
        if not self.eeg_loader or self.eeg_loader.data is None:
            return

        self._reset_processing_state(clear_eeg=False, preserve_session=True)
        self._set_current_analysis_channel(channel_info.get('channel_index'))
        self.status_bar.showMessage(
            f"分析通道已切换为 {self.current_channel_name}，请重新执行 SWS 检测和音乐生成。"
        )
        self.update_ui_state()

    def on_detect_sws(self):
        if self.current_eeg_data is None:
            QMessageBox.warning(self, '提示', '请先导入 EEG 数据')
            return

        self.status_bar.showMessage(f'正在检测通道 {self.current_channel_name} 的 SWS...')
        QApplication.processEvents()

        thresholds = self.sws_panel.get_thresholds()
        self.sws_detector = SWSDetector({
            'delta_power_threshold': thresholds['delta_power_threshold'],
            'amplitude_threshold_uv': thresholds['amplitude_threshold'],
            'slow_wave_ratio_threshold': thresholds['slow_wave_ratio_threshold'],
        })

        self.sws_results = self.sws_detector.detect(self.current_eeg_data, self.current_fs)
        self.sws_panel.update_results(self.sws_results)
        self._save_sws_result()

        mins = int(self.sws_results['total_sws_duration'] // 60)
        secs = self.sws_results['total_sws_duration'] % 60
        if self._has_valid_sws_results():
            self.status_bar.showMessage(
                f"SWS 检测完成：通道={self.current_channel_name}，时长={mins}分 {secs:.1f}秒，"
                f"Epoch 数={self.sws_results['sws_epoch_count']}，密度={self.sws_results['slow_wave_density']:.2f}/分钟"
            )
        else:
            self.status_bar.showMessage(
                f"SWS 检测完成：通道={self.current_channel_name}，当前未检出典型 SWS；"
                f"仍可按全段 EEG 特征生成音乐，建议再检查通道和阈值。"
            )
        self.update_ui_state()

    def on_generate_music(self):
        if self.current_eeg_data is None:
            QMessageBox.warning(self, '提示', '请先导入 EEG 数据')
            return

        if self.sws_results is None:
            QMessageBox.warning(self, '提示', '请先执行 SWS 检测')
            return

        try:
            self.status_bar.showMessage(f'正在提取通道 {self.current_channel_name} 的 EEG 特征...')
            QApplication.processEvents()

            self.feature_extractor = FeatureExtractor(self.current_fs)
            all_features = self.feature_extractor.extract(self.current_eeg_data)

            if not all_features:
                QMessageBox.warning(self, '提示', '无法提取 EEG 特征')
                return

            stats = self.feature_extractor.get_individualized_stats()
            if self._has_valid_sws_results():
                slow_wave_density = self.sws_results.get('slow_wave_density', 3.0)
            else:
                slow_wave_density = 3.0
                self.status_bar.showMessage(
                    f'当前通道未检出典型 SWS，正在使用全段 EEG 特征和默认慢波密度生成音乐...'
                )
                QApplication.processEvents()
            user_music_params = self.music_panel.get_generation_params()

            self.music_mapper = EEGMusicMapper()
            self.music_mapper.configure_individualized(stats, slow_wave_density, user_music_params)
            individual_params = self.music_mapper.get_individual_params()
            scale_name = self._translate_scale_name(individual_params['scale_name'])
            instrument_name = self._translate_instrument_name(individual_params['instrument_name'])

            self.status_bar.showMessage(
                f"音乐参数：通道={self.current_channel_name}，调式={scale_name}，"
                f"节奏={individual_params['tempo']} BPM，乐器={instrument_name}"
            )
            QApplication.processEvents()

            self.music_features = self.music_mapper.generate_ambient_structure(all_features)
            if not self.music_features:
                QMessageBox.warning(self, '提示', '无法生成音乐特征')
                return

            self.status_bar.showMessage(f'已生成 {len(self.music_features)} 个音符，正在创建 MIDI...')
            QApplication.processEvents()

            subject_id = self.current_subject.get('subject_id', 'default')
            subject_dirs = get_subject_dirs(subject_id)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            self.midi_generator = MIDIGenerator(
                bpm=individual_params['tempo'],
                instrument_program=individual_params.get('instrument_program'),
            )
            self.midi_path = self.midi_generator.generate(
                self.music_features,
                target_duration_minutes=30,
                output_dir=subject_dirs['midi'],
                filename=f"{subject_id}_{timestamp}.mid",
            )

            self.status_bar.showMessage('正在合成 30 分钟音频...')
            QApplication.processEvents()

            self.audio_synthesizer = AudioSynthesizer()
            self.audio_path = self.audio_synthesizer.synthesize(
                self.midi_path,
                output_dir=subject_dirs['audio'],
                filename=f"{subject_id}_{timestamp}.wav",
            )

            audio_info = self.audio_synthesizer.get_audio_info()
            music_duration = audio_info.get('duration', 0) if audio_info else 0

            music_params = self.music_mapper.get_music_parameters()
            display_scale = self._translate_scale_name(music_params.get('scale', '大调'))
            display_instrument = self._translate_instrument_name(music_params.get('instrument', '柔和钢琴'))
            self._save_music_result(music_params, music_duration)

            self.music_panel.update_results({
                'midi_path': self.midi_path,
                'audio_path': self.audio_path,
                'avg_pitch': music_params.get('avg_pitch', 0),
                'avg_tempo': music_params.get('avg_tempo', 0),
                'note_count': music_params.get('note_count', 0),
                'music_duration': music_duration,
                'scale': display_scale,
                'instrument': display_instrument,
            })

            self.music_panel.update_visualization(self.music_features)

            mins = int(music_duration // 60)
            secs = int(music_duration % 60)
            self.status_bar.showMessage(
                f"音乐生成完成：通道={self.current_channel_name}，时长={mins}分 {secs}秒，"
                f"调式={display_scale}，节奏={music_params.get('avg_tempo', 60)} BPM"
            )
            self.update_ui_state()

        except Exception as e:
            error_msg = f'音乐生成失败: {str(e)}\n\n{traceback.format_exc()}'
            QMessageBox.critical(self, '错误', error_msg)
            self.status_bar.showMessage('音乐生成失败')

    def on_play_music(self):
        if self.audio_path is None:
            QMessageBox.warning(self, '提示', '请先生成音乐')
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
            self.status_bar.showMessage(f'已跳转到 {position_seconds:.1f} 秒')

    def on_export_csv(self):
        if self.current_subject is None:
            QMessageBox.warning(self, '提示', '请先选择患者')
            return

        if self.sws_results is None:
            QMessageBox.warning(self, '提示', '请先执行 SWS 检测')
            return

        exporter = CSVExporter()
        music_params = self.music_mapper.get_music_parameters() if self.music_mapper else {}

        exporter.add_result(
            subject_id=self.current_subject.get('subject_id', ''),
            sws_duration=self.sws_results.get('total_sws_duration', 0),
            avg_delta_power=self.sws_results.get('avg_delta_power', 0),
            avg_pitch=music_params.get('avg_pitch', 0),
            avg_tempo=music_params.get('avg_tempo', 0),
            session_id=self.current_session_id or '',
            channel_name=self.current_channel_name or '',
            music_duration=self.music_panel.music_duration,
            scale=music_params.get('scale', ''),
            instrument=music_params.get('instrument', ''),
            instrument_program=music_params.get('instrument_program', ''),
        )

        csv_path = exporter.export()
        QMessageBox.information(self, '导出成功', f'数据已追加到：\n{csv_path}')
        self.status_bar.showMessage(f'数据已追加：{csv_path}')

    def on_volume_changed(self, volume):
        if self.audio_player:
            self.audio_player.set_volume(volume)

    def on_export_music(self):
        if self.audio_path is None or not os.path.exists(self.audio_path):
            QMessageBox.warning(self, '提示', '请先生成音乐')
            return

        if self.midi_path is None or not os.path.exists(self.midi_path):
            QMessageBox.warning(self, '提示', '请先生成音乐')
            return

        save_dir = QFileDialog.getExistingDirectory(self, '选择导出目录')
        if not save_dir:
            return

        midi_dest = os.path.join(save_dir, 'brain_music.mid')
        wav_dest = os.path.join(save_dir, 'brain_music.wav')

        try:
            shutil.copy2(self.midi_path, midi_dest)
            shutil.copy2(self.audio_path, wav_dest)
            QMessageBox.information(
                self,
                '导出成功',
                f'音乐文件已导出\n\nMIDI: {midi_dest}\nWAV: {wav_dest}',
            )
            self.status_bar.showMessage(f'音乐已导出到：{save_dir}')
        except Exception as e:
            QMessageBox.critical(self, '导出失败', f'导出失败: {str(e)}')

    def update_ui_state(self):
        has_subject = self.current_subject is not None
        has_eeg = self.current_eeg_data is not None
        has_detected_sws = self.sws_results is not None
        has_music = bool(self.music_features) or bool(self.audio_path and os.path.exists(self.audio_path))

        self.eeg_panel.set_enabled(has_subject)
        self.sws_panel.set_enabled(has_eeg)
        self.music_panel.set_generate_enabled(has_detected_sws)
        self.music_panel.set_play_enabled(has_music)
        self.music_panel.set_export_enabled(has_detected_sws)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 9))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
