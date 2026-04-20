import json
import os
import shutil
import sys

from PyQt5.QtCore import Qt, QThread
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from gui.eeg_panel import EEGPanel
from gui.history_manager import build_restored_music_results, build_restored_sws_results
from gui.music_panel import MusicPanel
from gui.subject_panel import SubjectPanel
from gui.sws_panel import SWSPanel
from gui.workers import MusicGenerationWorker, SWSDetectionWorker
from music.audio_synth import AudioPlayer
from reports.csv_exporter import CSVExporter
from utils.config import ensure_subject_directories, get_subject_dirs, validate_subject_id
from utils.ui_text import (
    APP_TITLE,
    STATUS_READY,
    display_instrument_name,
    display_scale_name,
    format_detecting_sws,
    format_duration,
    format_history_loaded,
    format_subject_selected,
)


def _make_json_serializable(value):
    if isinstance(value, dict):
        return {key: _make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(item) for item in value]
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes, bytearray)):
        return _make_json_serializable(value.tolist())
    return value


class MainWindow(QMainWindow):

    def __init__(self, db: DatabaseManager = None):
        super().__init__()

        self.db = db or DatabaseManager()
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

        self._active_task = None
        self._sws_thread = None
        self._music_thread = None
        self._sws_worker = None
        self._music_worker = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)

        self.setStyleSheet(
            '''
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
            '''
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        header_label = QLabel(APP_TITLE)
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
        self.subject_panel.subject_context_cleared.connect(self.on_subject_context_cleared)
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
        self.music_panel.progress_requested.connect(self.on_music_progress_requested)
        tabs.addTab(self.music_panel, '音乐生成')

        right_layout.addWidget(tabs)

        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(STATUS_READY)

        self.update_ui_state()

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def _has_valid_sws_results(self) -> bool:
        return bool(self.sws_results) and self.sws_results.get('total_sws_duration', 0) > 0

    def _reset_processing_state(self, clear_eeg: bool = False, preserve_session: bool = False):
        self._shutdown_audio_player()

        self.sws_results = None
        self.music_features = None
        self.midi_path = None
        self.audio_path = None

        if not preserve_session:
            self.current_session_id = None

        self.sws_panel.reset_results()
        self.music_panel.reset_results()

        if clear_eeg:
            self.current_eeg_data = None
            self.current_fs = None
            self.current_channel_index = 0
            self.current_channel_name = None
            self.eeg_panel.reset()

    def _clear_current_subject_context(self, status_message: str):
        self.current_subject = None
        self._reset_processing_state(clear_eeg=True)
        self.status_bar.showMessage(status_message)
        self.update_ui_state()

    def _set_current_analysis_channel(self, channel_index: int = None):
        loader = self.eeg_panel.get_loader()
        if not loader or loader.data is None:
            return

        if channel_index is None:
            channel_index = self.eeg_panel.get_selected_channel_index()

        self.current_channel_index = channel_index
        self.current_channel_name = self.eeg_panel.get_selected_channel_name()
        self.current_eeg_data = loader.get_channel_data(channel_index)
        self.current_fs = loader.sample_rate

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
        except Exception as exc:
            self.current_session_id = None
            self._show_warning('保存失败', f'无法保存 EEG 会话：{exc}')

    def _save_sws_result(self):
        if not self.current_session_id or not self.sws_results:
            return

        try:
            self.db.upsert_sws_result(
                session_id=self.current_session_id,
                sws_duration=self.sws_results.get('total_sws_duration', 0),
                avg_delta_power=self.sws_results.get('avg_delta_power', 0),
                sws_segments=json.dumps(
                    _make_json_serializable(self.sws_results.get('sws_epochs', [])),
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:
            self._show_warning('保存失败', f'无法保存 SWS 结果：{exc}')

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
        except Exception as exc:
            self._show_warning('保存失败', f'无法保存音乐结果：{exc}')

    def _restore_eeg_session(self, session_data):
        file_path = session_data.get('file_path')
        if not file_path:
            return False, '最新会话没有可恢复的 EEG 文件。'
        if not os.path.exists(file_path):
            return False, f'历史 EEG 文件缺失：{os.path.basename(file_path)}'

        try:
            extension = os.path.splitext(file_path)[1].lower()
            loader = self.eeg_panel.get_loader()
            if extension in ('.edf', '.bdf'):
                info = loader.load_edf(file_path)
            elif extension == '.csv':
                info = loader.load_csv(file_path, sample_rate=session_data.get('sample_rate'))
            else:
                return False, f'不支持的 EEG 文件类型：{extension or "未知"}'
        except Exception as exc:
            return False, f'恢复 EEG 历史失败：{exc}'

        self.eeg_panel.current_file = file_path
        self.eeg_panel.update_info(info)
        self._set_current_analysis_channel(0)
        return True, None

    def _restore_sws_history(self, session_data):
        restored_results = build_restored_sws_results(session_data, self.current_eeg_data, self.current_fs)
        if not restored_results:
            return False

        self.sws_results = restored_results
        self.sws_panel.update_results(self.sws_results)
        return True

    def _restore_music_history(self, session_data):
        restored_results, warning = build_restored_music_results(session_data)
        if not restored_results:
            return False, warning

        self.midi_path = restored_results.get('midi_path')
        self.audio_path = restored_results.get('audio_path')
        self.music_panel.update_results(restored_results)
        has_audio = bool(restored_results.get('audio_exists', False))
        has_midi = bool(restored_results.get('midi_exists', False))
        self.music_panel.set_play_enabled(has_audio)
        self.music_panel.set_export_enabled(
            has_midi or has_audio,
            can_export_music_bundle=has_midi and has_audio,
        )
        return has_midi or has_audio, warning

    def _load_subject_history(self, subject_id: str):
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
                restored_parts.append('EEG 数据')
            elif eeg_warning:
                warnings.append(eeg_warning)

            if self._restore_sws_history(latest_result):
                restored_parts.append('SWS 结果')

            music_restored, music_warning = self._restore_music_history(latest_result)
            if music_restored:
                restored_parts.append('音乐结果')
            elif music_warning:
                warnings.append(music_warning)

            restored_text = '、'.join(restored_parts) if restored_parts else '仅元数据'
            message = format_history_loaded(
                subject_id,
                self.current_subject.get('name', ''),
                restored_text,
                len(sessions),
            )
            if warnings:
                message += f"；提示：{'；'.join(warnings)}"
            return message
        except Exception as exc:
            return f'加载历史失败：{exc}'

    def _start_worker(self, task_name: str, worker, worker_attr: str, thread_attr: str, result_handler):
        thread = QThread(self)
        worker.moveToThread(thread)

        setattr(self, worker_attr, worker)
        setattr(self, thread_attr, thread)
        self._active_task = task_name

        thread.started.connect(worker.run)
        worker.progress.connect(self.status_bar.showMessage)
        worker.result.connect(result_handler)
        worker.error.connect(self._show_error)
        worker.cancelled.connect(lambda message, name=task_name: self._handle_worker_cancelled(name, message))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._on_worker_finished(task_name, worker_attr, thread_attr))

        self.update_ui_state()
        thread.start()

    def _handle_worker_cancelled(self, task_name: str, message: str):
        if self._active_task == task_name:
            self.status_bar.showMessage(message)

    def _on_worker_finished(self, task_name: str, worker_attr: str, thread_attr: str):
        setattr(self, worker_attr, None)
        setattr(self, thread_attr, None)
        if self._active_task == task_name:
            self._active_task = None
        self.update_ui_state()

    def _cancel_background_tasks(self):
        for worker in (self._sws_worker, self._music_worker):
            if worker:
                worker.request_cancel()

    def _wait_for_background_tasks(self, timeout_ms: int = 5000) -> bool:
        for thread in (self._sws_thread, self._music_thread):
            if thread and thread.isRunning() and not thread.wait(timeout_ms):
                return False
        return True

    def _shutdown_audio_player(self):
        if self.audio_player:
            self.audio_player.close()
            self.audio_player = None

    def _handle_sws_result(self, results: dict):
        self.sws_results = results
        self.sws_panel.update_results(results)
        self._save_sws_result()

        if self._has_valid_sws_results():
            duration_text = format_duration(results['total_sws_duration'])
            self.status_bar.showMessage(
                f"SWS 检测完成：通道={self.current_channel_name}，时长={duration_text}，"
                f"Epoch 数={results['sws_epoch_count']}，慢波密度={results['slow_wave_density']:.2f}/分钟"
            )
        else:
            self.status_bar.showMessage(
                f'SWS 检测完成：通道={self.current_channel_name}，当前未检出典型 SWS。'
                '仍可使用全段 EEG 特征和默认慢波密度生成音乐。'
            )

    def _handle_music_result(self, payload: dict):
        self.music_features = payload['music_features']
        self.midi_path = payload['midi_path']
        self.audio_path = payload['audio_path']

        music_params = payload['music_params']
        music_duration = payload['music_duration']
        display_scale = display_scale_name(music_params.get('scale', 'Major'))
        display_instrument = display_instrument_name(music_params.get('instrument', 'Soft Piano'))

        self._save_music_result(music_params, music_duration)
        self.music_panel.update_results(
            {
                'midi_path': self.midi_path,
                'audio_path': self.audio_path,
                'avg_pitch': music_params.get('avg_pitch', 0),
                'avg_tempo': music_params.get('avg_tempo', 0),
                'note_count': music_params.get('note_count', 0),
                'music_duration': music_duration,
                'scale': display_scale,
                'instrument': display_instrument,
                'instrument_program': music_params.get('instrument_program'),
            }
        )
        self.music_panel.update_visualization(self.music_features)

        duration_text = format_duration(music_duration)
        source_label = 'SWS 片段' if payload.get('source_mode') == 'sws_only' else '全段 EEG'
        self.status_bar.showMessage(
            f"音乐生成完成：通道={self.current_channel_name}，来源={source_label}，时长={duration_text}，"
            f"调式={display_scale}，乐器={display_instrument}，节奏={music_params.get('avg_tempo', 60):.0f} BPM"
        )

    def on_subject_selected(self, subject_data):
        try:
            subject_id = validate_subject_id(subject_data.get('subject_id', ''))
            ensure_subject_directories(subject_id)
        except ValueError as exc:
            self._clear_current_subject_context('当前受试者不可用，请重新选择。')
            self._show_warning('受试者 ID 无效', f'{exc}\n请修正或删除该受试者后重试。')
            return

        self._reset_processing_state(clear_eeg=True)
        self.current_subject = {**subject_data, 'subject_id': subject_id}
        history_message = self._load_subject_history(subject_id)
        self.status_bar.showMessage(
            history_message or format_subject_selected(subject_id, subject_data.get('name', ''))
        )
        self.update_ui_state()

    def on_subject_context_cleared(self):
        self._clear_current_subject_context('当前受试者已删除，请重新选择受试者。')

    def on_eeg_loaded(self, eeg_info):
        loader = self.eeg_panel.get_loader()
        if loader and loader.data is not None:
            self._reset_processing_state(clear_eeg=False)
            self._set_current_analysis_channel()
            self._save_eeg_session(eeg_info)
            self.status_bar.showMessage(
                f'EEG 已加载：通道={self.current_channel_name}，采样率={self.current_fs:.3f} Hz，'
                f'时长={format_duration(loader.duration)}'
            )
        self.update_ui_state()

    def on_eeg_channel_selected(self, channel_info):
        if self.eeg_panel.get_loader().data is None:
            return

        self._reset_processing_state(clear_eeg=False, preserve_session=True)
        self._set_current_analysis_channel(channel_info.get('channel_index'))
        self.status_bar.showMessage(
            f'分析通道已切换为 {self.current_channel_name}，请重新执行 SWS 检测和音乐生成。'
        )
        self.update_ui_state()

    def on_detect_sws(self):
        if self._active_task:
            return
        if self.current_eeg_data is None:
            self._show_warning('提示', '请先导入 EEG 数据。')
            return

        thresholds = self.sws_panel.get_thresholds()
        worker = SWSDetectionWorker(
            eeg_data=self.current_eeg_data.copy(),
            fs=self.current_fs,
            config={
                'delta_power_threshold': thresholds['delta_power_threshold'],
                'amplitude_threshold_uv': thresholds['amplitude_threshold'],
                'slow_wave_ratio_threshold': thresholds['slow_wave_ratio_threshold'],
            },
            channel_name=self.current_channel_name or '通道 1',
        )
        self.status_bar.showMessage(format_detecting_sws(self.current_channel_name or '通道 1'))
        self._start_worker('sws', worker, '_sws_worker', '_sws_thread', self._handle_sws_result)

    def on_generate_music(self):
        if self._active_task:
            return
        if self.current_eeg_data is None:
            self._show_warning('提示', '请先导入 EEG 数据。')
            return
        if self.sws_results is None:
            self._show_warning('提示', '请先执行 SWS 检测。')
            return

        if not self._has_valid_sws_results():
            self.status_bar.showMessage(
                '当前通道未检出典型 SWS，将回退到全段 EEG 特征生成音乐。'
            )

        try:
            subject_id = validate_subject_id(self.current_subject.get('subject_id', ''))
        except ValueError as exc:
            self._clear_current_subject_context('当前受试者不可用，请重新选择。')
            self._show_warning('受试者 ID 无效', f'{exc}\n请修正或删除该受试者后重试。')
            return

        subject_dirs = get_subject_dirs(subject_id)
        worker = MusicGenerationWorker(
            eeg_data=self.current_eeg_data.copy(),
            fs=self.current_fs,
            sws_results=dict(self.sws_results),
            user_music_params=self.music_panel.get_generation_params(),
            subject_id=subject_id,
            subject_dirs=subject_dirs,
            channel_name=self.current_channel_name or '通道 1',
        )
        self._start_worker('music', worker, '_music_worker', '_music_thread', self._handle_music_result)

    def on_play_music(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            self._show_warning('提示', '请先生成音乐。')
            return

        try:
            if self.audio_player is None:
                self.audio_player = AudioPlayer()
            self.audio_player.play(self.audio_path)
            self.audio_player.set_volume(self.music_panel.get_volume())
            self.music_panel.on_playback_started()
            self.music_panel.sync_playback_position(0.0)
            self.status_bar.showMessage('正在播放音乐...')
        except Exception as exc:
            self._show_error('播放失败', str(exc))

    def on_pause_music(self):
        if not self.audio_player:
            return

        try:
            paused = self.audio_player.pause()
            self.music_panel.on_playback_paused(paused)
            self.status_bar.showMessage('音乐已暂停。' if paused else '音乐已继续播放。')
        except Exception as exc:
            self._show_error('播放失败', str(exc))

    def on_stop_music(self):
        self._shutdown_audio_player()
        self.music_panel.on_playback_stopped(reset_position=True)
        self.status_bar.showMessage('音乐已停止。')

    def on_seek_music(self, position_seconds):
        if not self.audio_player or not self.audio_path:
            return

        try:
            self.audio_player.seek(position_seconds)
            self.music_panel.sync_playback_position(position_seconds)
            if self.audio_player.is_paused:
                self.music_panel.on_playback_paused(True)
            else:
                self.music_panel.on_playback_started()
                self.music_panel.sync_playback_position(position_seconds)
            self.status_bar.showMessage(f'已跳转到 {position_seconds:.1f} 秒。')
        except Exception as exc:
            self._show_error('跳转失败', str(exc))

    def on_music_progress_requested(self):
        if not self.audio_player:
            return

        position = self.audio_player.get_position()
        self.music_panel.sync_playback_position(position)
        if self.audio_player.is_playing and not self.audio_player.is_busy():
            self._shutdown_audio_player()
            self.music_panel.on_playback_stopped(reset_position=True)
            self.status_bar.showMessage('音乐播放结束。')

    def on_export_csv(self):
        if self.current_subject is None:
            self._show_warning('提示', '请先选择受试者。')
            return
        if self.sws_results is None:
            self._show_warning('提示', '请先执行 SWS 检测。')
            return

        exporter = CSVExporter()
        music_params = self.music_panel.music_results or {}
        exporter.add_result(
            subject_id=self.current_subject.get('subject_id', ''),
            sws_duration=self.sws_results.get('total_sws_duration', 0),
            avg_delta_power=self.sws_results.get('avg_delta_power', 0),
            avg_pitch=music_params.get('avg_pitch', 0) or 0,
            avg_tempo=music_params.get('avg_tempo', 0) or 0,
            session_id=self.current_session_id or '',
            channel_name=self.current_channel_name or '',
            music_duration=self.music_panel.music_duration,
            scale=music_params.get('scale', ''),
            instrument=music_params.get('instrument', ''),
            instrument_program=music_params.get('instrument_program', ''),
        )

        csv_path = exporter.export()
        QMessageBox.information(self, '导出成功', f'数据已追加到：\n{csv_path}')
        self.status_bar.showMessage(f'数据已导出：{csv_path}')

    def on_volume_changed(self, volume):
        if self.audio_player:
            self.audio_player.set_volume(volume)

    def on_export_music(self):
        if not self.audio_path or not os.path.exists(self.audio_path):
            self._show_warning('提示', '请先生成音乐。')
            return
        if not self.midi_path or not os.path.exists(self.midi_path):
            self._show_warning('提示', '请先生成音乐。')
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
                f'音乐文件已导出：\n\nMIDI: {midi_dest}\nWAV: {wav_dest}',
            )
            self.status_bar.showMessage(f'音乐已导出到：{save_dir}')
        except Exception as exc:
            self._show_error('导出失败', str(exc))

    def closeEvent(self, event):
        if self._active_task:
            reply = QMessageBox.question(
                self,
                '确认退出',
                '后台任务仍在运行。是否取消当前任务并退出？',
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

            self.status_bar.showMessage('正在取消后台任务，请稍候...')
            self._cancel_background_tasks()
            if not self._wait_for_background_tasks():
                self._show_warning('退出失败', '后台任务仍在收尾，请稍后重试。')
                event.ignore()
                return

        self._shutdown_audio_player()
        super().closeEvent(event)

    def update_ui_state(self):
        busy = self._active_task is not None
        has_subject = self.current_subject is not None
        has_eeg = self.current_eeg_data is not None
        has_sws = self.sws_results is not None
        has_audio = bool(self.audio_path and os.path.exists(self.audio_path))
        has_midi = bool(self.midi_path and os.path.exists(self.midi_path))
        has_music_files = bool(
            has_audio or has_midi
        )
        has_exportable_music_bundle = has_audio and has_midi

        self.subject_panel.setEnabled(not busy)
        self.eeg_panel.setEnabled(not busy)
        self.sws_panel.setEnabled(not busy)
        self.music_panel.setEnabled(not busy)

        self.eeg_panel.set_enabled(has_subject and not busy)
        self.sws_panel.set_enabled(has_eeg and not busy)
        self.music_panel.set_generate_enabled(has_sws and not busy)
        self.music_panel.set_play_enabled(has_audio and not busy)
        self.music_panel.set_export_enabled(
            (has_sws or has_music_files) and not busy,
            can_export_music_bundle=has_exportable_music_bundle,
        )


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 9))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
