import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from gui.main_window import MainWindow


class FakeWorker:

    def __init__(self):
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True


class FakeThread:

    def __init__(self, wait_result=True):
        self.wait_result = wait_result
        self.wait_calls = []

    def isRunning(self):
        return True

    def wait(self, timeout_ms):
        self.wait_calls.append(timeout_ms)
        return self.wait_result


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def create_window(tmp_path):
    db = DatabaseManager(
        db_path=str(tmp_path / 'test.db'),
        output_dir=str(tmp_path / 'output'),
    )
    return MainWindow(db=db)


def test_restore_csv_history_uses_saved_sample_rate_without_prompt(tmp_path, monkeypatch, app):
    csv_path = tmp_path / 'history.csv'
    pd.DataFrame({'C3': [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

    window = create_window(tmp_path)
    monkeypatch.setattr(
        window.eeg_panel,
        '_prompt_csv_sample_rate',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('should not prompt')),
    )

    restored, warning = window._restore_eeg_session(
        {
            'file_path': str(csv_path),
            'sample_rate': 128.0,
        }
    )

    assert restored is True
    assert warning is None
    assert window.current_fs == pytest.approx(128.0)
    assert window.eeg_panel.loader.sample_rate == pytest.approx(128.0)

    window.close()
    app.processEvents()


def test_close_event_cancels_background_tasks_before_exit(tmp_path, monkeypatch, app):
    window = create_window(tmp_path)
    worker = FakeWorker()
    thread = FakeThread(wait_result=True)
    shutdown_state = {'called': False}

    window._active_task = 'sws'
    window._sws_worker = worker
    window._sws_thread = thread

    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
    monkeypatch.setattr(window, '_shutdown_audio_player', lambda: shutdown_state.__setitem__('called', True))

    event = QCloseEvent()
    window.closeEvent(event)
    app.processEvents()

    assert event.isAccepted()
    assert worker.cancel_requested is True
    assert thread.wait_calls == [5000]
    assert shutdown_state['called'] is True


def test_close_event_stays_open_when_user_rejects_cancellation(tmp_path, monkeypatch, app):
    window = create_window(tmp_path)
    worker = FakeWorker()

    window._active_task = 'music'
    window._music_worker = worker

    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.No)

    event = QCloseEvent()
    window.closeEvent(event)
    app.processEvents()

    assert not event.isAccepted()
    assert worker.cancel_requested is False


def test_save_sws_result_serializes_numpy_scalars(tmp_path, app):
    window = create_window(tmp_path)
    assert window.db.add_subject('S200', 'SWS Test')
    window.current_session_id = window.db.add_eeg_session('S200', '/tmp/eeg.edf', 256.0, 1, 60.0)
    window.sws_results = {
        'total_sws_duration': 30.0,
        'avg_delta_power': np.float64(12.5),
        'sws_epochs': [
            {
                'epoch_index': np.int64(0),
                'start_time': 0.0,
                'end_time': 30.0,
                'slow_wave_count': np.int64(28),
                'slow_wave_ratio': np.float64(0.95),
                'is_sws': np.bool_(True),
            }
        ],
    }

    window._save_sws_result()
    latest = window.db.get_latest_results('S200')
    restored_epochs = json.loads(latest['sws_segments'])

    assert restored_epochs[0]['epoch_index'] == 0
    assert restored_epochs[0]['slow_wave_count'] == 28
    assert restored_epochs[0]['is_sws'] is True

    window.close()
    app.processEvents()


def test_restore_music_history_allows_audio_playback_but_disables_bundle_export_when_midi_is_missing(tmp_path, app):
    audio_path = tmp_path / 'restored.wav'
    audio_path.write_bytes(b'RIFF')

    window = create_window(tmp_path)
    restored, warning = window._restore_music_history(
        {
            'midi_path': str(tmp_path / 'missing.mid'),
            'audio_path': str(audio_path),
            'avg_pitch': 61.0,
            'avg_tempo': 62.0,
            'music_duration': 30.0,
        }
    )
    app.processEvents()

    assert restored is True
    assert warning is None
    assert window.audio_path == str(audio_path)
    assert window.midi_path is None
    assert window.music_panel.play_btn.isEnabled() is True
    assert window.music_panel.export_music_btn.isEnabled() is False

    window.close()
    app.processEvents()


def test_invalid_subject_selection_clears_main_window_context(tmp_path, monkeypatch, app):
    window = create_window(tmp_path)
    warnings = []

    monkeypatch.setattr(window, '_show_warning', lambda title, message: warnings.append((title, message)))

    window.current_subject = {'subject_id': 'S001', 'name': 'Valid Subject'}
    info = window.eeg_panel.loader.load_simulated(duration_minutes=0.1, sample_rate=32, sws_ratio=0.3)
    window.eeg_panel.update_info(info)
    window.current_eeg_data = window.eeg_panel.loader.get_channel_data(0)
    window.current_fs = window.eeg_panel.loader.sample_rate
    window.sws_results = {'total_sws_duration': 30.0}
    window.music_panel.update_results({'midi_path': None, 'audio_path': None, 'music_duration': 10.0})

    window.on_subject_selected({'subject_id': 'A:B', 'name': 'Invalid Subject'})
    app.processEvents()

    assert warnings
    assert window.current_subject is None
    assert window.current_eeg_data is None
    assert window.sws_results is None
    assert window.music_panel.music_results is None
    assert window.eeg_panel.loader.data is None

    window.close()
    app.processEvents()


def test_deleting_current_subject_clears_main_window_state(tmp_path, monkeypatch, app):
    window = create_window(tmp_path)
    assert window.db.add_subject('S300', 'Delete Me')

    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)

    window.subject_panel.load_subjects()
    window.subject_panel.table.selectRow(0)
    window.subject_panel.select_subject()

    info = window.eeg_panel.loader.load_simulated(duration_minutes=0.1, sample_rate=32, sws_ratio=0.3)
    window.eeg_panel.update_info(info)
    window.current_eeg_data = window.eeg_panel.loader.get_channel_data(0)
    window.current_fs = window.eeg_panel.loader.sample_rate
    window.current_channel_name = window.eeg_panel.get_selected_channel_name()
    window.sws_results = {'total_sws_duration': 30.0}
    window.music_panel.update_results({'midi_path': None, 'audio_path': None, 'music_duration': 10.0})
    window.update_ui_state()

    assert window.current_subject is not None
    assert window.current_eeg_data is not None

    window.subject_panel.delete_subject()
    app.processEvents()

    assert window.current_subject is None
    assert window.current_eeg_data is None
    assert window.sws_results is None
    assert window.music_panel.music_results is None
    assert window.eeg_panel.loader.data is None
    assert not window.eeg_panel.import_csv_btn.isEnabled()
    assert not window.music_panel.generate_btn.isEnabled()

    window.close()
    app.processEvents()
