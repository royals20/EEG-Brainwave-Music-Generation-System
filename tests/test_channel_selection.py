import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from gui.main_window import MainWindow


def test_channel_selection_updates_analysis_input(tmp_path):
    app = QApplication.instance() or QApplication([])
    db = DatabaseManager(
        db_path=str(tmp_path / 'test.db'),
        output_dir=str(tmp_path / 'output'),
    )
    assert db.add_subject('S001', 'Test User')

    window = MainWindow(db=db)
    window.current_subject = {'subject_id': 'S001', 'name': 'Test User'}

    info = window.eeg_panel.loader.load_simulated(duration_minutes=0.1, sample_rate=64, sws_ratio=0.5)
    window.eeg_panel.update_info(info)
    window.on_eeg_loaded(info)

    first_channel_data = window.current_eeg_data.copy()
    first_session_id = window.current_session_id

    window.sws_results = {'dummy': True}
    window.music_features = [{'pitch': 60}]

    window.eeg_panel.channel_combo.setCurrentIndex(1)
    second_channel_data = window.current_eeg_data.copy()

    assert not np.array_equal(first_channel_data, second_channel_data)
    assert window.current_channel_index == 1
    assert window.current_session_id == first_session_id
    assert window.sws_results is None
    assert window.music_features is None

    window.close()
    app.processEvents()
