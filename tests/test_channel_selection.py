import os
import sys
import tempfile

import numpy as np
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from gui.main_window import MainWindow


def test_channel_selection_updates_analysis_input():
    app = QApplication.instance() or QApplication([])

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(os.path.join(tmpdir, 'test.db'))
        db.add_subject('S001', 'Test User')

        window = MainWindow()
        window.db = db
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

        assert not np.array_equal(first_channel_data, second_channel_data), 'Analysis input should follow selected channel'
        assert window.current_channel_index == 1, 'Current channel index should track UI selection'
        assert window.current_session_id == first_session_id, 'Changing channel should not create a new EEG session'
        assert window.sws_results is None, 'SWS results should be cleared after changing channel'
        assert window.music_features is None, 'Music results should be cleared after changing channel'

        window.close()
        app.processEvents()

    print('channel selection integration test passed')


if __name__ == '__main__':
    test_channel_selection_updates_analysis_input()
