import os
import sys

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gui.eeg_panel as eeg_panel_module
from gui.eeg_panel import EEGPanel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_import_csv_confirms_inferred_sample_rate(tmp_path, monkeypatch, app):
    csv_path = tmp_path / 'with_timestamp.csv'
    pd.DataFrame(
        {
            'timestamp': [0.0, 0.25, 0.5],
            'C3': [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    panel = EEGPanel()
    emitted = []
    prompted = {}
    panel.eeg_loaded.connect(emitted.append)

    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(csv_path), 'CSV'))

    def fake_get_double(parent, title, message, value, minimum, maximum, decimals):
        prompted['title'] = title
        prompted['message'] = message
        prompted['value'] = value
        return value, True

    monkeypatch.setattr(QInputDialog, 'getDouble', fake_get_double)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    panel.import_csv()
    app.processEvents()

    assert prompted['value'] == pytest.approx(4.0)
    assert 'timestamp' in prompted['message']
    assert panel.loader.sample_rate == pytest.approx(4.0)
    assert emitted and emitted[0]['sample_rate'] == pytest.approx(4.0)


def test_import_csv_cancels_when_manual_sample_rate_prompt_is_rejected(tmp_path, monkeypatch, app):
    csv_path = tmp_path / 'without_time.csv'
    pd.DataFrame({'Fp1': [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

    panel = EEGPanel()
    emitted = []
    prompted = {}
    panel.eeg_loaded.connect(emitted.append)

    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(csv_path), 'CSV'))

    def fake_get_double(parent, title, message, value, minimum, maximum, decimals):
        prompted['message'] = message
        prompted['value'] = value
        return value, False

    monkeypatch.setattr(QInputDialog, 'getDouble', fake_get_double)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    panel.import_csv()
    app.processEvents()

    assert panel.loader.data is None
    assert emitted == []
    assert prompted['value'] == pytest.approx(256.0)


def test_view_delta_reuses_cached_preview(monkeypatch, app):
    panel = EEGPanel()
    info = panel.loader.load_simulated(duration_minutes=1, sample_rate=256, sws_ratio=0.5)
    panel.update_info(info)

    original_bandpass = eeg_panel_module.bandpass_filter
    call_count = {'count': 0}

    def counting_bandpass(eeg, low, high, fs, order=4):
        call_count['count'] += 1
        return original_bandpass(eeg, low, high, fs, order)

    monkeypatch.setattr(eeg_panel_module, 'bandpass_filter', counting_bandpass)

    panel.view_delta()
    panel.view_delta()
    app.processEvents()

    assert call_count['count'] == 1


def test_view_delta_handles_short_low_sample_rate_signal_without_crashing(app):
    panel = EEGPanel()
    panel.loader.data = np.array([np.linspace(-15.0, 15.0, 16, dtype=float)])
    panel.loader.sample_rate = 8.0
    panel.loader.channel_names = ['C3']
    panel.loader.duration = 2.0
    panel.loader.is_simulated = False

    panel.update_info(panel.loader.get_info())
    panel.view_delta()
    app.processEvents()

    assert 0 in panel._delta_view_cache
    assert len(panel._delta_view_cache[0][3]) == 16


def test_view_raw_uses_sample_indices_without_building_full_time_vector(monkeypatch, app):
    panel = EEGPanel()
    panel.loader.data = np.array([np.linspace(-15.0, 15.0, 200001, dtype=float)])
    panel.loader.sample_rate = 200.0
    panel.loader.channel_names = ['C3']
    panel.loader.duration = panel.loader.data.shape[1] / panel.loader.sample_rate
    panel.loader.is_simulated = False

    captured = {}
    monkeypatch.setattr(
        panel.loader,
        'get_time_vector',
        lambda: (_ for _ in ()).throw(AssertionError('should not build full time vector')),
    )
    monkeypatch.setattr(
        panel.canvas,
        'plot_eeg',
        lambda time_axis, eeg_data, title='': captured.update(
            time_axis=np.asarray(time_axis),
            eeg_data=np.asarray(eeg_data),
            title=title,
        ),
    )

    panel.view_raw()
    app.processEvents()

    assert len(captured['time_axis']) == len(captured['eeg_data'])
    assert len(captured['time_axis']) <= 50000
    assert captured['time_axis'][0] == pytest.approx(0.0)
