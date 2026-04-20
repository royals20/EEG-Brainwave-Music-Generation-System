import os
import sys

import pytest
from PyQt5.QtWidgets import QApplication

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.music_panel import MusicPanel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_music_panel_round_trip_generation_params(app):
    panel = MusicPanel()

    assert panel.target_duration_spin.isEnabled() is True
    assert panel.generate_btn.text() == '生成音乐'
    assert panel.play_btn.text() == '播放'
    assert panel.pause_btn.text() == '暂停'
    assert panel.stop_btn.text() == '停止'

    panel.target_duration_spin.setValue(12.5)
    panel.base_pitch_spin.setValue(52)
    panel.pitch_range_spin.setValue(18)
    panel.tempo_spin.setValue(72)
    panel.melody_density_spin.setValue(5)
    panel.harmony_richness_spin.setValue(3)

    params = panel.get_generation_params()

    assert params['keep_original_duration'] is False
    assert params['target_duration_minutes'] == pytest.approx(12.5)
    assert params['base_pitch'] == 52
    assert params['pitch_range'] == 18
    assert params['base_tempo'] == 72
    assert params['melody_density'] == 5
    assert params['harmony_richness'] == 3
    assert params['phrase_length_bars'] == 2
    assert params['section_length_bars'] == 8
    assert params['humanization_ms'] == 20
    assert params['enable_pad'] is True
    assert params['enable_bass'] is True
    assert params['enable_motif'] is True
    assert params['render_backend'] == 'auto'
    assert params['soundfont_path'] == ''

    panel.target_duration_spin.setValue(0.0)
    params = panel.get_generation_params()
    assert params['keep_original_duration'] is True
    assert params['target_duration_minutes'] is None


def test_music_panel_export_bundle_requires_both_midi_and_wav(app):
    panel = MusicPanel()

    panel.update_results({'midi_path': 'brain.mid', 'audio_path': None, 'music_duration': 10.0})
    assert panel.export_music_btn.isEnabled() is False

    panel.update_results({'midi_path': None, 'audio_path': 'brain.wav', 'music_duration': 10.0})
    assert panel.export_music_btn.isEnabled() is False

    panel.update_results({'midi_path': 'brain.mid', 'audio_path': 'brain.wav', 'music_duration': 10.0})
    assert panel.export_music_btn.isEnabled() is True
