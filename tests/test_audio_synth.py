import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music.audio_synth import _resolve_render_backend


def test_render_backend_falls_back_to_pretty_midi_when_soundfont_is_missing():
    resolved_backend, fallback_reason = _resolve_render_backend('fluidsynth', r'C:\missing\not-found.sf2')

    assert resolved_backend == 'pretty_midi'
    assert fallback_reason is not None


def test_pretty_midi_backend_can_be_selected_explicitly():
    resolved_backend, fallback_reason = _resolve_render_backend('pretty_midi', None)

    assert resolved_backend == 'pretty_midi'
    assert fallback_reason is None
