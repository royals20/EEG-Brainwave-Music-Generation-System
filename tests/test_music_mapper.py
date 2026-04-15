import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music.mapper import EEGMusicMapper
from music.midi_generator import MIDIGenerator


def test_music_mapper_user_params():
    stats = {
        'delta_power_level': 'medium',
        'amplitude_level': 'medium',
        'avg_slow_wave_frequency': 1.0,
    }
    features = [
        {'delta_power': 0.0, 'mean_amplitude': 50.0, 'dominant_frequency': 1.0},
        {'delta_power': 0.0, 'mean_amplitude': 60.0, 'dominant_frequency': 4.0},
    ]

    default_mapper = EEGMusicMapper()
    default_mapper.configure_individualized(stats, 3.0)
    default_notes = default_mapper.generate_ambient_structure(features)
    default_params = default_mapper.get_music_parameters()

    custom_mapper = EEGMusicMapper()
    custom_mapper.configure_individualized(
        stats,
        3.0,
        {'base_pitch': 48, 'pitch_range': 12, 'base_tempo': 90},
    )
    custom_notes = custom_mapper.generate_ambient_structure(features)
    custom_params = custom_mapper.get_music_parameters()

    assert default_notes, "Default mapper should produce notes"
    assert custom_notes, "Custom mapper should produce notes"
    assert default_params['avg_tempo'] == 60, "Default tempo should remain unchanged"
    assert custom_params['avg_tempo'] == 90, "Base tempo override should affect output tempo"
    assert custom_notes[0]['pitch'] < default_notes[0]['pitch'], "Base pitch override should lower generated pitch"
    assert max(note['pitch'] for note in custom_notes) <= 60, "Pitch range override should constrain the upper note range"

    print("music mapper override test passed")


def test_instrument_program_flows_from_mapper_to_midi():
    stats = {
        'delta_power_level': 'high',
        'amplitude_level': 'high',
        'avg_slow_wave_frequency': 1.0,
    }
    features = [
        {'delta_power': 1.0, 'mean_amplitude': 80.0, 'dominant_frequency': 1.2},
    ]

    mapper = EEGMusicMapper()
    mapper.configure_individualized(stats, 3.0)
    mapped_notes = mapper.generate_ambient_structure(features)
    individual_params = mapper.get_individual_params()
    music_params = mapper.get_music_parameters()

    assert mapped_notes, "Mapper should produce notes for instrument propagation"
    assert individual_params['instrument_program'] == 49, "High amplitude should map to warm strings program"
    assert mapped_notes[0]['instrument'] == 49, "Mapped notes should retain instrument metadata"
    assert music_params['instrument_program'] == 49, "Music parameters should expose the resolved program"

    midi_generator = MIDIGenerator(bpm=individual_params['tempo'], instrument_program=None)
    resolved_program = midi_generator.individual_generator._resolve_main_instrument_program(mapped_notes)

    assert resolved_program == 49, "MIDI generator should infer instrument program from mapped notes when not passed explicitly"

    print("music instrument propagation test passed")


if __name__ == "__main__":
    test_music_mapper_user_params()
    test_instrument_program_flows_from_mapper_to_midi()
