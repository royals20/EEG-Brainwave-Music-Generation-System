import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music.mapper import EEGMusicMapper, ROLE_TRACK_ORDER, get_scale_pitch_classes
from music.midi_generator import IndividualizedMIDIGenerator, MIDIGenerator


def build_window_features(count: int = 36, window_seconds: float = 2.0):
    features = []
    current_time = 0.0
    for index in range(count):
        features.append(
            {
                'start_time': current_time,
                'end_time': current_time + window_seconds,
                'duration': window_seconds,
                'calmness': 0.74 if index < count // 2 else 0.46,
                'density': 0.28 + (index % 4) * 0.07,
                'brightness': 0.58 if index < count // 2 else 0.30,
                'dominant_frequency': 0.9 + (index % 5) * 0.15,
                'mean_amplitude': 58.0 + (index % 6) * 6.0,
                'slow_wave_ratio': 0.22 + (index % 3) * 0.08,
                'energy_change': 0.04 * (index % 4),
                'delta_ratio': 0.60 if index < count // 2 else 0.42,
            }
        )
        current_time += 1.0
    return features


def test_legacy_ambient_mapping_still_supports_user_pitch_and_tempo_overrides():
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

    assert default_notes
    assert custom_notes
    assert default_params['avg_tempo'] == 60
    assert custom_params['avg_tempo'] == 90
    assert custom_notes[0]['pitch'] < default_notes[0]['pitch']
    assert max(note['pitch'] for note in custom_notes if not note.get('is_pulse')) <= 60


def test_structured_mapping_creates_pad_bass_and_motif_roles_in_scale():
    mapper = EEGMusicMapper()
    mapper.configure_individualized(
        {
            'delta_power_level': 'medium',
            'amplitude_level': 'medium',
            'avg_slow_wave_frequency': 1.0,
        },
        3.0,
        {'base_pitch': 60, 'base_tempo': 60},
    )
    features = mapper.generate_structured_ambient_music(
        build_window_features(),
        total_duration_seconds=48.0,
        composition_settings={
            'phrase_length_bars': 2,
            'section_length_bars': 8,
            'melody_density': 3,
            'harmony_richness': 4,
            'enable_pad': True,
            'enable_bass': True,
            'enable_motif': True,
        },
    )

    roles = {feature['role'] for feature in features}
    assert roles == set(ROLE_TRACK_ORDER)

    for feature in features:
        allowed_pitch_classes = get_scale_pitch_classes(60, feature.get('mode', 'major'))
        assert feature['pitch'] % 12 in allowed_pitch_classes


def test_structured_mapping_repeats_phrases_with_tail_variation_and_stepwise_motion():
    mapper = EEGMusicMapper()
    mapper.configure_individualized(
        {
            'delta_power_level': 'medium',
            'amplitude_level': 'medium',
            'avg_slow_wave_frequency': 1.0,
        },
        3.0,
    )
    features = mapper.generate_structured_ambient_music(
        build_window_features(),
        total_duration_seconds=48.0,
        composition_settings={
            'phrase_length_bars': 2,
            'section_length_bars': 8,
            'melody_density': 4,
            'enable_pad': True,
            'enable_bass': True,
            'enable_motif': True,
        },
    )

    motif_events = [feature for feature in features if feature['role'] == 'motif']
    repeated_events = [feature for feature in motif_events if feature.get('is_phrase_repeat')]
    assert repeated_events

    phrase_a_tail = [feature for feature in motif_events if not feature.get('is_phrase_repeat') and feature['phrase_index'] == 0][-1]
    phrase_b_tail = [feature for feature in repeated_events if feature['phrase_index'] == 0][-1]
    assert phrase_a_tail['pitch'] != phrase_b_tail['pitch']

    for current, nxt in zip(motif_events, motif_events[1:]):
        assert abs(nxt['pitch'] - current['pitch']) <= 7


def test_midi_generator_groups_tracks_without_pad_duplication():
    mapper = EEGMusicMapper()
    mapper.configure_individualized(
        {
            'delta_power_level': 'medium',
            'amplitude_level': 'medium',
            'avg_slow_wave_frequency': 1.0,
        },
        3.0,
    )
    features = mapper.generate_structured_ambient_music(
        build_window_features(),
        total_duration_seconds=32.0,
        composition_settings={
            'phrase_length_bars': 2,
            'section_length_bars': 8,
            'melody_density': 2,
            'enable_pad': True,
            'enable_bass': False,
            'enable_motif': True,
        },
    )

    midi_generator = MIDIGenerator(bpm=mapper.get_music_parameters()['bpm'], instrument_program=None)
    track_groups = midi_generator.individual_generator._build_track_events(features)

    assert [group['role'] for group in track_groups] == ['pad', 'motif']
    assert sum(len(group['events']) for group in track_groups) == len(features)
    assert all(group['program'] is not None for group in track_groups)


def test_midi_generator_scales_when_explicit_but_not_by_default():
    features = [
        {'pitch': 60, 'velocity': 60, 'start_time': 0.0, 'duration': 10.0, 'role': 'motif', 'track': 'motif'},
        {'pitch': 64, 'velocity': 65, 'start_time': 15.0, 'duration': 5.0, 'role': 'motif', 'track': 'motif'},
    ]
    generator = IndividualizedMIDIGenerator(bpm=60)

    default_tracks = generator._build_track_events(features)
    scaled_tracks = generator._build_track_events(features, target_duration_minutes=1.0)

    assert len(default_tracks) == 1
    assert len(default_tracks[0]['events']) == len(features)
    assert max(event['end'] for event in default_tracks[0]['events']) == 20.0
    assert max(event['end'] for event in scaled_tracks[0]['events']) == 60.0
