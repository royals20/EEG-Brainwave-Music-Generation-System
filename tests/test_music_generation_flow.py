import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_processing.feature_extractor import FeatureExtractor, extract_music_windows_from_signal
from eeg_processing.sws_detector import SWSDetector
from gui.workers import prepare_music_source_features, prepare_music_window_features
from music.mapper import EEGMusicMapper
from music.midi_generator import IndividualizedMIDIGenerator


def build_sparse_sws_signal(fs: int = 256, duration_seconds: int = 600):
    rng = np.random.default_rng(0)
    time_axis = np.arange(fs * duration_seconds, dtype=float) / fs
    eeg = rng.normal(0.0, 5.0, fs * duration_seconds)

    for start_second in (60, 180, 420):
        segment = slice(start_second * fs, (start_second + 30) * fs)
        eeg[segment] += 50.0 * np.sin(2 * np.pi * 1.0 * time_axis[segment])

    return eeg


def test_music_generation_prefers_detected_sws_epochs():
    fs = 256
    eeg = build_sparse_sws_signal(fs=fs)
    sws_results = SWSDetector().detect(eeg, fs)

    sws_features, sws_meta = prepare_music_source_features(
        eeg,
        fs,
        sws_results,
        FeatureExtractor(fs),
    )
    fallback_features, fallback_meta = prepare_music_source_features(
        eeg,
        fs,
        {'sws_epoch_count': 0, 'sws_epochs': []},
        FeatureExtractor(fs),
    )

    assert sws_meta['source_mode'] == 'sws_only'
    assert fallback_meta['source_mode'] == 'full_eeg_fallback'
    assert sws_meta['source_epoch_count'] == sws_results['sws_epoch_count']
    assert sws_meta['source_epoch_count'] < fallback_meta['source_epoch_count']
    assert sws_meta['effective_duration_seconds'] == sws_results['sws_epoch_count'] * 30
    assert len(sws_features) == sws_results['sws_epoch_count']
    assert len(fallback_features) == fallback_meta['source_epoch_count']


def test_music_window_extraction_uses_two_second_windows_and_compressed_sws_timeline():
    fs = 256
    eeg = build_sparse_sws_signal(fs=fs)
    sws_results = SWSDetector().detect(eeg, fs)
    window_features, window_meta = prepare_music_window_features(
        eeg,
        fs,
        sws_results,
        FeatureExtractor(fs),
        window_size=2.0,
        hop_size=1.0,
    )

    assert window_meta['source_mode'] == 'sws_only'
    assert window_meta['source_epoch_count'] == sws_results['sws_epoch_count']
    assert window_meta['music_window_count'] == len(window_features)
    assert len(window_features) == sws_results['sws_epoch_count'] * 29
    assert window_features[0]['start_time'] == 0.0
    assert window_features[-1]['end_time'] <= window_meta['effective_duration_seconds']


def test_structured_music_duration_matches_effective_sws_duration_without_global_rescaling():
    fs = 256
    eeg = build_sparse_sws_signal(fs=fs)
    sws_results = SWSDetector().detect(eeg, fs)
    extractor = FeatureExtractor(fs)
    _, sws_meta = prepare_music_source_features(eeg, fs, sws_results, extractor)
    sws_windows, _ = prepare_music_window_features(eeg, fs, sws_results, extractor)

    mapper = EEGMusicMapper()
    mapper.configure_individualized(extractor.get_individualized_stats(), sws_results['slow_wave_density'])
    notes = mapper.generate_structured_ambient_music(
        sws_windows,
        total_duration_seconds=sws_meta['effective_duration_seconds'],
        composition_settings={
            'keep_original_duration': True,
            'phrase_length_bars': 2,
            'section_length_bars': 8,
            'enable_pad': True,
            'enable_bass': True,
            'enable_motif': True,
        },
    )

    generator = IndividualizedMIDIGenerator(bpm=mapper.get_music_parameters()['bpm'])
    track_groups = generator._build_track_events(notes)

    beat_duration = 60.0 / mapper.get_music_parameters()['bpm']
    bar_duration = beat_duration * 4.0
    midi_end_time = max(event['end'] for group in track_groups for event in group['events'])

    assert mapper.get_total_duration() <= sws_meta['effective_duration_seconds']
    assert midi_end_time <= sws_meta['effective_duration_seconds']
    assert abs(sws_meta['effective_duration_seconds'] - midi_end_time) <= bar_duration


def test_music_window_extraction_handles_short_low_sample_rate_signal():
    fs = 8
    eeg = np.linspace(-20.0, 20.0, fs * 2, dtype=float)

    window_features = extract_music_windows_from_signal(
        eeg,
        fs,
        window_size=2.0,
        hop_size=1.0,
    )

    assert len(window_features) == 1
    assert window_features[0]['start_time'] == 0.0
    assert window_features[0]['end_time'] == pytest.approx(2.0)
