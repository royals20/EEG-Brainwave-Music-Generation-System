import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_processing.feature_extractor import FeatureExtractor, _smooth_array, smooth_features
from eeg_processing.sws_detector import SWSDetector, detect_slow_waves_hilbert


def test_sws_detector_handles_volt_scale_signal():
    fs = 256
    duration_seconds = 120
    time_axis = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_volts = 80e-6 * np.sin(2 * np.pi * 1.0 * time_axis)

    detector = SWSDetector()
    results = detector.detect(eeg_volts, fs)

    assert results['total_sws_duration'] >= 30
    assert results['sws_epoch_count'] >= 1


def test_feature_extractor_handles_volt_scale_signal():
    fs = 256
    duration_seconds = 60
    time_axis = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_volts = 60e-6 * np.sin(2 * np.pi * 0.9 * time_axis)

    extractor = FeatureExtractor(fs)
    features = extractor.extract(eeg_volts, smooth=False)

    assert features
    assert max(feature['mean_amplitude'] for feature in features) > 10
    assert any(feature['slow_wave_ratio'] > 0 for feature in features)


def test_slow_wave_count_tracks_cycles_instead_of_single_run():
    fs = 256
    duration_seconds = 30
    time_axis = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_uv = 40.0 * np.sin(2 * np.pi * 1.0 * time_axis)

    result = detect_slow_waves_hilbert(eeg_uv, fs, amplitude_threshold=75.0)

    assert 28 <= result['slow_wave_count'] <= 30
    assert result['slow_wave_ratio'] > 0.9


def test_feature_extractor_delta_level_is_not_inverted_by_skewed_distribution():
    extractor = FeatureExtractor(256)
    extractor.features_list = [
        {'delta_power': value, 'mean_amplitude': 60.0, 'dominant_frequency': 1.0}
        for value in [1.0, 100.0, 100.0, 100.0, 100.0]
    ]

    stats = extractor.get_individualized_stats()

    assert stats['delta_power_level'] == 'high'


def test_smooth_array_preserves_centered_edge_shrinking_behavior():
    smoothed = _smooth_array([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert np.allclose(smoothed, np.array([1.5, 2.0, 3.0, 4.0, 4.5]))


def test_smooth_features_matches_expected_centered_means():
    features = [
        {'delta_power': 1.0, 'mean_amplitude': 10.0},
        {'delta_power': 2.0, 'mean_amplitude': 20.0},
        {'delta_power': 3.0, 'mean_amplitude': 30.0},
        {'delta_power': 4.0, 'mean_amplitude': 40.0},
        {'delta_power': 5.0, 'mean_amplitude': 50.0},
    ]

    smoothed = smooth_features(features, window_size=3)

    assert np.allclose([item['delta_power'] for item in smoothed], [1.5, 2.0, 3.0, 4.0, 4.5])
    assert np.allclose([item['mean_amplitude'] for item in smoothed], [15.0, 20.0, 30.0, 40.0, 45.0])
