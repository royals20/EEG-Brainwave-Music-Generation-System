import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_processing.eeg_loader import EEGLoader
from eeg_processing.feature_extractor import FeatureExtractor
from eeg_processing.sws_detector import SWSDetector


def test_sws_detector_handles_volt_scale_signal():
    fs = 256
    duration_seconds = 120
    t = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_volts = 80e-6 * np.sin(2 * np.pi * 1.0 * t)

    detector = SWSDetector()
    results = detector.detect(eeg_volts, fs)

    assert results['total_sws_duration'] >= 30, 'Volt-scale EEG should be normalized and produce non-zero SWS duration'
    assert results['sws_epoch_count'] >= 1, 'At least one epoch should be detected as SWS'


def test_feature_extractor_handles_volt_scale_signal():
    fs = 256
    duration_seconds = 60
    t = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_volts = 60e-6 * np.sin(2 * np.pi * 0.9 * t)

    extractor = FeatureExtractor(fs)
    features = extractor.extract(eeg_volts, smooth=False)

    assert features, 'Feature extractor should return epoch features'
    assert max(feature['mean_amplitude'] for feature in features) > 10, 'Amplitude should be interpreted in microvolts'
    assert any(feature['slow_wave_ratio'] > 0 for feature in features), 'Slow-wave ratio should not collapse to zero for volt-scale input'


def test_csv_loader_auto_scales_to_microvolts():
    fs = 256
    duration_seconds = 30
    t = np.linspace(0, duration_seconds, fs * duration_seconds, endpoint=False)
    eeg_volts = 50e-6 * np.sin(2 * np.pi * 1.2 * t)

    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8') as f:
        csv_path = f.name
        f.write('ch1\n')
        for value in eeg_volts:
            f.write(f'{value}\n')

    try:
        loader = EEGLoader()
        info = loader.load_csv(csv_path, sample_rate=fs)
        channel_data = loader.get_channel_data(0)

        assert info['channel_count'] == 1, 'Single-column CSV should load as one channel'
        assert np.max(np.abs(channel_data)) > 10, 'CSV volt-scale EEG should be scaled to microvolts'
    finally:
        os.remove(csv_path)
