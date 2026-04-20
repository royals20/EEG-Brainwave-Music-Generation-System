import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_processing.eeg_loader import EEGLoader


def test_inspect_csv_infers_sample_rate_from_timestamp_column(tmp_path):
    csv_path = tmp_path / 'multi_channel.csv'
    pd.DataFrame(
        {
            'timestamp': [0.0, 1.0, 2.0],
            'C3': [1.0, 2.0, 3.0],
            'C4': [4.0, 5.0, 6.0],
        }
    ).to_csv(csv_path, index=False)

    loader = EEGLoader()
    csv_info = loader.inspect_csv(str(csv_path))

    assert csv_info['channel_count'] == 2
    assert csv_info['sample_count'] == 3
    assert csv_info['channel_names'] == ['C3', 'C4']
    assert csv_info['time_column_name'] == 'timestamp'
    assert csv_info['can_auto_infer_sample_rate'] is True
    assert csv_info['inferred_sample_rate'] == pytest.approx(1.0)


def test_load_csv_uses_inferred_sample_rate_when_time_column_is_regular(tmp_path):
    csv_path = tmp_path / 'with_time.csv'
    pd.DataFrame(
        {
            'time': [0.0, 0.25, 0.5, 0.75],
            'Fz': [1.0, 2.0, 3.0, 4.0],
            'Cz': [4.0, 3.0, 2.0, 1.0],
        }
    ).to_csv(csv_path, index=False)

    loader = EEGLoader()
    info = loader.load_csv(str(csv_path))

    assert info['sample_rate'] == pytest.approx(4.0)
    assert info['duration'] == pytest.approx(1.0)
    assert info['channel_names'] == ['Fz', 'Cz']
    assert loader.get_all_data().shape == (2, 4)
    assert np.allclose(loader.get_time_vector(), np.array([0.0, 0.25, 0.5, 0.75]))


def test_inspect_csv_requires_manual_sample_rate_when_time_column_is_irregular(tmp_path):
    csv_path = tmp_path / 'irregular.csv'
    pd.DataFrame(
        {
            'timestamp': [0.0, 0.5, 1.4],
            'Fz': [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    loader = EEGLoader()
    csv_info = loader.inspect_csv(str(csv_path))

    assert csv_info['can_auto_infer_sample_rate'] is False
    assert csv_info['inferred_sample_rate'] is None
    assert csv_info['sample_rate_reason']


def test_load_csv_requires_manual_sample_rate_when_no_time_column_exists(tmp_path):
    csv_path = tmp_path / 'single_channel.csv'
    pd.DataFrame({'Fp1': [50e-6, 60e-6, 70e-6]}).to_csv(csv_path, index=False)

    loader = EEGLoader()
    with pytest.raises(ValueError, match='采样率'):
        loader.load_csv(str(csv_path))


def test_csv_loader_supports_manual_single_channel_data(tmp_path):
    csv_path = tmp_path / 'manual_single_channel.csv'
    pd.DataFrame({'Fp1': [50e-6, 60e-6, 70e-6]}).to_csv(csv_path, index=False)

    loader = EEGLoader()
    info = loader.load_csv(str(csv_path), sample_rate=2)

    assert info['channel_count'] == 1
    assert info['duration'] == 1.5
    assert info['channel_names'] == ['Fp1']
    assert loader.get_channel_data(0).max() > 10


def test_csv_loader_rejects_non_numeric_columns(tmp_path):
    csv_path = tmp_path / 'invalid.csv'
    pd.DataFrame(
        {
            'Fz': [1.0, 2.0, 3.0],
            'label': ['a', 'b', 'c'],
        }
    ).to_csv(csv_path, index=False)

    loader = EEGLoader()
    with pytest.raises(ValueError, match='非数值'):
        loader.load_csv(str(csv_path), sample_rate=1)


def test_inspect_then_load_csv_reuses_cached_parse(tmp_path, monkeypatch):
    csv_path = tmp_path / 'cached.csv'
    pd.DataFrame(
        {
            'time': [0.0, 0.5, 1.0],
            'Fz': [1.0, 2.0, 3.0],
        }
    ).to_csv(csv_path, index=False)

    loader = EEGLoader()
    original_reader = loader._read_csv_frame
    read_calls = {'count': 0}

    def counting_reader(file_path):
        read_calls['count'] += 1
        return original_reader(file_path)

    monkeypatch.setattr(loader, '_read_csv_frame', counting_reader)

    csv_info = loader.inspect_csv(str(csv_path))
    loader.load_csv(str(csv_path), sample_rate=csv_info['inferred_sample_rate'])

    assert read_calls['count'] == 1
