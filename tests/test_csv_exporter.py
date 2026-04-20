import os
import sys

import reports.csv_exporter as csv_exporter_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.csv_exporter import CSVExporter


def test_csv_exporter_appends_history_and_expands_columns(tmp_path):
    exporter = CSVExporter(str(tmp_path))
    exporter.add_result(
        subject_id='S001',
        sws_duration=120.0,
        avg_delta_power=1.2,
        avg_pitch=61.0,
        avg_tempo=62.0,
    )
    output_path = exporter.export('experiment_results.csv')

    exporter = CSVExporter(str(tmp_path))
    exporter.add_result(
        subject_id='S002',
        sws_duration=240.0,
        avg_delta_power=2.4,
        avg_pitch=63.0,
        avg_tempo=64.0,
        session_id='2',
        channel_name='C4',
        music_duration=1800.0,
        scale='Minor',
        instrument='Soft Piano',
        export_time='2026-04-15T15:00:00',
    )
    exporter.export('experiment_results.csv')

    rows = CSVExporter(str(tmp_path)).load_existing(output_path)
    assert len(rows) == 2
    assert rows[0]['subject_id'] == 'S001'
    assert rows[0]['channel_name'] == ''
    assert rows[1]['subject_id'] == 'S002'
    assert rows[1]['session_id'] == '2'
    assert rows[1]['channel_name'] == 'C4'
    assert rows[1]['music_duration'] == '1800.0'


def test_csv_exporter_appends_without_reloading_existing_rows_when_schema_is_unchanged(tmp_path, monkeypatch):
    exporter = CSVExporter(str(tmp_path))
    exporter.add_result(
        subject_id='S001',
        sws_duration=120.0,
        avg_delta_power=1.2,
        avg_pitch=61.0,
        avg_tempo=62.0,
    )
    output_path = exporter.export('experiment_results.csv')

    original_load_csv_rows = csv_exporter_module._load_csv_rows
    load_calls = {'count': 0}

    def counting_load_csv_rows(filepath):
        load_calls['count'] += 1
        return original_load_csv_rows(filepath)

    monkeypatch.setattr(csv_exporter_module, '_load_csv_rows', counting_load_csv_rows)

    exporter = CSVExporter(str(tmp_path))
    exporter.add_result(
        subject_id='S002',
        sws_duration=240.0,
        avg_delta_power=2.4,
        avg_pitch=63.0,
        avg_tempo=64.0,
    )
    exporter.export('experiment_results.csv')

    assert load_calls['count'] == 0

    rows = original_load_csv_rows(output_path)
    assert len(rows) == 2
    assert rows[1]['subject_id'] == 'S002'
