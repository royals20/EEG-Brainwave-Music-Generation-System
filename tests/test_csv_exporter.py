import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.csv_exporter import CSVExporter


def test_csv_exporter_appends_history_and_expands_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = CSVExporter(tmpdir)
        exporter.add_result(
            subject_id='S001',
            sws_duration=120.0,
            avg_delta_power=1.2,
            avg_pitch=61.0,
            avg_tempo=62.0,
        )
        output_path = exporter.export('experiment_results.csv')

        exporter = CSVExporter(tmpdir)
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

        rows = CSVExporter(tmpdir).load_existing(output_path)

        assert len(rows) == 2, 'Repeated exports should append rows instead of overwriting history'
        assert rows[0]['subject_id'] == 'S001', 'Original export row should remain in the CSV'
        assert rows[0]['channel_name'] == '', 'Older rows should be preserved when new columns are added later'
        assert rows[1]['subject_id'] == 'S002', 'New export row should be appended to the end'
        assert rows[1]['session_id'] == '2', 'Appended rows should retain traceability fields'
        assert rows[1]['channel_name'] == 'C4', 'Appended rows should retain channel metadata'
        assert rows[1]['music_duration'] == '1800.0', 'Appended rows should retain music metadata'

    print('csv exporter append test passed')


if __name__ == '__main__':
    test_csv_exporter_appends_history_and_expands_columns()
