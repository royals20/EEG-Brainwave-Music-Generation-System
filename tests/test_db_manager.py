import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from utils.config import ensure_subject_directories


def create_database_manager(tmp_path):
    return DatabaseManager(
        db_path=str(tmp_path / 'test.db'),
        output_dir=str(tmp_path / 'output'),
    )


def get_index_names(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row[0] for row in rows}


def test_database_manager_crud_and_latest_results(tmp_path):
    db = create_database_manager(tmp_path)

    assert db.add_subject('S001', '张三', age=25, gender='男', psqi_score=10.5)
    assert not db.add_subject('S001', '重复用户')
    assert db.last_error

    subject = db.get_subject('S001')
    assert subject is not None
    assert subject['name'] == '张三'

    assert db.update_subject('S001', age=26, psqi_score=12.0)
    updated_subject = db.get_subject('S001')
    assert updated_subject['age'] == 26
    assert updated_subject['psqi_score'] == 12.0

    assert db.add_subject('S002', '李四', age=30, gender='女')
    assert len(db.get_all_subjects()) == 2
    assert len(db.search_subjects('张')) == 1

    session_id = db.add_eeg_session('S001', '/tmp/eeg.edf', 256.0, 19, 3600.0)
    assert session_id > 0
    assert len(db.get_subject_sessions('S001')) == 1

    sws_id = db.add_sws_result(session_id, 1200.0, 150.5, '[]')
    assert sws_id > 0
    assert db.upsert_sws_result(session_id, 1500.0, 180.5, '[{"epoch_index": 0}]') == sws_id

    music_id = db.add_music_result(session_id, '/tmp/music.mid', '/tmp/music.wav', 440.0, 120.0, 180.0)
    assert music_id > 0
    assert db.upsert_music_result(session_id, '/tmp/music_v2.mid', '/tmp/music_v2.wav', 442.0, 90.0, 300.0) == music_id

    latest = db.get_latest_results('S001')
    assert latest is not None
    assert latest['sws_duration'] == 1500.0
    assert latest['avg_tempo'] == 90.0


def test_subject_failures_store_specific_last_error(tmp_path):
    db = create_database_manager(tmp_path)

    assert not db.update_subject('missing', age=18)
    assert db.last_error == '未找到受试者 missing。'

    assert not db.delete_subject('missing')
    assert db.last_error == '未找到受试者 missing。'


def test_delete_subject_cascades_records_and_output_directory(tmp_path):
    db = create_database_manager(tmp_path)
    assert db.add_subject('S010', '待删除用户')

    subject_dirs = ensure_subject_directories('S010', output_dir=str(tmp_path / 'output'))
    sample_output = os.path.join(subject_dirs['audio'], 'sample.wav')
    with open(sample_output, 'w', encoding='utf-8') as handle:
        handle.write('audio')

    session_id = db.add_eeg_session('S010', '/tmp/eeg.edf', 128.0, 2, 120.0)
    db.add_sws_result(session_id, 30.0, 1.2, '[]')
    db.add_music_result(session_id, '/tmp/sample.mid', '/tmp/sample.wav', 61.0, 62.0, 120.0)

    assert db.delete_subject('S010')
    assert db.get_subject('S010') is None
    assert db.get_subject_sessions('S010') == []
    assert db.get_latest_results('S010') is None
    assert not os.path.exists(subject_dirs['subject_dir'])

    assert db.add_subject('S010', '重建用户')
    assert db.get_subject_sessions('S010') == []


def test_startup_cleanup_removes_orphan_records(tmp_path):
    db_path = tmp_path / 'orphan.db'
    output_dir = tmp_path / 'output'
    db = DatabaseManager(db_path=str(db_path), output_dir=str(output_dir))
    assert db.add_subject('S100', '临时用户')
    session_id = db.add_eeg_session('S100', '/tmp/eeg.edf', 256.0, 2, 120.0)
    db.add_sws_result(session_id, 30.0, 1.0, '[]')
    db.add_music_result(session_id, '/tmp/sample.mid', '/tmp/sample.wav', 60.0, 60.0, 120.0)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute('PRAGMA foreign_keys = OFF')
        conn.execute("DELETE FROM subjects WHERE subject_id = 'S100'")
        conn.commit()

    repaired_db = DatabaseManager(db_path=str(db_path), output_dir=str(output_dir))
    assert repaired_db.get_subject('S100') is None
    assert repaired_db.get_subject_sessions('S100') == []


def test_database_manager_creates_history_indexes_for_new_database(tmp_path):
    db_path = tmp_path / 'indexed.db'
    DatabaseManager(db_path=str(db_path), output_dir=str(tmp_path / 'output'))

    index_names = get_index_names(db_path)
    assert 'idx_eeg_sessions_subject_import_time' in index_names
    assert 'idx_sws_results_session_detection_time' in index_names
    assert 'idx_music_results_session_generation_time' in index_names


def test_database_manager_creates_indexes_after_legacy_migration(tmp_path):
    db_path = tmp_path / 'legacy.db'
    output_dir = tmp_path / 'output'

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            '''
            CREATE TABLE subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                psqi_score REAL,
                group_type TEXT,
                record_date TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE eeg_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                file_path TEXT,
                sample_rate REAL,
                channel_count INTEGER,
                duration_seconds REAL,
                import_time TEXT,
                FOREIGN KEY (subject_id) REFERENCES subjects (subject_id)
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE sws_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                sws_duration REAL,
                avg_delta_power REAL,
                sws_segments TEXT,
                detection_time TEXT,
                FOREIGN KEY (session_id) REFERENCES eeg_sessions (id)
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE music_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                midi_path TEXT,
                audio_path TEXT,
                avg_pitch REAL,
                avg_tempo REAL,
                music_duration REAL,
                generation_time TEXT,
                FOREIGN KEY (session_id) REFERENCES eeg_sessions (id)
            )
            '''
        )
        conn.commit()

    DatabaseManager(db_path=str(db_path), output_dir=str(output_dir))

    index_names = get_index_names(db_path)
    assert 'idx_eeg_sessions_subject_import_time' in index_names
    assert 'idx_sws_results_session_detection_time' in index_names
    assert 'idx_music_results_session_generation_time' in index_names
