import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.config import DATABASE_PATH, OUTPUT_DIR, get_subject_dirs, validate_subject_id


EEG_SESSIONS_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS eeg_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT NOT NULL,
        file_path TEXT,
        sample_rate REAL,
        channel_count INTEGER,
        duration_seconds REAL,
        import_time TEXT,
        FOREIGN KEY (subject_id) REFERENCES subjects (subject_id) ON DELETE CASCADE
    )
'''

SWS_RESULTS_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS sws_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        sws_duration REAL,
        avg_delta_power REAL,
        sws_segments TEXT,
        detection_time TEXT,
        FOREIGN KEY (session_id) REFERENCES eeg_sessions (id) ON DELETE CASCADE
    )
'''

MUSIC_RESULTS_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS music_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        midi_path TEXT,
        audio_path TEXT,
        avg_pitch REAL,
        avg_tempo REAL,
        music_duration REAL,
        generation_time TEXT,
        FOREIGN KEY (session_id) REFERENCES eeg_sessions (id) ON DELETE CASCADE
    )
'''

CASCADE_TABLE_SPECS = [
    (
        'eeg_sessions',
        EEG_SESSIONS_TABLE_SQL,
        ['id', 'subject_id', 'file_path', 'sample_rate', 'channel_count', 'duration_seconds', 'import_time'],
    ),
    (
        'sws_results',
        SWS_RESULTS_TABLE_SQL,
        ['id', 'session_id', 'sws_duration', 'avg_delta_power', 'sws_segments', 'detection_time'],
    ),
    (
        'music_results',
        MUSIC_RESULTS_TABLE_SQL,
        ['id', 'session_id', 'midi_path', 'audio_path', 'avg_pitch', 'avg_tempo', 'music_duration', 'generation_time'],
    ),
]

INDEX_SQL_STATEMENTS = [
    '''
    CREATE INDEX IF NOT EXISTS idx_eeg_sessions_subject_import_time
    ON eeg_sessions (subject_id, import_time DESC, id DESC)
    ''',
    '''
    CREATE INDEX IF NOT EXISTS idx_sws_results_session_detection_time
    ON sws_results (session_id, detection_time DESC, id DESC)
    ''',
    '''
    CREATE INDEX IF NOT EXISTS idx_music_results_session_generation_time
    ON music_results (session_id, generation_time DESC, id DESC)
    ''',
]


class DatabaseManager:

    def __init__(self, db_path: str = None, output_dir: str = None):
        self.db_path = db_path or DATABASE_PATH
        self.output_dir = output_dir or OUTPUT_DIR
        self.last_error = None
        self._ensure_db_directory()
        self._init_database()

    def _set_last_error(self, message: Optional[str]):
        self.last_error = message

    def _ensure_db_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    @contextmanager
    def connection(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def readonly_connection(self):
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS subjects (
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
            self._ensure_cascade_tables(conn)

        self.cleanup_orphan_records()

    def _ensure_cascade_tables(self, conn: sqlite3.Connection):
        for _, create_sql, _ in CASCADE_TABLE_SPECS:
            conn.execute(create_sql)

        if any(self._table_needs_cascade_migration(conn, table_name) for table_name, _, _ in CASCADE_TABLE_SPECS):
            self._migrate_cascade_tables(conn)

        for _, create_sql, _ in CASCADE_TABLE_SPECS:
            conn.execute(create_sql)
        self._ensure_indexes(conn)

    def _ensure_indexes(self, conn: sqlite3.Connection):
        for index_sql in INDEX_SQL_STATEMENTS:
            conn.execute(index_sql)

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> List[str]:
        rows = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
        return [row['name'] for row in rows]

    def _table_needs_cascade_migration(self, conn: sqlite3.Connection, table_name: str) -> bool:
        if not self._table_exists(conn, table_name):
            return False

        fk_rows = conn.execute(f'PRAGMA foreign_key_list({table_name})').fetchall()
        if not fk_rows:
            return True

        return any(str(row['on_delete']).upper() != 'CASCADE' for row in fk_rows)

    def _migrate_cascade_tables(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        legacy_mappings = []

        cursor.execute('PRAGMA foreign_keys = OFF')
        try:
            for table_name, _, _ in CASCADE_TABLE_SPECS:
                if not self._table_exists(conn, table_name):
                    continue
                legacy_name = f'{table_name}__legacy'
                cursor.execute(f'ALTER TABLE {table_name} RENAME TO {legacy_name}')
                legacy_mappings.append((table_name, legacy_name))

            for _, create_sql, _ in CASCADE_TABLE_SPECS:
                cursor.execute(create_sql)

            for table_name, legacy_name in legacy_mappings:
                expected_columns = next(columns for name, _, columns in CASCADE_TABLE_SPECS if name == table_name)
                available_columns = self._table_columns(conn, legacy_name)
                common_columns = [column for column in expected_columns if column in available_columns]
                if common_columns:
                    column_list = ', '.join(common_columns)
                    cursor.execute(
                        f'INSERT INTO {table_name} ({column_list}) '
                        f'SELECT {column_list} FROM {legacy_name}'
                    )

            for _, legacy_name in legacy_mappings:
                cursor.execute(f'DROP TABLE {legacy_name}')
        finally:
            cursor.execute('PRAGMA foreign_keys = ON')

    def cleanup_orphan_records(self):
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                DELETE FROM sws_results
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM eeg_sessions
                    WHERE eeg_sessions.id = sws_results.session_id
                )
                '''
            )
            cursor.execute(
                '''
                DELETE FROM music_results
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM eeg_sessions
                    WHERE eeg_sessions.id = music_results.session_id
                )
                '''
            )
            cursor.execute(
                '''
                DELETE FROM eeg_sessions
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM subjects
                    WHERE subjects.subject_id = eeg_sessions.subject_id
                )
                '''
            )

    def _remove_subject_output_directory(self, subject_id: str):
        try:
            validate_subject_id(subject_id)
        except ValueError:
            return

        subject_dir = os.path.abspath(
            get_subject_dirs(subject_id, output_dir=self.output_dir)['subject_dir']
        )
        subjects_root = os.path.abspath(os.path.join(self.output_dir, 'subjects'))

        if os.path.commonpath([subject_dir, subjects_root]) != subjects_root:
            raise ValueError(f'受试者输出目录不安全：{subject_dir}')

        if os.path.isdir(subject_dir):
            shutil.rmtree(subject_dir)

    def add_subject(
        self,
        subject_id: str,
        name: str,
        age: int = None,
        gender: str = None,
        psqi_score: float = None,
        group_type: str = None,
        record_date: str = None,
    ) -> bool:
        self._set_last_error(None)
        try:
            subject_id = validate_subject_id(subject_id)
            with self.connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute(
                    '''
                    INSERT INTO subjects
                    (subject_id, name, age, gender, psqi_score, group_type, record_date, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (subject_id, name, age, gender, psqi_score, group_type, record_date, now, now),
                )
            return True
        except sqlite3.IntegrityError:
            self._set_last_error('受试者 ID 已存在。')
            return False
        except ValueError as exc:
            self._set_last_error(str(exc))
            return False

    def update_subject(self, subject_id: str, **kwargs) -> bool:
        self._set_last_error(None)
        if not kwargs:
            self._set_last_error('没有可更新的字段。')
            return False
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                kwargs['updated_at'] = datetime.now().isoformat()
                set_clause = ', '.join([f'{key} = ?' for key in kwargs.keys()])
                values = list(kwargs.values()) + [subject_id]
                cursor.execute(f'UPDATE subjects SET {set_clause} WHERE subject_id = ?', values)
                success = cursor.rowcount > 0
                if not success:
                    self._set_last_error(f'未找到受试者 {subject_id}。')
            return success
        except Exception as exc:
            self._set_last_error(f'更新受试者失败：{exc}')
            return False

    def delete_subject(self, subject_id: str) -> bool:
        self._set_last_error(None)
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM subjects WHERE subject_id = ?', (subject_id,))
                success = cursor.rowcount > 0
                if success:
                    self._remove_subject_output_directory(subject_id)
                else:
                    self._set_last_error(f'未找到受试者 {subject_id}。')
            return success
        except Exception as exc:
            self._set_last_error(f'删除受试者失败：{exc}')
            return False

    def get_subject(self, subject_id: str) -> Optional[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            row = conn.execute('SELECT * FROM subjects WHERE subject_id = ?', (subject_id,)).fetchone()
            return dict(row) if row else None

    def get_all_subjects(self) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            rows = conn.execute('SELECT * FROM subjects ORDER BY created_at DESC').fetchall()
            return [dict(row) for row in rows]

    def search_subjects(self, keyword: str) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            rows = conn.execute(
                '''
                SELECT * FROM subjects
                WHERE subject_id LIKE ? OR name LIKE ?
                ORDER BY created_at DESC
                ''',
                (f'%{keyword}%', f'%{keyword}%'),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_eeg_session(
        self,
        subject_id: str,
        file_path: str,
        sample_rate: float,
        channel_count: int,
        duration_seconds: float,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                '''
                INSERT INTO eeg_sessions
                (subject_id, file_path, sample_rate, channel_count, duration_seconds, import_time)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (subject_id, file_path, sample_rate, channel_count, duration_seconds, now),
            )
            return cursor.lastrowid

    def add_sws_result(self, session_id: int, sws_duration: float, avg_delta_power: float, sws_segments: str) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                '''
                INSERT INTO sws_results
                (session_id, sws_duration, avg_delta_power, sws_segments, detection_time)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (session_id, sws_duration, avg_delta_power, sws_segments, now),
            )
            return cursor.lastrowid

    def upsert_sws_result(self, session_id: int, sws_duration: float, avg_delta_power: float, sws_segments: str) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            row = cursor.execute(
                '''
                SELECT id FROM sws_results
                WHERE session_id = ?
                ORDER BY detection_time DESC, id DESC
                LIMIT 1
                ''',
                (session_id,),
            ).fetchone()

            if row:
                result_id = row['id']
                cursor.execute(
                    '''
                    UPDATE sws_results
                    SET sws_duration = ?, avg_delta_power = ?, sws_segments = ?, detection_time = ?
                    WHERE id = ?
                    ''',
                    (sws_duration, avg_delta_power, sws_segments, now, result_id),
                )
                return result_id

            cursor.execute(
                '''
                INSERT INTO sws_results
                (session_id, sws_duration, avg_delta_power, sws_segments, detection_time)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (session_id, sws_duration, avg_delta_power, sws_segments, now),
            )
            return cursor.lastrowid

    def add_music_result(
        self,
        session_id: int,
        midi_path: str,
        audio_path: str,
        avg_pitch: float,
        avg_tempo: float,
        music_duration: float,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                '''
                INSERT INTO music_results
                (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, generation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, now),
            )
            return cursor.lastrowid

    def upsert_music_result(
        self,
        session_id: int,
        midi_path: str,
        audio_path: str,
        avg_pitch: float,
        avg_tempo: float,
        music_duration: float,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            row = cursor.execute(
                '''
                SELECT id FROM music_results
                WHERE session_id = ?
                ORDER BY generation_time DESC, id DESC
                LIMIT 1
                ''',
                (session_id,),
            ).fetchone()

            if row:
                result_id = row['id']
                cursor.execute(
                    '''
                    UPDATE music_results
                    SET midi_path = ?, audio_path = ?, avg_pitch = ?, avg_tempo = ?, music_duration = ?, generation_time = ?
                    WHERE id = ?
                    ''',
                    (midi_path, audio_path, avg_pitch, avg_tempo, music_duration, now, result_id),
                )
                return result_id

            cursor.execute(
                '''
                INSERT INTO music_results
                (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, generation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, now),
            )
            return cursor.lastrowid

    def get_subject_sessions(self, subject_id: str) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            rows = conn.execute(
                '''
                SELECT * FROM eeg_sessions
                WHERE subject_id = ?
                ORDER BY import_time DESC
                ''',
                (subject_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_latest_results(self, subject_id: str) -> Optional[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            row = conn.execute(
                '''
                SELECT e.id AS session_id,
                       e.subject_id,
                       e.file_path,
                       e.sample_rate,
                       e.channel_count,
                       e.duration_seconds,
                       e.import_time,
                       s.sws_duration,
                       s.avg_delta_power,
                       s.sws_segments,
                       s.detection_time,
                       m.midi_path,
                       m.audio_path,
                       m.avg_pitch,
                       m.avg_tempo,
                       m.music_duration,
                       m.generation_time
                FROM eeg_sessions e
                LEFT JOIN sws_results s ON s.id = (
                    SELECT sr.id
                    FROM sws_results sr
                    WHERE sr.session_id = e.id
                    ORDER BY sr.detection_time DESC, sr.id DESC
                    LIMIT 1
                )
                LEFT JOIN music_results m ON m.id = (
                    SELECT mr.id
                    FROM music_results mr
                    WHERE mr.session_id = e.id
                    ORDER BY mr.generation_time DESC, mr.id DESC
                    LIMIT 1
                )
                WHERE e.subject_id = ?
                ORDER BY e.import_time DESC
                LIMIT 1
                ''',
                (subject_id,),
            ).fetchone()
            return dict(row) if row else None
