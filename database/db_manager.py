import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from utils.config import DATABASE_PATH


class DatabaseManager:
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def connection(self):
        """数据库连接上下文管理器，自动管理连接的获取、提交和关闭"""
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
        """只读数据库连接上下文管理器，不自动提交"""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器，支持显式提交或回滚"""
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
            
            cursor.execute('''
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
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS eeg_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    file_path TEXT,
                    sample_rate REAL,
                    channel_count INTEGER,
                    duration_seconds REAL,
                    import_time TEXT,
                    FOREIGN KEY (subject_id) REFERENCES subjects (subject_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sws_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    sws_duration REAL,
                    avg_delta_power REAL,
                    sws_segments TEXT,
                    detection_time TEXT,
                    FOREIGN KEY (session_id) REFERENCES eeg_sessions (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS music_results (
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
            ''')
    
    def add_subject(self, subject_id: str, name: str, age: int = None,
                    gender: str = None, psqi_score: float = None,
                    group_type: str = None, record_date: str = None) -> bool:
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO subjects 
                    (subject_id, name, age, gender, psqi_score, group_type, record_date, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (subject_id, name, age, gender, psqi_score, group_type, record_date, now, now))
            return True
        except sqlite3.IntegrityError:
            return False
    
    def update_subject(self, subject_id: str, **kwargs) -> bool:
        if not kwargs:
            return False
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                kwargs['updated_at'] = datetime.now().isoformat()
                set_clause = ', '.join([f'{k} = ?' for k in kwargs.keys()])
                values = list(kwargs.values()) + [subject_id]
                cursor.execute(f'UPDATE subjects SET {set_clause} WHERE subject_id = ?', values)
                success = cursor.rowcount > 0
            return success
        except Exception:
            return False
    
    def delete_subject(self, subject_id: str) -> bool:
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM subjects WHERE subject_id = ?', (subject_id,))
                success = cursor.rowcount > 0
            return success
        except Exception:
            return False
    
    def get_subject(self, subject_id: str) -> Optional[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM subjects WHERE subject_id = ?', (subject_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_subjects(self) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM subjects ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def search_subjects(self, keyword: str) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM subjects 
                WHERE subject_id LIKE ? OR name LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{keyword}%', f'%{keyword}%'))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def add_eeg_session(self, subject_id: str, file_path: str, sample_rate: float,
                        channel_count: int, duration_seconds: float) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO eeg_sessions 
                (subject_id, file_path, sample_rate, channel_count, duration_seconds, import_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (subject_id, file_path, sample_rate, channel_count, duration_seconds, now))
            session_id = cursor.lastrowid
        return session_id
    
    def add_sws_result(self, session_id: int, sws_duration: float,
                       avg_delta_power: float, sws_segments: str) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO sws_results 
                (session_id, sws_duration, avg_delta_power, sws_segments, detection_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, sws_duration, avg_delta_power, sws_segments, now))
            result_id = cursor.lastrowid
        return result_id
    
    def add_music_result(self, session_id: int, midi_path: str, audio_path: str,
                         avg_pitch: float, avg_tempo: float, music_duration: float) -> int:
        with self.connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO music_results 
                (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, generation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, midi_path, audio_path, avg_pitch, avg_tempo, music_duration, now))
            result_id = cursor.lastrowid
        return result_id
    
    def get_subject_sessions(self, subject_id: str) -> List[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM eeg_sessions WHERE subject_id = ? ORDER BY import_time DESC
            ''', (subject_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_latest_results(self, subject_id: str) -> Optional[Dict[str, Any]]:
        with self.readonly_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.*, s.sws_duration, s.avg_delta_power, m.midi_path, m.audio_path,
                       m.avg_pitch, m.avg_tempo, m.music_duration
                FROM eeg_sessions e
                LEFT JOIN sws_results s ON e.id = s.session_id
                LEFT JOIN music_results m ON e.id = m.session_id
                WHERE e.subject_id = ?
                ORDER BY e.import_time DESC
                LIMIT 1
            ''', (subject_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
