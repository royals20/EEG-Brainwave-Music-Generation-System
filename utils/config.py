import os
import sys
from typing import Dict, Optional


INVALID_SUBJECT_ID_CHARS = set('<>:"/\\|?*')


def get_base_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = get_base_dir()

DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'eeg_system.db')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SUBJECTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'subjects')
MIDI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'midi')
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'audio')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'csv')


def validate_subject_id(subject_id: str) -> str:
    normalized = '' if subject_id is None else str(subject_id).strip()
    if not normalized:
        raise ValueError('受试者 ID 不能为空。')

    if normalized in {'.', '..'}:
        raise ValueError('受试者 ID 不能为 "." 或 ".."。')

    if any(char in INVALID_SUBJECT_ID_CHARS for char in normalized):
        raise ValueError('受试者 ID 不能包含以下字符：<>:"/\\|?*。')

    if any(ord(char) < 32 for char in normalized):
        raise ValueError('受试者 ID 不能包含控制字符。')

    return normalized


def get_subject_dirs(subject_id: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    subject_root = os.path.join(output_dir or OUTPUT_DIR, 'subjects')
    subject_dir = os.path.join(subject_root, subject_id)
    return {
        'subject_dir': subject_dir,
        'midi': os.path.join(subject_dir, 'midi'),
        'audio': os.path.join(subject_dir, 'audio'),
        'csv': os.path.join(subject_dir, 'csv'),
        'data': os.path.join(subject_dir, 'data'),
    }


def ensure_subject_directories(subject_id: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    validated_subject_id = validate_subject_id(subject_id)
    dirs = get_subject_dirs(validated_subject_id, output_dir=output_dir)
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

SWS_DETECTION_CONFIG = {
    'window_size': 30,
    'delta_band': (0.5, 4.0),
    'delta_power_threshold': 1e-6,
    'amplitude_threshold_uv': 75.0,
    'slow_wave_ratio_threshold': 0.2,
    'min_sws_duration': 5,
}

MUSIC_MAPPING_CONFIG = {
    'base_pitch': 60,
    'pitch_range': 14,
    'max_frequency': 4.0,
    'min_velocity': 45,
    'max_velocity': 88,
    'base_tempo': 60,
    'mode': 'healing_ambient',
    'music_window_seconds': 2.0,
    'music_hop_seconds': 1.0,
    'keep_original_duration': True,
    'target_music_duration_minutes': 0.0,
    'phrase_length_bars': 2,
    'section_length_bars': 8,
    'melody_density': 2,
    'harmony_richness': 4,
    'humanization_ms': 20,
    'enable_pad': True,
    'enable_bass': True,
    'enable_motif': True,
    'render_backend': 'auto',
    'soundfont_path': '',
    'random_seed': 17,
}

EEG_SIMULATION_CONFIG = {
    'duration_minutes': 10,
    'sample_rate': 256,
    'sws_ratio': 0.3,
}

def ensure_directories():
    for dir_path in [OUTPUT_DIR, SUBJECTS_OUTPUT_DIR, MIDI_OUTPUT_DIR, AUDIO_OUTPUT_DIR, CSV_OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    data_dir = os.path.dirname(DATABASE_PATH)
    os.makedirs(data_dir, exist_ok=True)


ensure_directories()
