import os
import sys

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()

DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'eeg_system.db')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MIDI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'midi')
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'audio')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'csv')

def get_subject_dirs(subject_id: str):
    subject_dir = os.path.join(OUTPUT_DIR, 'subjects', subject_id)
    subject_midi_dir = os.path.join(subject_dir, 'midi')
    subject_audio_dir = os.path.join(subject_dir, 'audio')
    subject_csv_dir = os.path.join(subject_dir, 'csv')
    subject_data_dir = os.path.join(subject_dir, 'data')
    return {
        'subject_dir': subject_dir,
        'midi': subject_midi_dir,
        'audio': subject_audio_dir,
        'csv': subject_csv_dir,
        'data': subject_data_dir
    }

def ensure_subject_directories(subject_id: str):
    dirs = get_subject_dirs(subject_id)
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
    'pitch_range': 24,
    'max_frequency': 4.0,
    'min_velocity': 40,
    'max_velocity': 90,
    'base_tempo': 60,
    'target_music_duration_minutes': 30,
}

EEG_SIMULATION_CONFIG = {
    'duration_minutes': 10,
    'sample_rate': 256,
    'sws_ratio': 0.3,
}

def ensure_directories():
    for dir_path in [OUTPUT_DIR, MIDI_OUTPUT_DIR, AUDIO_OUTPUT_DIR, CSV_OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    data_dir = os.path.dirname(DATABASE_PATH)
    os.makedirs(data_dir, exist_ok=True)

ensure_directories()
