import json
import os
from typing import Any, Dict, Optional, Tuple


def build_restored_sws_results(
    session_data: Dict[str, Any],
    current_eeg_data,
    current_fs: Optional[float],
) -> Optional[Dict[str, Any]]:
    if session_data.get('sws_duration') is None:
        return None

    sws_epochs = []
    sws_segments = session_data.get('sws_segments')
    if sws_segments:
        try:
            sws_epochs = json.loads(sws_segments)
        except (TypeError, json.JSONDecodeError):
            sws_epochs = []

    total_sws_duration = session_data.get('sws_duration') or 0
    total_slow_waves = sum(epoch.get('slow_wave_count', 0) for epoch in sws_epochs)
    slow_wave_density = total_slow_waves / (total_sws_duration / 60.0) if total_sws_duration > 0 else 0

    total_epoch_count = 0
    if current_eeg_data is not None and current_fs:
        epoch_seconds = 30
        total_epoch_count = int(len(current_eeg_data) // (epoch_seconds * current_fs))

    return {
        'total_sws_duration': total_sws_duration,
        'avg_delta_power': session_data.get('avg_delta_power') or 0,
        'slow_wave_density': slow_wave_density,
        'sws_epoch_count': len(sws_epochs),
        'total_epoch_count': total_epoch_count,
        'epochs': sws_epochs,
        'sws_epochs': sws_epochs,
    }


def build_restored_music_results(session_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if session_data.get('midi_path') is None and session_data.get('audio_path') is None:
        return None, None

    midi_path = session_data.get('midi_path')
    audio_path = session_data.get('audio_path')

    midi_exists = bool(midi_path and os.path.exists(midi_path))
    audio_exists = bool(audio_path and os.path.exists(audio_path))
    warning = None
    if not midi_exists and not audio_exists:
        warning = '历史音乐文件缺失。'

    return (
        {
            'midi_path': midi_path if midi_exists else None,
            'audio_path': audio_path if audio_exists else None,
            'avg_pitch': session_data.get('avg_pitch'),
            'avg_tempo': session_data.get('avg_tempo'),
            'note_count': None,
            'music_duration': session_data.get('music_duration') or 0,
            'midi_exists': midi_exists,
            'audio_exists': audio_exists,
        },
        warning,
    )
