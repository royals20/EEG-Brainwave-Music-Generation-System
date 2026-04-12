import numpy as np
from typing import Dict, Any, List, Tuple

from utils.config import MUSIC_MAPPING_CONFIG


PENTATONIC_SCALE = [60, 62, 64, 67, 69]
MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71]
MINOR_SCALE = [60, 62, 63, 65, 67, 68, 70]

SCALES = {
    'low': PENTATONIC_SCALE,
    'medium': MAJOR_SCALE,
    'high': MINOR_SCALE
}

SCALE_NAMES = {
    'low': 'Pentatonic (五声音阶)',
    'medium': 'Major (大调)',
    'high': 'Minor (小调)'
}

INSTRUMENT_PROGRAMS = {
    'low': 88,
    'medium': 0,
    'high': 49
}

INSTRUMENT_NAMES = {
    'low': 'Ambient Pad',
    'medium': 'Soft Piano',
    'high': 'Warm Strings'
}


def get_scale_for_delta_power(delta_level: str) -> List[int]:
    return SCALES.get(delta_level, MAJOR_SCALE)


def get_tempo_for_frequency(frequency: float) -> int:
    if frequency < 0.8:
        return 50
    elif frequency > 1.2:
        return 70
    else:
        return 60


def get_note_density_for_slow_wave_density(slow_wave_density: float) -> float:
    if slow_wave_density < 3:
        return 2.0
    elif slow_wave_density > 6:
        return 0.5
    else:
        return 1.0


def get_instrument_for_amplitude(amplitude_level: str) -> int:
    return INSTRUMENT_PROGRAMS.get(amplitude_level, 0)


def quantize_to_scale(pitch: int, scale: List[int]) -> int:
    octave = pitch // 12
    note_in_octave = pitch % 12
    
    scale_notes_mod = [n % 12 for n in scale]
    
    closest_note = min(scale_notes_mod, key=lambda x: min(abs(x - note_in_octave), 12 - abs(x - note_in_octave)))
    
    return octave * 12 + closest_note


def map_amplitude_to_velocity(amplitude: float, min_amp: float, max_amp: float) -> int:
    if max_amp == min_amp:
        ratio = 0.5
    else:
        ratio = (amplitude - min_amp) / (max_amp - min_amp)
    
    velocity = int(40 + ratio * 40)
    return max(40, min(80, velocity))


class IndividualizedMusicMapper:
    
    def __init__(self):
        self.scale = MAJOR_SCALE
        self.tempo = 60
        self.note_density = 1.0
        self.instrument_program = 0
        self.mapped_features = []
        self.individual_params = {}
    
    def configure_from_eeg_stats(self, stats: Dict[str, Any], slow_wave_density: float = 3.0):
        delta_level = stats.get('delta_power_level', 'medium')
        amplitude_level = stats.get('amplitude_level', 'medium')
        frequency = stats.get('avg_slow_wave_frequency', 1.0)
        
        self.scale = get_scale_for_delta_power(delta_level)
        self.tempo = get_tempo_for_frequency(frequency)
        self.note_density = get_note_density_for_slow_wave_density(slow_wave_density)
        self.instrument_program = get_instrument_for_amplitude(amplitude_level)
        
        self.individual_params = {
            'scale_name': SCALE_NAMES.get(delta_level, 'Major'),
            'tempo': self.tempo,
            'note_density': self.note_density,
            'instrument_name': INSTRUMENT_NAMES.get(amplitude_level, 'Soft Piano'),
            'delta_level': delta_level,
            'amplitude_level': amplitude_level,
            'frequency': frequency
        }
    
    def map_features(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = []
        
        if not eeg_features:
            return self.mapped_features
        
        amplitudes = [f.get('mean_amplitude', 0) for f in eeg_features]
        delta_powers = [f.get('delta_power', 0) for f in eeg_features]
        
        min_amp = min(amplitudes) if amplitudes else 0
        max_amp = max(amplitudes) if amplitudes else 200
        
        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4
        
        note_interval = bar_duration / (4 / self.note_density)
        
        current_time = 0.0
        
        for i, features in enumerate(eeg_features):
            delta_power = features.get('delta_power', 0)
            amplitude = features.get('mean_amplitude', 0)
            frequency = features.get('dominant_frequency', 1.0)
            
            base_pitch = 60 + int((frequency / 4.0) * 12)
            pitch = quantize_to_scale(base_pitch, self.scale)
            
            velocity = map_amplitude_to_velocity(amplitude, min_amp, max_amp)
            
            note_duration = beat_duration * 4
            
            note = {
                'pitch': pitch,
                'velocity': velocity,
                'start_time': current_time,
                'duration': note_duration,
                'delta_power': delta_power,
                'amplitude': amplitude
            }
            self.mapped_features.append(note)
            
            if delta_power > np.mean(delta_powers):
                harmony_pitch = quantize_to_scale(pitch + 4, self.scale)
                harmony_note = {
                    'pitch': harmony_pitch,
                    'velocity': max(40, velocity - 15),
                    'start_time': current_time,
                    'duration': note_duration * 0.75,
                    'is_harmony': True
                }
                self.mapped_features.append(harmony_note)
            
            current_time += note_interval
        
        return self.mapped_features
    
    def generate_ambient_structure(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = []
        
        if not eeg_features:
            return self.mapped_features
        
        amplitudes = [f.get('mean_amplitude', 0) for f in eeg_features]
        delta_powers = [f.get('delta_power', 0) for f in eeg_features]
        
        min_amp = min(amplitudes) if amplitudes else 0
        max_amp = max(amplitudes) if amplitudes else 200
        avg_delta = np.mean(delta_powers) if delta_powers else 0
        
        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4
        
        current_time = 0.0
        
        for i, features in enumerate(eeg_features):
            delta_power = features.get('delta_power', 0)
            amplitude = features.get('mean_amplitude', 0)
            frequency = features.get('dominant_frequency', 1.0)
            
            base_pitch = 60 + int((frequency / 4.0) * 12)
            root_pitch = quantize_to_scale(base_pitch, self.scale)
            
            velocity = map_amplitude_to_velocity(amplitude, min_amp, max_amp)
            
            note_duration = bar_duration
            
            root_note = {
                'pitch': root_pitch,
                'velocity': velocity,
                'start_time': current_time,
                'duration': note_duration,
                'instrument': self.instrument_program
            }
            self.mapped_features.append(root_note)
            
            if delta_power > avg_delta * 0.8:
                third_pitch = quantize_to_scale(root_pitch + 4, self.scale)
                third_note = {
                    'pitch': third_pitch,
                    'velocity': max(40, velocity - 10),
                    'start_time': current_time + beat_duration * 0.5,
                    'duration': note_duration * 0.8,
                    'instrument': self.instrument_program
                }
                self.mapped_features.append(third_note)
            
            if delta_power > avg_delta * 1.2:
                fifth_pitch = quantize_to_scale(root_pitch + 7, self.scale)
                fifth_note = {
                    'pitch': fifth_pitch,
                    'velocity': max(40, velocity - 20),
                    'start_time': current_time + beat_duration,
                    'duration': note_duration * 0.6,
                    'instrument': self.instrument_program
                }
                self.mapped_features.append(fifth_note)
            
            current_time += bar_duration
        
        return self.mapped_features
    
    def get_music_parameters(self) -> Dict[str, Any]:
        if not self.mapped_features:
            return {}
        
        pitches = [f['pitch'] for f in self.mapped_features if not f.get('is_harmony', False)]
        velocities = [f['velocity'] for f in self.mapped_features]
        
        return {
            'avg_pitch': np.mean(pitches) if pitches else 60,
            'avg_velocity': np.mean(velocities) if velocities else 60,
            'avg_tempo': self.tempo,
            'pitch_range': (min(pitches), max(pitches)) if pitches else (60, 60),
            'note_count': len(self.mapped_features),
            'bpm': self.tempo,
            'scale': self.individual_params.get('scale_name', 'Major'),
            'instrument': self.individual_params.get('instrument_name', 'Soft Piano')
        }
    
    def get_total_duration(self) -> float:
        if not self.mapped_features:
            return 0.0
        
        max_end = max(f['start_time'] + f['duration'] for f in self.mapped_features)
        return max_end
    
    def get_individual_params(self) -> Dict[str, Any]:
        return self.individual_params


class EEGMusicMapper:
    
    def __init__(self, config: dict = None):
        self.config = config or MUSIC_MAPPING_CONFIG.copy()
        self.mapped_features = []
        self.individual_mapper = IndividualizedMusicMapper()
    
    def configure_individualized(self, stats: Dict[str, Any], slow_wave_density: float = 3.0):
        self.individual_mapper.configure_from_eeg_stats(stats, slow_wave_density)
    
    def map_features(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = self.individual_mapper.map_features(eeg_features)
        return self.mapped_features
    
    def generate_ambient_structure(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = self.individual_mapper.generate_ambient_structure(eeg_features)
        return self.mapped_features
    
    def get_music_parameters(self) -> Dict[str, Any]:
        return self.individual_mapper.get_music_parameters()
    
    def get_total_duration(self) -> float:
        return self.individual_mapper.get_total_duration()
    
    def get_individual_params(self) -> Dict[str, Any]:
        return self.individual_mapper.get_individual_params()
