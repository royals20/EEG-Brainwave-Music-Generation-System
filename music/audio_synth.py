import os
import subprocess
import numpy as np
from typing import Optional, Dict, Any
import wave

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from utils.config import AUDIO_OUTPUT_DIR, ensure_directories


def synthesize_wav(midi_path: str, output_path: str = None,
                   sample_rate: int = 44100) -> str:
    ensure_directories()
    
    if output_path is None:
        output_path = os.path.join(AUDIO_OUTPUT_DIR, 'brain_music.wav')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError("pretty_midi library required for synthesis")
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    if not pm.instruments or all(len(inst.notes) == 0 for inst in pm.instruments):
        audio = np.zeros(int(sample_rate * 60), dtype=np.float32)
    else:
        audio = pm.synthesize(fs=sample_rate)
    
    if len(audio) == 0:
        audio = np.zeros(int(sample_rate * 60), dtype=np.float32)
    
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    write_wav(output_path, audio_int16, sample_rate)
    
    return output_path


def write_wav(filepath: str, audio: np.ndarray, sample_rate: int):
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


class AudioSynthesizer:
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or AUDIO_OUTPUT_DIR
        self.audio_path = None
        self.sample_rate = 44100
    
    def synthesize(self, midi_path: str, filename: str = 'brain_music.wav', output_dir: str = None) -> str:
        if output_dir:
            self.output_dir = output_dir
        output_path = os.path.join(self.output_dir, filename)
        self.audio_path = synthesize_wav(midi_path, output_path, self.sample_rate)
        return self.audio_path
    
    def get_audio_info(self) -> Dict[str, Any]:
        if self.audio_path is None or not os.path.exists(self.audio_path):
            return {}
        
        with wave.open(self.audio_path, 'rb') as wav_file:
            return {
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'framerate': wav_file.getframerate(),
                'frames': wav_file.getnframes(),
                'duration': wav_file.getnframes() / wav_file.getframerate()
            }


class AudioPlayer:
    
    def __init__(self):
        self.is_playing = False
        self.is_paused = False
        self.current_file = None
        
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    
    def play(self, audio_path: str):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame library required for audio playback")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.is_playing:
            self.stop()
        
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        self.is_playing = True
        self.is_paused = False
        self.current_file = audio_path
    
    def pause(self):
        if PYGAME_AVAILABLE and self.is_playing:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
            else:
                pygame.mixer.music.pause()
                self.is_paused = True
    
    def stop(self):
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
    
    def seek(self, position_seconds: float):
        if PYGAME_AVAILABLE and self.current_file:
            was_playing = self.is_playing and not self.is_paused
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.current_file)
            pygame.mixer.music.play(start=position_seconds)
            self.is_playing = True
            self.is_paused = False
            if not was_playing:
                pygame.mixer.music.pause()
                self.is_paused = True
    
    def set_volume(self, volume: float):
        if PYGAME_AVAILABLE:
            pygame.mixer.music.set_volume(max(0, min(1, volume)))
    
    def get_position(self) -> float:
        if PYGAME_AVAILABLE and self.is_playing:
            return pygame.mixer.music.get_pos() / 1000.0
        return 0
    
    def is_busy(self) -> bool:
        if PYGAME_AVAILABLE:
            return pygame.mixer.music.get_busy()
        return False
