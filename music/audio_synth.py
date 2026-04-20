import os
import time
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

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

try:
    import fluidsynth  # noqa: F401

    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False

from utils.config import AUDIO_OUTPUT_DIR, ensure_directories


def _resolve_render_backend(
    backend: str,
    soundfont_path: Optional[str],
) -> Tuple[str, Optional[str]]:
    requested = (backend or 'auto').lower()
    has_soundfont = bool(soundfont_path and os.path.exists(soundfont_path))

    if requested == 'pretty_midi':
        return 'pretty_midi', None

    if requested in ('auto', 'fluidsynth'):
        if FLUIDSYNTH_AVAILABLE and has_soundfont:
            return 'fluidsynth', None
        fallback_reason = 'FluidSynth 不可用或 SoundFont 文件缺失'
        return 'pretty_midi', fallback_reason

    return 'pretty_midi', '渲染后端无效，已回退到 PrettyMIDI'


def write_wav(filepath: str, audio: np.ndarray, sample_rate: int):
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def synthesize_wav(
    midi_path: str,
    output_path: str = None,
    sample_rate: int = 44100,
    backend: str = 'auto',
    soundfont_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    ensure_directories()

    if output_path is None:
        output_path = os.path.join(AUDIO_OUTPUT_DIR, 'brain_music.wav')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError('未安装 pretty_midi 库，无法合成音频。')

    midi = pretty_midi.PrettyMIDI(midi_path)
    resolved_backend, fallback_reason = _resolve_render_backend(backend, soundfont_path)

    if not midi.instruments or all(len(instrument.notes) == 0 for instrument in midi.instruments):
        audio = np.zeros(int(sample_rate * 1), dtype=np.float32)
    elif resolved_backend == 'fluidsynth':
        try:
            audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
        except Exception:
            resolved_backend = 'pretty_midi'
            fallback_reason = 'FluidSynth 渲染失败，已回退到 PrettyMIDI'
            audio = midi.synthesize(fs=sample_rate)
    else:
        audio = midi.synthesize(fs=sample_rate)

    if len(audio) == 0:
        audio = np.zeros(int(sample_rate * 1), dtype=np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    write_wav(output_path, audio_int16, sample_rate)
    return output_path, {
        'requested_backend': (backend or 'auto').lower(),
        'resolved_backend': resolved_backend,
        'soundfont_path': soundfont_path or '',
        'fallback_reason': fallback_reason,
    }


class AudioSynthesizer:

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or AUDIO_OUTPUT_DIR
        self.audio_path = None
        self.sample_rate = 44100
        self.render_info: Dict[str, Any] = {}

    def synthesize(
        self,
        midi_path: str,
        filename: str = 'brain_music.wav',
        output_dir: str = None,
        backend: str = 'auto',
        soundfont_path: Optional[str] = None,
    ) -> str:
        if output_dir:
            self.output_dir = output_dir
        output_path = os.path.join(self.output_dir, filename)
        self.audio_path, self.render_info = synthesize_wav(
            midi_path,
            output_path,
            self.sample_rate,
            backend=backend,
            soundfont_path=soundfont_path,
        )
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
                'duration': wav_file.getnframes() / wav_file.getframerate(),
                **self.render_info,
            }

    def get_render_info(self) -> Dict[str, Any]:
        return dict(self.render_info)


class AudioPlayer:

    def __init__(self):
        self.is_playing = False
        self.is_paused = False
        self.current_file = None
        self.mixer_ready = False
        self._position_offset = 0.0
        self._play_anchor = None
        self._last_position = 0.0

        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self.mixer_ready = True
            except pygame.error as exc:
                raise RuntimeError(f'音频播放初始化失败：{exc}') from exc

    def _ensure_backend(self):
        if not PYGAME_AVAILABLE or not self.mixer_ready:
            raise ImportError('未安装 pygame 库，无法播放音频。')

    def _mark_playback_started(self, audio_path: str, start_seconds: float):
        self.current_file = audio_path
        self.is_playing = True
        self.is_paused = False
        self._position_offset = max(0.0, start_seconds)
        self._last_position = self._position_offset
        self._play_anchor = time.monotonic()

    def play(self, audio_path: str, start_seconds: float = 0.0):
        self._ensure_backend()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'未找到音频文件：{audio_path}')

        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play(start=max(0.0, start_seconds))
        except pygame.error as exc:
            raise RuntimeError(f'无法播放音频文件：{exc}') from exc

        self._mark_playback_started(audio_path, start_seconds)

    def pause(self):
        self._ensure_backend()
        if not self.current_file:
            return False

        if self.is_playing and not self.is_paused:
            self._last_position = self.get_position()
            pygame.mixer.music.pause()
            self.is_playing = False
            self.is_paused = True
            self._play_anchor = None
        elif self.is_paused:
            pygame.mixer.music.unpause()
            self.is_playing = True
            self.is_paused = False
            self._position_offset = self._last_position
            self._play_anchor = time.monotonic()

        return self.is_paused

    def stop(self):
        if self.mixer_ready:
            pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.current_file = None
        self._position_offset = 0.0
        self._play_anchor = None
        self._last_position = 0.0

    def seek(self, position_seconds: float):
        self._ensure_backend()
        if not self.current_file:
            raise RuntimeError('当前没有可用于跳转的音频文件。')

        was_paused = self.is_paused
        self.play(self.current_file, start_seconds=max(0.0, position_seconds))
        if was_paused:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.is_paused = True
            self._play_anchor = None
            self._position_offset = max(0.0, position_seconds)
            self._last_position = self._position_offset

    def set_volume(self, volume: float):
        if self.mixer_ready:
            pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))

    def get_position(self) -> float:
        if self.is_playing and self._play_anchor is not None:
            self._last_position = self._position_offset + (time.monotonic() - self._play_anchor)
        return max(0.0, self._last_position)

    def is_busy(self) -> bool:
        if self.mixer_ready and not self.is_paused:
            return pygame.mixer.music.get_busy()
        return False

    def close(self):
        self.stop()
        if self.mixer_ready and pygame.mixer.get_init():
            pygame.mixer.quit()
        self.mixer_ready = False
