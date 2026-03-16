"""
Audio synthesis: convert MIDI → WAV using FluidSynth (via pretty_midi).

Falls back to a simple sine-wave synthesiser if FluidSynth is unavailable.
"""

import os
from typing import List, Optional

import numpy as np

from algorithms.music_mapper import NoteEvent
from utils.logger import logger


def midi_to_wav(midi_path: str, wav_path: str, sf2_path: Optional[str] = None) -> str:
    """Render a MIDI file to WAV using FluidSynth.

    Parameters
    ----------
    midi_path : str
        Input .mid file.
    wav_path : str
        Output .wav file.
    sf2_path : str, optional
        Path to a SoundFont (.sf2) file.

    Returns
    -------
    wav_path : str
    """
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=44100, sf2_path=sf2_path)
        _write_wav(wav_path, audio, 44100)
        logger.info("WAV synthesised via FluidSynth: %s", wav_path)
        return wav_path
    except Exception as exc:
        logger.warning("FluidSynth failed (%s) – using sine synthesis.", exc)
        return _sine_synth_from_midi(midi_path, wav_path)


def synthesise_from_notes(notes: List[NoteEvent], wav_path: str, sr: int = 44100) -> str:
    """Directly synthesise note events to WAV (sine waves)."""
    if not notes:
        logger.warning("No notes to synthesise.")
        return wav_path
    total_dur = max(n.start_time + n.duration for n in notes) + 1.0
    audio = np.zeros(int(total_dur * sr))
    for n in notes:
        freq = 440.0 * 2 ** ((n.pitch - 69) / 12.0)
        t = np.arange(int(n.duration * sr)) / sr
        # Simple sine with ADSR-ish envelope
        env = _envelope(len(t), sr)
        tone = np.sin(2 * np.pi * freq * t) * env * (n.velocity / 127.0)
        start_idx = int(n.start_time * sr)
        end_idx = start_idx + len(tone)
        if end_idx > len(audio):
            tone = tone[: len(audio) - start_idx]
            end_idx = len(audio)
        audio[start_idx:end_idx] += tone

    # Normalise
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    _write_wav(wav_path, audio, sr)
    logger.info("Sine-synth WAV written: %s", wav_path)
    return wav_path


def _sine_synth_from_midi(midi_path: str, wav_path: str) -> str:
    """Fallback: read MIDI and synthesise with sine waves."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes: List[NoteEvent] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(NoteEvent(
                pitch=n.pitch,
                start_time=n.start,
                duration=n.end - n.start,
                velocity=n.velocity,
            ))
    return synthesise_from_notes(notes, wav_path)


def _envelope(length: int, sr: int) -> np.ndarray:
    """Simple ADSR-like envelope."""
    attack = min(int(0.05 * sr), length)
    release = min(int(0.1 * sr), length)
    env = np.ones(length)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release)
    return env


def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a numpy array to a WAV file."""
    import wave
    import struct

    audio_16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(audio_16)}h", *audio_16))
