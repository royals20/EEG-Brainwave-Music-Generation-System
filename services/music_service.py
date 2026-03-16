"""
Music service: high-level helpers for music generation and playback.
"""

import os
from typing import List, Optional

from algorithms.feature_extractor import SWSFeatures
from algorithms.music_mapper import NoteEvent, map_features_to_notes
from algorithms.midi_generator import generate_midi
from algorithms.audio_synthesizer import midi_to_wav, synthesise_from_notes
from algorithms.sws_detector import SlowWaveEvent
from utils.logger import logger


class MusicService:
    """Convenience wrapper around the music generation pipeline."""

    @staticmethod
    def generate(
        features: SWSFeatures,
        events: List[SlowWaveEvent],
        output_dir: str,
    ) -> tuple[str, str]:
        """Generate MIDI and WAV files from SWS features.

        Returns
        -------
        (midi_path, wav_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        notes = map_features_to_notes(features, events)

        midi_path = os.path.join(output_dir, "brainwave_music.mid")
        generate_midi(notes, midi_path)

        wav_path = os.path.join(output_dir, "brainwave_music.wav")
        try:
            midi_to_wav(midi_path, wav_path)
        except Exception:
            synthesise_from_notes(notes, wav_path)

        logger.info("Music generated in %s", output_dir)
        return midi_path, wav_path
