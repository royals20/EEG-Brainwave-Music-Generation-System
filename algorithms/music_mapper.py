"""
Neuro-inspired EEG-to-music mapping model.

Mapping rules:
  1. delta_power       → pitch       (high delta → lower pitch)
  2. slow_wave_amplitude → velocity  (amplitude_norm × 127)
  3. slow_wave_frequency → note dur  (1 / frequency)
  4. slow_wave_density   → note density (notes per bar)

Scale: C-major pentatonic (C D E G A)
Tempo: 60 BPM, 4/4 time
Style: ambient sleep music
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from algorithms.feature_extractor import SWSFeatures
from algorithms.sws_detector import SlowWaveEvent
from utils.logger import logger

# C-major pentatonic across several octaves (MIDI note numbers)
_PENTATONIC = []
for octave in range(3, 7):  # C3 – C6
    base = 12 * (octave + 1)  # C of octave in MIDI
    _PENTATONIC.extend([base, base + 2, base + 4, base + 7, base + 9])
_PENTATONIC = sorted(_PENTATONIC)

TEMPO_BPM = 60
BEAT_DURATION = 60.0 / TEMPO_BPM  # 1 second per beat


@dataclass
class NoteEvent:
    """A single note to be rendered in MIDI."""
    pitch: int        # MIDI note number
    start_time: float # seconds
    duration: float   # seconds
    velocity: int     # 0 – 127


def map_features_to_notes(
    features: SWSFeatures,
    events: List[SlowWaveEvent],
) -> List[NoteEvent]:
    """Map slow-wave events and features to musical note events.

    Each detected slow wave produces one note. Global features (delta_power,
    density) shape the overall pitch range and spacing.
    """
    if not events:
        logger.warning("No slow-wave events to map.")
        return []

    notes: List[NoteEvent] = []
    n_pitches = len(_PENTATONIC)

    # Global pitch bias: higher delta_power → lower register
    pitch_bias = int((1.0 - features.delta_power_norm) * (n_pitches - 1))

    cumulative_time = 0.0

    for ev in events:
        # 1. Pitch: combine global bias with per-event amplitude variation
        amp_norm = np.clip(ev.amplitude / 300.0, 0.0, 1.0)
        idx = int(pitch_bias + (1.0 - amp_norm) * 8)
        idx = np.clip(idx, 0, n_pitches - 1)
        pitch = _PENTATONIC[idx]

        # 2. Velocity from amplitude
        velocity = int(np.clip(amp_norm * 127, 30, 127))

        # 3. Duration from slow-wave frequency
        duration = np.clip(1.0 / ev.frequency if ev.frequency > 0 else 1.0, 0.3, 4.0)

        # 4. Start time: sequential with slight overlap for legato feel
        start_time = cumulative_time
        cumulative_time += duration * 0.8  # 80 % of duration → slight overlap

        notes.append(NoteEvent(
            pitch=pitch,
            start_time=round(start_time, 3),
            duration=round(duration, 3),
            velocity=velocity,
        ))

    logger.info("Mapped %d slow-wave events to %d notes.", len(events), len(notes))
    return notes
