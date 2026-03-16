"""
MIDI file generation from NoteEvent objects using pretty_midi.
"""

from typing import List

import pretty_midi

from algorithms.music_mapper import NoteEvent, TEMPO_BPM
from utils.logger import logger


def generate_midi(notes: List[NoteEvent], output_path: str, tempo: int = TEMPO_BPM) -> str:
    """Create a MIDI file from a list of NoteEvent objects.

    Parameters
    ----------
    notes : list of NoteEvent
    output_path : str
        Destination .mid file path.
    tempo : int
        BPM (default 60).

    Returns
    -------
    output_path : str
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Piano instrument (program 0)
    instrument = pretty_midi.Instrument(program=0, name="Brainwave Piano")

    for n in notes:
        note = pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start_time,
            end=n.start_time + n.duration,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_path)
    logger.info("MIDI written: %s (%d notes, %d BPM)", output_path, len(notes), tempo)
    return output_path
