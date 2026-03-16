from pathlib import Path
from typing import Iterable

import numpy as np

from eeg_music_system.algorithms.feature_extractor import EEGFeatures

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional runtime dependency
    pretty_midi = None


class MusicMapper:
    SCALE = [60, 62, 64, 65, 67, 69, 71, 72]  # C major

    def map_to_notes(self, features: EEGFeatures, max_notes: int = 64) -> Iterable[tuple]:
        if len(features.amplitude) == 0:
            return []
        idx = np.linspace(0, len(features.amplitude) - 1, max_notes).astype(int)
        notes = []
        amp = features.amplitude[idx]
        power = features.power[idx]
        rhythm = np.clip(features.rhythm[idx], 0.25, 2.0)

        amp_norm = (amp - amp.min()) / (np.ptp(amp) + 1e-6)
        power_norm = (power - power.min()) / (np.ptp(power) + 1e-6)
        for i in range(len(idx)):
            pitch = self.SCALE[int(amp_norm[i] * (len(self.SCALE) - 1))]
            velocity = int(40 + power_norm[i] * 70)
            duration = float(np.clip(rhythm[i] * 8, 0.25, 1.5))
            notes.append((pitch, velocity, duration))
        return notes

    def generate_midi(self, features: EEGFeatures, out_path: Path, tempo: int = 60) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        notes = self.map_to_notes(features)
        if pretty_midi is None:
            try:
                from midiutil import MIDIFile

                midi_file = MIDIFile(1)
                midi_file.addTempo(0, 0, tempo)
                start = 0.0
                for pitch, velocity, duration in notes:
                    midi_file.addNote(0, 0, pitch, start, duration, velocity)
                    start += duration
                with out_path.open("wb") as stream:
                    midi_file.writeFile(stream)
            except Exception:
                out_path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60")
            return out_path

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=88)  # ambient pad
        current_time = 0.0
        for pitch, velocity, duration in notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_time,
                    end=current_time + duration,
                )
            )
            current_time += duration
        midi.instruments.append(instrument)
        midi.write(str(out_path))
        return out_path

    def midi_to_wav(self, midi_path: Path, wav_path: Path) -> Path:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if pretty_midi is not None:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            audio = pm.synthesize(fs=44100)
            from scipy.io.wavfile import write

            write(str(wav_path), 44100, np.int16(audio * 32767))
            return wav_path

        # Fallback: create a silent wav when synthesis backend is unavailable.
        from scipy.io.wavfile import write

        write(str(wav_path), 44100, np.zeros(44100, dtype=np.int16))
        return wav_path
