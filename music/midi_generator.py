import os
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

from utils.config import MIDI_OUTPUT_DIR, MUSIC_MAPPING_CONFIG


class IndividualizedMIDIGenerator:
    
    def __init__(self, bpm: int = 60, instrument_program: int = 0):
        self.bpm = bpm
        self.beats_per_bar = 4
        self.bars_per_loop = 4
        self.instrument_program = instrument_program
    
    def generate(self, features: List[Dict[str, Any]], 
                 output_path: str = None,
                 target_duration_minutes: float = 30) -> str:
        if output_path is None:
            output_path = os.path.join(MIDI_OUTPUT_DIR, 'brain_music.mid')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if PRETTY_MIDI_AVAILABLE:
            return self._generate_pretty_midi(features, output_path, target_duration_minutes)
        elif MIDIUTIL_AVAILABLE:
            return self._generate_midiutil(features, output_path, target_duration_minutes)
        else:
            raise ImportError("No MIDI library available")
    
    def _generate_pretty_midi(self, features: List[Dict[str, Any]], 
                               output_path: str,
                               target_duration_minutes: float) -> str:
        pm = pretty_midi.PrettyMIDI()
        
        main_instrument = pretty_midi.Instrument(program=self.instrument_program)
        
        pad_instrument = pretty_midi.Instrument(program=88)
        
        if not features:
            self._add_default_notes(main_instrument, pad_instrument)
        else:
            for feature in features:
                pitch = int(feature.get('pitch', 60))
                velocity = int(feature.get('velocity', 60))
                start_time = float(feature.get('start_time', 0))
                duration = float(feature.get('duration', 4.0))
                
                pitch = max(36, min(96, pitch))
                velocity = max(40, min(80, velocity))
                start_time = max(0.0, start_time)
                duration = max(0.5, min(16.0, duration))
                
                main_note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=start_time + duration
                )
                main_instrument.notes.append(main_note)
                
                pad_note = pretty_midi.Note(
                    velocity=max(30, velocity - 25),
                    pitch=pitch,
                    start=start_time,
                    end=start_time + duration * 1.5
                )
                pad_instrument.notes.append(pad_note)
        
        pm.instruments.append(main_instrument)
        pm.instruments.append(pad_instrument)
        
        current_duration = pm.get_end_time()
        target_duration = target_duration_minutes * 60
        
        if current_duration > 0 and current_duration < target_duration:
            pm = self._extend_midi(pm, target_duration)
        
        pm.write(output_path)
        return output_path
    
    def _generate_midiutil(self, features: List[Dict[str, Any]],
                            output_path: str,
                            target_duration_minutes: float) -> str:
        midi = MIDIFile(2)
        
        midi.addTempo(0, 0, self.bpm)
        midi.addTempo(1, 0, self.bpm)
        
        midi.addProgramChange(0, 0, 0, self.instrument_program)
        midi.addProgramChange(1, 0, 0, 88)
        
        beat_duration = 60.0 / self.bpm
        
        if not features:
            midi.addNote(0, 0, 60, 0, 4, 60)
            midi.addNote(1, 0, 60, 0, 6, 35)
        else:
            for feature in features:
                pitch = int(feature.get('pitch', 60))
                velocity = int(feature.get('velocity', 60))
                start_time = float(feature.get('start_time', 0))
                duration = float(feature.get('duration', 4.0))
                
                pitch = max(36, min(96, pitch))
                velocity = max(40, min(80, velocity))
                
                start_beat = start_time / beat_duration
                duration_beats = duration / beat_duration
                
                midi.addNote(0, 0, pitch, start_beat, duration_beats, velocity)
                midi.addNote(1, 0, pitch, start_beat, duration_beats * 1.5, max(30, velocity - 25))
        
        with open(output_path, 'wb') as f:
            midi.writeFile(f)
        
        return output_path
    
    def _add_default_notes(self, main_inst, pad_inst):
        for i in range(8):
            start = i * 4.0
            main_inst.notes.append(pretty_midi.Note(
                velocity=50, pitch=60, start=start, end=start + 4.0
            ))
            pad_inst.notes.append(pretty_midi.Note(
                velocity=25, pitch=60, start=start, end=start + 6.0
            ))
    
    def _extend_midi(self, pm: 'pretty_midi.PrettyMIDI', 
                     target_duration: float) -> 'pretty_midi.PrettyMIDI':
        current_duration = pm.get_end_time()
        
        if current_duration <= 0:
            return pm
        
        n_repeats = int(np.ceil(target_duration / current_duration))
        
        for instrument in pm.instruments:
            original_notes = instrument.notes.copy()
            
            for i in range(1, n_repeats):
                offset = i * current_duration
                for note in original_notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + offset,
                        end=note.end + offset
                    )
                    instrument.notes.append(new_note)
            
            instrument.notes.sort(key=lambda x: x.start)
        
        return pm


class MIDIGenerator:
    
    def __init__(self, output_dir: str = None, bpm: int = 60, instrument_program: int = 0):
        self.output_dir = output_dir or MIDI_OUTPUT_DIR
        self.bpm = bpm
        self.instrument_program = instrument_program
        self.midi_path = None
        self.individual_generator = IndividualizedMIDIGenerator(bpm, instrument_program)
    
    def generate(self, features: List[Dict[str, Any]], 
                 filename: str = 'brain_music.mid',
                 target_duration_minutes: float = 30,
                 output_dir: str = None) -> str:
        if output_dir:
            self.output_dir = output_dir
        output_path = os.path.join(self.output_dir, filename)
        self.midi_path = self.individual_generator.generate(
            features, output_path, target_duration_minutes
        )
        return self.midi_path
    
    def set_bpm(self, bpm: int):
        self.bpm = bpm
        self.individual_generator.bpm = bpm
    
    def set_instrument(self, program: int):
        self.instrument_program = program
        self.individual_generator.instrument_program = program
    
    def get_midi_info(self) -> Dict[str, Any]:
        if self.midi_path is None or not os.path.exists(self.midi_path):
            return {}
        
        if PRETTY_MIDI_AVAILABLE:
            pm = pretty_midi.PrettyMIDI(self.midi_path)
            return {
                'duration': pm.get_end_time(),
                'tempo': self.bpm,
                'instrument_count': len(pm.instruments),
                'note_count': sum(len(inst.notes) for inst in pm.instruments)
            }
        
        return {'path': self.midi_path, 'tempo': self.bpm}
