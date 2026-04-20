import os
from typing import Any, Dict, List, Optional, Tuple

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

from music.mapper import ROLE_INSTRUMENT_NAMES, ROLE_INSTRUMENT_PROGRAMS, ROLE_TRACK_ORDER
from utils.config import MIDI_OUTPUT_DIR


class IndividualizedMIDIGenerator:

    def __init__(self, bpm: int = 60, instrument_program: Optional[int] = None):
        self.bpm = bpm
        self.instrument_program = instrument_program

    def _infer_instrument_program(self, features: List[Dict[str, Any]]) -> Optional[int]:
        for feature in features:
            program = feature.get('instrument')
            if program is not None:
                return int(program)
        return None

    def _resolve_main_instrument_program(self, features: List[Dict[str, Any]]) -> int:
        if self.instrument_program is not None:
            return int(self.instrument_program)

        inferred_program = self._infer_instrument_program(features)
        if inferred_program is not None:
            return inferred_program

        return 0

    def _default_features(self) -> List[Dict[str, Any]]:
        return [
            {
                'pitch': 60 + (index % 4) * 2,
                'velocity': 58,
                'start_time': float(index * 2.0),
                'duration': 1.5,
                'role': 'motif',
                'track': 'motif',
                'instrument': ROLE_INSTRUMENT_PROGRAMS['motif'],
            }
            for index in range(8)
        ]

    def _sanitize_feature(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        pitch = max(24, min(108, int(feature.get('pitch', 60))))
        velocity = max(30, min(120, int(feature.get('velocity', 60))))
        start_time = max(0.0, float(feature.get('start_time', 0.0)))
        duration = max(0.05, min(60.0, float(feature.get('duration', 1.0))))
        role = str(feature.get('role') or feature.get('track') or 'main')
        program = feature.get('instrument')
        if program is None:
            program = ROLE_INSTRUMENT_PROGRAMS.get(role, self._resolve_main_instrument_program([feature]))
        return {
            'pitch': pitch,
            'velocity': velocity,
            'start': start_time,
            'end': start_time + duration,
            'role': role,
            'track': str(feature.get('track') or role),
            'program': int(program),
            'instrument_name': str(feature.get('instrument_name') or ROLE_INSTRUMENT_NAMES.get(role, 'Main')),
        }

    def _track_sort_key(self, role: str) -> Tuple[int, str]:
        if role in ROLE_TRACK_ORDER:
            return ROLE_TRACK_ORDER.index(role), role
        return len(ROLE_TRACK_ORDER), role

    def _scale_events(
        self,
        track_events: List[Dict[str, Any]],
        target_duration_minutes: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if target_duration_minutes is None:
            return [event.copy() for event in track_events]

        target_duration_seconds = max(1.0, float(target_duration_minutes) * 60.0)
        source_duration = max((event['end'] for event in track_events), default=0.0)
        if source_duration <= 0:
            return [event.copy() for event in track_events]

        scale = target_duration_seconds / source_duration
        scaled = []
        for event in track_events:
            start = float(event['start']) * scale
            end = min(float(event['end']) * scale, target_duration_seconds)
            if end > start:
                scaled.append({**event, 'start': start, 'end': end})
        return scaled

    def _build_track_events(
        self,
        features: List[Dict[str, Any]],
        target_duration_minutes: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        source_features = features or self._default_features()

        grouped: Dict[str, Dict[str, Any]] = {}
        for feature in source_features:
            event = self._sanitize_feature(feature)
            role = event['role']
            if role not in grouped:
                grouped[role] = {
                    'role': role,
                    'program': event['program'],
                    'instrument_name': event['instrument_name'],
                    'events': [],
                }
            grouped[role]['events'].append(event)

        track_groups = []
        for role, group in grouped.items():
            ordered_events = sorted(group['events'], key=lambda event: (event['start'], event['pitch']))
            track_groups.append(
                {
                    'role': role,
                    'program': group['program'],
                    'instrument_name': group['instrument_name'],
                    'events': self._scale_events(ordered_events, target_duration_minutes=target_duration_minutes),
                }
            )

        track_groups.sort(key=lambda group: self._track_sort_key(group['role']))
        return track_groups

    def _build_note_events(
        self,
        features: List[Dict[str, Any]],
        target_duration_minutes: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        track_groups = self._build_track_events(features, target_duration_minutes=target_duration_minutes)
        if not track_groups:
            return [], []

        primary_events = track_groups[0]['events']
        secondary_events = []
        for group in track_groups[1:]:
            secondary_events.extend(group['events'])
        return primary_events, secondary_events

    def generate(
        self,
        features: List[Dict[str, Any]],
        output_path: str = None,
        target_duration_minutes: Optional[float] = None,
    ) -> str:
        if output_path is None:
            output_path = os.path.join(MIDI_OUTPUT_DIR, 'brain_music.mid')

        if target_duration_minutes is not None and target_duration_minutes <= 0:
            target_duration_minutes = None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if PRETTY_MIDI_AVAILABLE:
            return self._generate_pretty_midi(features, output_path, target_duration_minutes)
        if MIDIUTIL_AVAILABLE:
            return self._generate_midiutil(features, output_path, target_duration_minutes)
        raise ImportError('No MIDI library available')

    def _generate_pretty_midi(
        self,
        features: List[Dict[str, Any]],
        output_path: str,
        target_duration_minutes: Optional[float],
    ) -> str:
        pm = pretty_midi.PrettyMIDI()
        track_groups = self._build_track_events(features, target_duration_minutes=target_duration_minutes)

        for group in track_groups:
            instrument = pretty_midi.Instrument(
                program=int(group['program']),
                name=str(group['role']),
            )
            for event in group['events']:
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=int(event['velocity']),
                        pitch=int(event['pitch']),
                        start=float(event['start']),
                        end=float(event['end']),
                    )
                )
            pm.instruments.append(instrument)

        pm.write(output_path)
        return output_path

    def _generate_midiutil(
        self,
        features: List[Dict[str, Any]],
        output_path: str,
        target_duration_minutes: Optional[float],
    ) -> str:
        track_groups = self._build_track_events(features, target_duration_minutes=target_duration_minutes)
        midi = MIDIFile(max(1, len(track_groups)))
        beat_duration = 60.0 / self.bpm

        for track_index, group in enumerate(track_groups):
            midi.addTempo(track_index, 0, self.bpm)
            midi.addTrackName(track_index, 0, group['role'])
            midi.addProgramChange(track_index, track_index, 0, int(group['program']))
            for event in group['events']:
                midi.addNote(
                    track_index,
                    track_index,
                    int(event['pitch']),
                    float(event['start']) / beat_duration,
                    (float(event['end']) - float(event['start'])) / beat_duration,
                    int(event['velocity']),
                )

        with open(output_path, 'wb') as output_file:
            midi.writeFile(output_file)

        return output_path


class MIDIGenerator:

    def __init__(self, output_dir: str = None, bpm: int = 60, instrument_program: Optional[int] = None):
        self.output_dir = output_dir or MIDI_OUTPUT_DIR
        self.bpm = bpm
        self.instrument_program = instrument_program
        self.midi_path = None
        self.individual_generator = IndividualizedMIDIGenerator(bpm, instrument_program)

    def generate(
        self,
        features: List[Dict[str, Any]],
        filename: str = 'brain_music.mid',
        target_duration_minutes: float = None,
        output_dir: str = None,
    ) -> str:
        if output_dir:
            self.output_dir = output_dir

        self.instrument_program = self.individual_generator._resolve_main_instrument_program(features)
        self.individual_generator.instrument_program = self.instrument_program

        output_path = os.path.join(self.output_dir, filename)
        self.midi_path = self.individual_generator.generate(
            features,
            output_path=output_path,
            target_duration_minutes=target_duration_minutes,
        )
        return self.midi_path

    def set_bpm(self, bpm: int):
        self.bpm = bpm
        self.individual_generator.bpm = bpm

    def set_instrument(self, program: Optional[int]):
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
                'note_count': sum(len(instrument.notes) for instrument in pm.instruments),
            }

        return {'path': self.midi_path, 'tempo': self.bpm}
