from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.config import MUSIC_MAPPING_CONFIG


PENTATONIC_SCALE = [60, 62, 64, 67, 69]
MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71]
MINOR_SCALE = [60, 62, 63, 65, 67, 68, 70]

SCALES = {
    'low': PENTATONIC_SCALE,
    'medium': MAJOR_SCALE,
    'high': MINOR_SCALE,
}

SCALE_NAMES = {
    'low': 'Pentatonic',
    'medium': 'Major',
    'high': 'Minor',
}

INSTRUMENT_PROGRAMS = {
    'low': 88,
    'medium': 0,
    'high': 49,
}

INSTRUMENT_NAMES = {
    'low': 'Ambient Pad',
    'medium': 'Soft Piano',
    'high': 'Warm Strings',
}

MODE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'pentatonic': [0, 2, 4, 7, 9],
}

ROLE_TRACK_ORDER = ('pad', 'bass', 'motif')
ROLE_INSTRUMENT_PROGRAMS = {
    'pad': 49,
    'bass': 42,
    'motif': 0,
}

ROLE_INSTRUMENT_NAMES = {
    'pad': 'Warm Strings',
    'bass': 'Soft Cello',
    'motif': 'Soft Piano',
}

PROGRESSION_POOLS = {
    'major': {
        'calm': [
            {'degree': 0, 'quality': 'maj7', 'label': 'Imaj7'},
            {'degree': 5, 'quality': 'min7', 'label': 'vi7'},
            {'degree': 3, 'quality': 'maj7', 'label': 'IVmaj7'},
            {'degree': 4, 'quality': 'sus2', 'label': 'Vsus2'},
        ],
        'neutral': [
            {'degree': 0, 'quality': 'maj', 'label': 'I'},
            {'degree': 4, 'quality': 'maj', 'label': 'V'},
            {'degree': 5, 'quality': 'min', 'label': 'vi'},
            {'degree': 3, 'quality': 'maj', 'label': 'IV'},
        ],
        'active': [
            {'degree': 1, 'quality': 'min7', 'label': 'ii7'},
            {'degree': 4, 'quality': 'maj', 'label': 'V'},
            {'degree': 0, 'quality': 'maj7', 'label': 'Imaj7'},
            {'degree': 5, 'quality': 'min7', 'label': 'vi7'},
        ],
    },
    'dorian': {
        'calm': [
            {'degree': 0, 'quality': 'min7', 'label': 'i7'},
            {'degree': 6, 'quality': 'maj7', 'label': 'VIImaj7'},
            {'degree': 3, 'quality': 'maj7', 'label': 'IVmaj7'},
            {'degree': 4, 'quality': 'sus2', 'label': 'vsus2'},
        ],
        'neutral': [
            {'degree': 0, 'quality': 'min', 'label': 'i'},
            {'degree': 4, 'quality': 'min', 'label': 'v'},
            {'degree': 6, 'quality': 'maj', 'label': 'VII'},
            {'degree': 3, 'quality': 'maj', 'label': 'IV'},
        ],
        'active': [
            {'degree': 1, 'quality': 'min7', 'label': 'ii7'},
            {'degree': 4, 'quality': 'min', 'label': 'v'},
            {'degree': 0, 'quality': 'min7', 'label': 'i7'},
            {'degree': 6, 'quality': 'maj7', 'label': 'VIImaj7'},
        ],
    },
}

CHORD_QUALITY_INTERVALS = {
    'maj': [0, 4, 7, 12],
    'min': [0, 3, 7, 12],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'sus2': [0, 2, 7, 10],
}


def get_scale_for_delta_power(delta_level: str) -> List[int]:
    return SCALES.get(delta_level, MAJOR_SCALE)


def get_tempo_for_frequency(frequency: float, base_tempo: int = 60) -> int:
    if frequency < 0.8:
        tempo = base_tempo - 10
    elif frequency > 1.2:
        tempo = base_tempo + 10
    else:
        tempo = base_tempo
    return max(40, min(120, tempo))


def get_note_density_for_slow_wave_density(slow_wave_density: float) -> float:
    if slow_wave_density < 3:
        return 2.0
    if slow_wave_density > 6:
        return 0.5
    return 1.0


def get_instrument_for_amplitude(amplitude_level: str) -> int:
    return INSTRUMENT_PROGRAMS.get(amplitude_level, 0)


def get_instrument_name_for_amplitude(amplitude_level: str) -> str:
    return INSTRUMENT_NAMES.get(amplitude_level, 'Soft Piano')


def map_frequency_to_pitch(frequency: float, base_pitch: int, pitch_range: int, max_frequency: float) -> int:
    if max_frequency <= 0:
        max_frequency = 4.0
    normalized_frequency = max(0.0, min(frequency / max_frequency, 1.0))
    return int(round(base_pitch + normalized_frequency * pitch_range))


def quantize_to_scale(pitch: int, scale: Sequence[int]) -> int:
    octave = pitch // 12
    note_in_octave = pitch % 12
    scale_notes_mod = [note % 12 for note in scale]
    closest_note = min(
        scale_notes_mod,
        key=lambda note: min(abs(note - note_in_octave), 12 - abs(note - note_in_octave)),
    )
    return octave * 12 + closest_note


def map_amplitude_to_velocity(amplitude: float, min_amp: float, max_amp: float) -> int:
    if max_amp == min_amp:
        ratio = 0.5
    else:
        ratio = (amplitude - min_amp) / (max_amp - min_amp)

    velocity = int(40 + ratio * 40)
    return max(40, min(80, velocity))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def get_scale_pitch_classes(base_pitch: int, mode: str) -> List[int]:
    intervals = MODE_INTERVALS.get(mode, MODE_INTERVALS['major'])
    tonic_pc = base_pitch % 12
    return sorted({(tonic_pc + interval) % 12 for interval in intervals})


def nearest_pitch_for_pitch_class(
    pitch_class: int,
    target_pitch: float,
    lower: int,
    upper: int,
) -> int:
    candidates = [pitch for pitch in range(lower, upper + 1) if pitch % 12 == pitch_class % 12]
    if not candidates:
        return int(clamp(target_pitch, lower, upper))
    return min(candidates, key=lambda pitch: (abs(pitch - target_pitch), pitch))


def deterministic_humanize_seconds(note_index: int, seed: int, humanization_ms: float, allow_shift: bool) -> float:
    if not allow_shift or humanization_ms <= 0:
        return 0.0

    spread = float(humanization_ms) / 1000.0
    phase = (note_index + 1) * (seed + 3) * 0.61803398875
    return float(np.sin(phase) * spread * 0.5)


def build_chord_definition(
    tonic_pitch: int,
    mode: str,
    chord_spec: Dict[str, Any],
    harmony_richness: int,
) -> Dict[str, Any]:
    mode_intervals = MODE_INTERVALS.get(mode, MODE_INTERVALS['major'])
    degree_index = int(chord_spec['degree']) % len(mode_intervals)
    root_pc = (tonic_pitch + mode_intervals[degree_index]) % 12
    intervals = list(CHORD_QUALITY_INTERVALS.get(chord_spec['quality'], CHORD_QUALITY_INTERVALS['maj']))
    intervals = intervals[:3] if harmony_richness <= 2 else intervals[:4]
    pitch_classes = [int((root_pc + interval) % 12) for interval in intervals]
    return {
        'root_pc': int(root_pc),
        'quality': chord_spec['quality'],
        'label': chord_spec['label'],
        'degree': degree_index,
        'intervals': intervals,
        'pitch_classes': pitch_classes,
    }


def build_close_voicing(
    root_pc: int,
    intervals: Sequence[int],
    previous_voicing: Optional[Sequence[int]],
    lower: int,
    upper: int,
    center: float,
) -> List[int]:
    best_score = None
    best_voicing: Optional[List[int]] = None
    voice_count = len(intervals)
    root_candidates = [pitch for pitch in range(lower - 12, upper + 13) if pitch % 12 == root_pc % 12]

    for root_pitch in root_candidates:
        for inversion in range(voice_count):
            base_voicing = []
            for index, interval in enumerate(intervals):
                shifted_interval = interval + (12 if index < inversion else 0)
                base_voicing.append(root_pitch + shifted_interval)
            base_voicing = sorted(base_voicing)

            for octave_shift in (-12, 0, 12):
                voicing = [pitch + octave_shift for pitch in base_voicing]
                average_pitch = float(np.mean(voicing))
                while average_pitch < center - 6:
                    voicing = [pitch + 12 for pitch in voicing]
                    average_pitch = float(np.mean(voicing))
                while average_pitch > center + 6:
                    voicing = [pitch - 12 for pitch in voicing]
                    average_pitch = float(np.mean(voicing))

                if min(voicing) < lower or max(voicing) > upper:
                    continue

                if previous_voicing and len(previous_voicing) == len(voicing):
                    pitch_diffs = [abs(current - previous) for current, previous in zip(voicing, previous_voicing)]
                    leap_penalty = sum(max(0, diff - 5) * 6 for diff in pitch_diffs)
                    score = float(sum(pitch_diffs) + leap_penalty + abs(np.mean(voicing) - np.mean(previous_voicing)))
                else:
                    score = float(abs(np.mean(voicing) - center))

                if best_score is None or score < best_score:
                    best_score = score
                    best_voicing = voicing

    if best_voicing is not None:
        return best_voicing

    return [nearest_pitch_for_pitch_class((root_pc + interval) % 12, center + interval, lower, upper) for interval in intervals]


def density_bucket(value: float) -> str:
    if value < 0.35:
        return 'low'
    if value > 0.68:
        return 'high'
    return 'medium'


class IndividualizedMusicMapper:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = MUSIC_MAPPING_CONFIG.copy()
        if config:
            self.config.update(config)

        self.scale = MAJOR_SCALE
        self.tempo = self.config.get('base_tempo', 60)
        self.note_density = 1.0
        self.instrument_program = 0
        self.mapped_features: List[Dict[str, Any]] = []
        self.individual_params: Dict[str, Any] = {}
        self.active_scale_label = 'Major'
        self.track_programs = ROLE_INSTRUMENT_PROGRAMS.copy()
        self.role_note_counts = {role: 0 for role in ROLE_TRACK_ORDER}
        self.velocity_envelope: List[int] = []

    def configure_from_eeg_stats(
        self,
        stats: Dict[str, Any],
        slow_wave_density: float = 3.0,
        user_overrides: Dict[str, Any] = None,
    ):
        if user_overrides:
            for key, value in user_overrides.items():
                if key in self.config and value is not None:
                    self.config[key] = value

        delta_level = stats.get('delta_power_level', 'medium')
        amplitude_level = stats.get('amplitude_level', 'medium')
        frequency = stats.get('avg_slow_wave_frequency', 1.0)

        self.scale = get_scale_for_delta_power(delta_level)
        self.tempo = get_tempo_for_frequency(
            frequency,
            self.config.get('base_tempo', MUSIC_MAPPING_CONFIG.get('base_tempo', 60)),
        )
        self.note_density = get_note_density_for_slow_wave_density(slow_wave_density)
        self.instrument_program = get_instrument_for_amplitude(amplitude_level)
        self.active_scale_label = SCALE_NAMES.get(delta_level, 'Major')
        self.track_programs = ROLE_INSTRUMENT_PROGRAMS.copy()

        self.individual_params = {
            'scale_name': self.active_scale_label,
            'tempo': self.tempo,
            'base_tempo': self.config.get('base_tempo', 60),
            'base_pitch': self.config.get('base_pitch', 60),
            'pitch_range': self.config.get('pitch_range', 12),
            'note_density': self.note_density,
            'instrument_name': 'Healing Ensemble',
            'instrument_program': self.instrument_program,
            'track_programs': self.track_programs.copy(),
            'delta_level': delta_level,
            'amplitude_level': amplitude_level,
            'frequency': frequency,
            'mode': self.config.get('mode', 'healing_ambient'),
        }

    def map_features(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = []
        if not eeg_features:
            return self.mapped_features

        amplitudes = [feature.get('mean_amplitude', 0) for feature in eeg_features]
        delta_powers = [feature.get('delta_power', 0) for feature in eeg_features]

        min_amp = min(amplitudes) if amplitudes else 0
        max_amp = max(amplitudes) if amplitudes else 200

        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4
        note_interval = bar_duration / (4 / self.note_density)
        current_time = 0.0
        average_delta = float(np.mean(delta_powers)) if delta_powers else 0.0

        for features in eeg_features:
            delta_power = features.get('delta_power', 0)
            amplitude = features.get('mean_amplitude', 0)
            frequency = features.get('dominant_frequency', 1.0)

            base_pitch = map_frequency_to_pitch(
                frequency,
                self.config.get('base_pitch', 60),
                self.config.get('pitch_range', 12),
                self.config.get('max_frequency', 4.0),
            )
            pitch = quantize_to_scale(base_pitch, self.scale)
            velocity = map_amplitude_to_velocity(amplitude, min_amp, max_amp)
            note_duration = beat_duration * 4

            self.mapped_features.append(
                {
                    'pitch': pitch,
                    'velocity': velocity,
                    'start_time': current_time,
                    'duration': note_duration,
                    'delta_power': delta_power,
                    'amplitude': amplitude,
                    'instrument': self.instrument_program,
                }
            )

            if delta_power > average_delta:
                harmony_pitch = quantize_to_scale(pitch + 4, self.scale)
                self.mapped_features.append(
                    {
                        'pitch': harmony_pitch,
                        'velocity': max(40, velocity - 15),
                        'start_time': current_time,
                        'duration': note_duration * 0.75,
                        'is_harmony': True,
                        'instrument': self.instrument_program,
                    }
                )

            current_time += note_interval

        return self.mapped_features

    def generate_ambient_structure(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = []
        if not eeg_features:
            return self.mapped_features

        amplitudes = [feature.get('mean_amplitude', 0) for feature in eeg_features]
        delta_powers = [feature.get('delta_power', 0) for feature in eeg_features]

        min_amp = min(amplitudes) if amplitudes else 0
        max_amp = max(amplitudes) if amplitudes else 200
        average_delta = float(np.mean(delta_powers)) if delta_powers else 0.0

        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4
        current_time = 0.0
        pulse_count = max(0, int(round(self.note_density * 2.0)) - 1)

        for features in eeg_features:
            delta_power = features.get('delta_power', 0)
            amplitude = features.get('mean_amplitude', 0)
            frequency = features.get('dominant_frequency', 1.0)
            slow_wave_ratio = float(features.get('slow_wave_ratio', 0) or 0)
            density_factor = 1.0 / max(self.note_density, 0.5)
            segment_duration = max(
                beat_duration * 3.0,
                bar_duration * (1.0 + 0.4 * density_factor + 0.6 * slow_wave_ratio),
            )
            segment_step = max(beat_duration * 1.5, segment_duration * 0.55)

            base_pitch = map_frequency_to_pitch(
                frequency,
                self.config.get('base_pitch', 60),
                self.config.get('pitch_range', 12),
                self.config.get('max_frequency', 4.0),
            )
            root_pitch = quantize_to_scale(base_pitch, self.scale)
            velocity = map_amplitude_to_velocity(amplitude, min_amp, max_amp)
            note_duration = max(
                beat_duration,
                min(segment_duration, segment_duration * (0.8 + 0.15 * slow_wave_ratio)),
            )

            self.mapped_features.append(
                {
                    'pitch': root_pitch,
                    'velocity': velocity,
                    'start_time': current_time,
                    'duration': note_duration,
                    'instrument': self.instrument_program,
                }
            )

            harmony_step = max(beat_duration * 0.75, segment_duration / max(self.note_density * 2.5, 1.5))

            if delta_power > average_delta * 0.8 or slow_wave_ratio >= 0.2:
                third_start = min(current_time + harmony_step, current_time + max(segment_duration - beat_duration, 0.0))
                third_pitch = quantize_to_scale(root_pitch + 4, self.scale)
                third_duration = max(
                    beat_duration,
                    min(segment_duration - (third_start - current_time), segment_duration * 0.6),
                )
                self.mapped_features.append(
                    {
                        'pitch': third_pitch,
                        'velocity': max(40, velocity - 10),
                        'start_time': third_start,
                        'duration': third_duration,
                        'instrument': self.instrument_program,
                    }
                )

            if delta_power > average_delta * 1.2 or slow_wave_ratio >= 0.35:
                fifth_start = min(
                    current_time + harmony_step * max(1.25, 2.0 / max(self.note_density, 0.5)),
                    current_time + max(segment_duration - beat_duration, 0.0),
                )
                fifth_pitch = quantize_to_scale(root_pitch + 7, self.scale)
                fifth_duration = max(
                    beat_duration,
                    min(segment_duration - (fifth_start - current_time), segment_duration * 0.45),
                )
                self.mapped_features.append(
                    {
                        'pitch': fifth_pitch,
                        'velocity': max(40, velocity - 20),
                        'start_time': fifth_start,
                        'duration': fifth_duration,
                        'instrument': self.instrument_program,
                    }
                )

            if pulse_count > 0:
                pulse_interval = segment_duration / (pulse_count + 1)
                pulse_pitch = quantize_to_scale(root_pitch + 12, self.scale)
                for pulse_index in range(pulse_count):
                    pulse_start = min(
                        current_time + pulse_interval * (pulse_index + 1),
                        current_time + max(segment_duration - beat_duration * 0.5, 0.0),
                    )
                    pulse_duration = max(
                        beat_duration * 0.5,
                        min(segment_duration - (pulse_start - current_time), segment_duration * 0.2),
                    )
                    self.mapped_features.append(
                        {
                            'pitch': pulse_pitch,
                            'velocity': max(35, velocity - 25),
                            'start_time': pulse_start,
                            'duration': pulse_duration,
                            'instrument': self.instrument_program,
                            'is_pulse': True,
                        }
                    )

            current_time += segment_step

        return self.mapped_features

    def _average_feature_slice(
        self,
        eeg_features: Sequence[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> Dict[str, float]:
        if not eeg_features:
            return {}

        selected = []
        midpoints = []
        for feature in eeg_features:
            feature_start = float(feature.get('start_time', 0.0))
            feature_end = float(feature.get('end_time', feature_start + feature.get('duration', 0.0)))
            midpoint = (feature_start + feature_end) / 2.0
            midpoints.append(midpoint)
            if feature_start < end_time and feature_end > start_time:
                selected.append(feature)

        if not selected:
            target_midpoint = (start_time + end_time) / 2.0
            nearest_index = min(range(len(eeg_features)), key=lambda index: abs(midpoints[index] - target_midpoint))
            selected = [eeg_features[nearest_index]]

        summary = {}
        keys = [
            'calmness',
            'density',
            'brightness',
            'dominant_frequency',
            'mean_amplitude',
            'slow_wave_ratio',
            'energy_change',
            'delta_ratio',
        ]
        for key in keys:
            summary[key] = float(np.mean([feature.get(key, 0.0) for feature in selected]))
        return summary

    def _build_bar_descriptors(
        self,
        eeg_features: Sequence[Dict[str, Any]],
        total_bars: int,
        total_duration_seconds: float,
        source_duration_seconds: float,
        settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        descriptors = []
        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4.0
        user_density = (int(settings.get('melody_density', 2)) - 1) / 4.0

        for bar_index in range(total_bars):
            bar_start = bar_index * bar_duration
            bar_end = min(total_duration_seconds, bar_start + bar_duration)
            normalized_start = bar_start / max(total_duration_seconds, 1e-6)
            normalized_end = bar_end / max(total_duration_seconds, 1e-6)
            source_start = normalized_start * source_duration_seconds
            source_end = max(source_start + 1e-6, normalized_end * source_duration_seconds)
            summary = self._average_feature_slice(eeg_features, source_start, source_end)

            combined_density = clamp(0.7 * summary.get('density', 0.5) + 0.3 * user_density, 0.0, 1.0)
            descriptors.append(
                {
                    **summary,
                    'bar_index': bar_index,
                    'start_time': bar_start,
                    'end_time': bar_end,
                    'duration': bar_end - bar_start,
                    'combined_density': combined_density,
                    'density_level': density_bucket(combined_density),
                }
            )

        return descriptors

    def _build_section_descriptors(
        self,
        bar_descriptors: Sequence[Dict[str, Any]],
        section_length: int,
    ) -> List[Dict[str, Any]]:
        sections = []
        for section_start in range(0, len(bar_descriptors), section_length):
            section_bars = list(bar_descriptors[section_start:section_start + section_length])
            calmness = float(np.mean([bar.get('calmness', 0.5) for bar in section_bars]))
            density = float(np.mean([bar.get('combined_density', 0.5) for bar in section_bars]))
            brightness = float(np.mean([bar.get('brightness', 0.5) for bar in section_bars]))

            if calmness >= 0.63 and density <= 0.45:
                family = 'calm'
            elif density >= 0.63 or brightness >= 0.65:
                family = 'active'
            else:
                family = 'neutral'

            mode = 'dorian' if brightness < 0.38 and calmness < 0.58 else 'major'
            motif_scale = 'pentatonic' if family == 'calm' and calmness > 0.70 else mode
            progression = PROGRESSION_POOLS[mode][family]

            sections.append(
                {
                    'section_index': len(sections),
                    'start_bar': section_start,
                    'bars': section_bars,
                    'calmness': calmness,
                    'density': density,
                    'brightness': brightness,
                    'family': family,
                    'mode': mode,
                    'motif_scale': motif_scale,
                    'progression': progression,
                }
            )

        return sections

    def _build_bar_contexts(
        self,
        eeg_features: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], float]:
        source_duration_seconds = max(
            float(feature.get('end_time', feature.get('start_time', 0.0) + feature.get('duration', 0.0)))
            for feature in eeg_features
        )
        source_duration_seconds = max(source_duration_seconds, 1.0)

        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4.0
        total_bars = max(1, int(total_duration_seconds // bar_duration))
        bar_descriptors = self._build_bar_descriptors(
            eeg_features,
            total_bars,
            total_duration_seconds,
            source_duration_seconds,
            settings,
        )

        section_length = 16 if int(settings.get('section_length_bars', 8)) >= 16 else 8
        section_descriptors = self._build_section_descriptors(bar_descriptors, section_length)
        contexts = []
        for section in section_descriptors:
            progression = section['progression']
            for local_bar_index, bar in enumerate(section['bars']):
                chord_spec = progression[local_bar_index % len(progression)]
                chord = build_chord_definition(
                    self.config.get('base_pitch', 60),
                    section['mode'],
                    chord_spec,
                    int(settings.get('harmony_richness', 4)),
                )
                contexts.append(
                    {
                        **bar,
                        'section_index': section['section_index'],
                        'progression_family': section['family'],
                        'mode': section['mode'],
                        'motif_scale': section['motif_scale'],
                        'chord': chord,
                        'chord_label': chord['label'],
                    }
                )

        return contexts, source_duration_seconds

    def _append_note(
        self,
        feature_list: List[Dict[str, Any]],
        *,
        pitch: int,
        velocity: int,
        start_time: float,
        duration: float,
        role: str,
        bar_index: int,
        section_index: int,
        total_duration_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if start_time >= total_duration_seconds or duration <= 0:
            return

        note_end = min(total_duration_seconds, start_time + duration)
        if note_end <= start_time:
            return

        payload = {
            'pitch': int(pitch),
            'velocity': int(clamp(velocity, self.config.get('min_velocity', 45), self.config.get('max_velocity', 88))),
            'start_time': float(max(0.0, start_time)),
            'duration': float(note_end - start_time),
            'role': role,
            'track': role,
            'bar_index': int(bar_index),
            'section_index': int(section_index),
            'instrument': ROLE_INSTRUMENT_PROGRAMS[role],
            'instrument_name': ROLE_INSTRUMENT_NAMES[role],
        }
        if extra:
            payload.update(extra)
        feature_list.append(payload)

    def _build_velocity_envelope(self, bar_contexts: Sequence[Dict[str, Any]], section_length: int) -> List[int]:
        raw_values = []
        for bar in bar_contexts:
            amplitude_component = 0.55 * clamp(bar.get('mean_amplitude', 0.0) / 120.0, 0.0, 1.0)
            calm_component = 0.25 * bar.get('calmness', 0.5)
            density_component = 0.20 * bar.get('combined_density', 0.5)
            raw_values.append(amplitude_component + calm_component + density_component)

        if not raw_values:
            return []

        smoothed = []
        half_window = max(1, section_length // 2)
        for index, value in enumerate(raw_values):
            start = max(0, index - half_window)
            end = min(len(raw_values), index + half_window + 1)
            smoothed.append(float(np.mean(raw_values[start:end])))

        minimum = min(smoothed)
        maximum = max(smoothed)
        velocities = []
        previous = None
        for value in smoothed:
            normalized = 0.5 if np.isclose(maximum, minimum) else (value - minimum) / (maximum - minimum)
            velocity = int(round(
                self.config.get('min_velocity', 45)
                + normalized * (self.config.get('max_velocity', 88) - self.config.get('min_velocity', 45))
            ))
            if previous is not None:
                velocity = int(clamp(velocity, previous - 12, previous + 12))
            velocity = int(clamp(velocity, self.config.get('min_velocity', 45), self.config.get('max_velocity', 88)))
            velocities.append(velocity)
            previous = velocity
        return velocities

    def _build_pad_track(
        self,
        bar_contexts: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
        feature_list: List[Dict[str, Any]],
    ) -> None:
        if not settings.get('enable_pad', True):
            return

        previous_voicing = None
        center = clamp(self.config.get('base_pitch', 60) + 12, 63, 72)
        for context in bar_contexts:
            chord = context['chord']
            voicing = build_close_voicing(
                chord['root_pc'],
                chord['intervals'],
                previous_voicing,
                lower=58,
                upper=82,
                center=center + (context.get('brightness', 0.5) - 0.5) * 6,
            )
            velocity = int(self.velocity_envelope[context['bar_index']])
            for voice_index, pitch in enumerate(voicing):
                self._append_note(
                    feature_list,
                    pitch=pitch,
                    velocity=velocity - voice_index * 3,
                    start_time=context['start_time'],
                    duration=context['duration'],
                    role='pad',
                    bar_index=context['bar_index'],
                    section_index=context['section_index'],
                    total_duration_seconds=total_duration_seconds,
                    extra={
                        'chord_label': chord['label'],
                        'mode': context['mode'],
                        'pitch_class': pitch % 12,
                    },
                )
            previous_voicing = voicing

        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4.0
        tail_duration = total_duration_seconds - len(bar_contexts) * bar_duration
        if tail_duration > 0 and previous_voicing and bar_contexts:
            last_context = bar_contexts[-1]
            for voice_index, pitch in enumerate(previous_voicing):
                self._append_note(
                    feature_list,
                    pitch=pitch,
                    velocity=self.velocity_envelope[last_context['bar_index']] - voice_index * 3,
                    start_time=len(bar_contexts) * bar_duration,
                    duration=tail_duration,
                    role='pad',
                    bar_index=len(bar_contexts),
                    section_index=last_context['section_index'],
                    total_duration_seconds=total_duration_seconds,
                    extra={
                        'chord_label': last_context['chord']['label'],
                        'mode': last_context['mode'],
                        'is_tail': True,
                        'pitch_class': pitch % 12,
                    },
                )

    def _build_bass_track(
        self,
        bar_contexts: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
        feature_list: List[Dict[str, Any]],
    ) -> None:
        if not settings.get('enable_bass', True):
            return

        previous_pitch = None
        seed = int(settings.get('random_seed', 17))
        note_index = 0
        beat_duration = 60.0 / self.tempo

        for context in bar_contexts:
            density_level = context['density_level']
            if density_level == 'low' and (context['bar_index'] % 2) == 1:
                continue

            chord = context['chord']
            use_fifth = density_level != 'low' and (context['bar_index'] + context['section_index']) % 2 == 1
            target_pc = (chord['root_pc'] + (7 if use_fifth else 0)) % 12
            bass_target = previous_pitch if previous_pitch is not None else 43
            bass_pitch = nearest_pitch_for_pitch_class(target_pc, bass_target, 36, 52)
            duration_beats = 2.0 if density_level == 'low' else 1.5 if density_level == 'medium' else 1.0
            offset = deterministic_humanize_seconds(note_index, seed, settings.get('humanization_ms', 20), allow_shift=False)
            self._append_note(
                feature_list,
                pitch=bass_pitch,
                velocity=self.velocity_envelope[context['bar_index']] - 8,
                start_time=context['start_time'] + offset,
                duration=duration_beats * beat_duration,
                role='bass',
                bar_index=context['bar_index'],
                section_index=context['section_index'],
                total_duration_seconds=total_duration_seconds,
                extra={
                    'chord_label': chord['label'],
                    'mode': context['mode'],
                    'bass_choice': 'fifth' if use_fifth else 'root',
                    'pitch_class': bass_pitch % 12,
                },
            )
            previous_pitch = bass_pitch
            note_index += 1

    def _motif_pattern_for_density(self, density_level: str) -> List[Tuple[float, float, bool]]:
        if density_level == 'low':
            return [(0.0, 1.5, True), (2.0, 1.5, True)]
        if density_level == 'high':
            return [(0.0, 0.75, True), (0.5, 0.75, False), (2.0, 0.75, True), (3.0, 0.75, False)]
        return [(0.0, 1.0, True), (1.5, 1.0, False), (3.0, 0.75, False)]

    def _choose_motif_pitch(
        self,
        *,
        scale_mode: str,
        chord_pitch_classes: Sequence[int],
        last_pitch: Optional[int],
        melodic_center: float,
        strong_beat: bool,
        motion_hint: int,
    ) -> int:
        scale_pitch_classes = get_scale_pitch_classes(self.config.get('base_pitch', 60), scale_mode)
        candidate_pitch_classes = chord_pitch_classes if strong_beat else scale_pitch_classes
        candidates = [pitch for pitch in range(60, 85) if pitch % 12 in candidate_pitch_classes]
        if not candidates:
            candidates = [pitch for pitch in range(60, 85) if pitch % 12 in scale_pitch_classes]

        if last_pitch is None:
            return min(candidates, key=lambda pitch: (abs(pitch - melodic_center), pitch))

        preferred_target = clamp(last_pitch + motion_hint, 60, 84)
        filtered_candidates = [pitch for pitch in candidates if abs(pitch - last_pitch) <= 7]
        if not filtered_candidates:
            filtered_candidates = candidates

        return min(
            filtered_candidates,
            key=lambda pitch: (
                abs(pitch - preferred_target),
                abs(pitch - melodic_center),
                abs(pitch - last_pitch),
            ),
        )

    def _compose_phrase(
        self,
        phrase_contexts: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
        phrase_index: int,
        starting_pitch: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        events: List[Dict[str, Any]] = []
        beat_duration = 60.0 / self.tempo
        last_pitch = starting_pitch
        seed = int(settings.get('random_seed', 17))
        humanization_ms = float(settings.get('humanization_ms', 20))

        for local_bar_index, context in enumerate(phrase_contexts):
            melodic_center = clamp(
                self.config.get('base_pitch', 60)
                + 12
                + round(context.get('brightness', 0.5) * 8)
                + round((context.get('calmness', 0.5) - 0.5) * 4),
                60,
                84,
            )
            pattern = self._motif_pattern_for_density(context['density_level'])
            for local_note_index, (beat_offset, beat_length, strong_beat) in enumerate(pattern):
                global_note_index = phrase_index * 100 + local_bar_index * 10 + local_note_index
                motion_cycle = (global_note_index % 5) - 2
                motion_hint = motion_cycle * (1 if strong_beat else 2)
                pitch = self._choose_motif_pitch(
                    scale_mode=context['motif_scale'],
                    chord_pitch_classes=context['chord']['pitch_classes'],
                    last_pitch=last_pitch,
                    melodic_center=melodic_center,
                    strong_beat=strong_beat,
                    motion_hint=motion_hint,
                )
                humanize = deterministic_humanize_seconds(
                    global_note_index,
                    seed,
                    humanization_ms,
                    allow_shift=beat_offset > 0.0,
                )
                start_time = max(context['start_time'], context['start_time'] + beat_offset * beat_duration + humanize)
                velocity = self.velocity_envelope[context['bar_index']] + (4 if strong_beat else -2)
                self._append_note(
                    events,
                    pitch=pitch,
                    velocity=velocity,
                    start_time=start_time,
                    duration=beat_length * beat_duration,
                    role='motif',
                    bar_index=context['bar_index'],
                    section_index=context['section_index'],
                    total_duration_seconds=total_duration_seconds,
                    extra={
                        'chord_label': context['chord']['label'],
                        'mode': context['motif_scale'],
                        'phrase_index': phrase_index,
                        'phrase_bar_offset': local_bar_index,
                        'is_phrase_repeat': False,
                        'pitch_class': pitch % 12,
                    },
                )
                last_pitch = pitch

        return events, last_pitch

    def _repeat_phrase_with_variation(
        self,
        phrase_events: Sequence[Dict[str, Any]],
        repeat_contexts: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        if not phrase_events or not repeat_contexts:
            return [], None

        source_phrase_start = min(event['start_time'] for event in phrase_events)
        repeat_phrase_start = repeat_contexts[0]['start_time']
        max_bar_offset = len(repeat_contexts)
        repeated_events = []
        last_pitch = None

        for event in phrase_events:
            local_bar_offset = int(event.get('phrase_bar_offset', 0))
            if local_bar_offset >= max_bar_offset:
                continue
            target_context = repeat_contexts[local_bar_offset]
            relative_start = event['start_time'] - source_phrase_start
            repeated = event.copy()
            repeated['start_time'] = repeat_phrase_start + relative_start
            repeated['bar_index'] = target_context['bar_index']
            repeated['section_index'] = target_context['section_index']
            repeated['chord_label'] = target_context['chord']['label']
            repeated['mode'] = target_context['motif_scale']
            repeated['is_phrase_repeat'] = True
            if repeated['start_time'] >= total_duration_seconds:
                continue
            repeated_events.append(repeated)
            last_pitch = repeated['pitch']

        if repeated_events:
            final_event = repeated_events[-1]
            scale_pitch_classes = get_scale_pitch_classes(self.config.get('base_pitch', 60), repeat_contexts[-1]['motif_scale'])
            candidates = [
                pitch
                for pitch in range(60, 85)
                if pitch % 12 in scale_pitch_classes and pitch != final_event['pitch'] and abs(pitch - final_event['pitch']) <= 7
            ]
            if candidates:
                motion_hint = 2 if (int(settings.get('random_seed', 17)) + final_event['bar_index']) % 2 == 0 else -2
                final_event['pitch'] = min(
                    candidates,
                    key=lambda pitch: (
                        abs(pitch - (final_event['pitch'] + motion_hint)),
                        abs(pitch - final_event['pitch']),
                    ),
                )
                final_event['pitch_class'] = final_event['pitch'] % 12
                last_pitch = final_event['pitch']

        return repeated_events, last_pitch

    def _build_motif_track(
        self,
        bar_contexts: Sequence[Dict[str, Any]],
        total_duration_seconds: float,
        settings: Dict[str, Any],
        feature_list: List[Dict[str, Any]],
    ) -> None:
        if not settings.get('enable_motif', True):
            return

        phrase_length = 4 if int(settings.get('phrase_length_bars', 2)) >= 4 else 2
        last_pitch = None
        phrase_index = 0

        for block_start in range(0, len(bar_contexts), phrase_length * 2):
            phrase_a_contexts = bar_contexts[block_start:block_start + phrase_length]
            phrase_a_events, last_pitch = self._compose_phrase(
                phrase_a_contexts,
                total_duration_seconds,
                settings,
                phrase_index,
                last_pitch,
            )
            feature_list.extend(phrase_a_events)

            phrase_b_contexts = bar_contexts[block_start + phrase_length:block_start + phrase_length * 2]
            phrase_b_events, repeated_last_pitch = self._repeat_phrase_with_variation(
                phrase_a_events,
                phrase_b_contexts,
                total_duration_seconds,
                settings,
            )
            if repeated_last_pitch is not None:
                last_pitch = repeated_last_pitch
            feature_list.extend(phrase_b_events)
            phrase_index += 1

    def generate_structured_ambient_music(
        self,
        eeg_features: List[Dict[str, Any]],
        total_duration_seconds: Optional[float] = None,
        composition_settings: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        self.mapped_features = []
        self.role_note_counts = {role: 0 for role in ROLE_TRACK_ORDER}

        if not eeg_features:
            return self.mapped_features

        settings = self.config.copy()
        if composition_settings:
            settings.update({key: value for key, value in composition_settings.items() if value is not None})

        inferred_total_duration = max(
            float(feature.get('end_time', feature.get('start_time', 0.0) + feature.get('duration', 0.0)))
            for feature in eeg_features
        )
        total_duration_seconds = float(total_duration_seconds or inferred_total_duration or 1.0)
        total_duration_seconds = max(total_duration_seconds, 1.0)

        bar_contexts, _ = self._build_bar_contexts(eeg_features, total_duration_seconds, settings)
        if not bar_contexts:
            return self.mapped_features

        section_length = 16 if int(settings.get('section_length_bars', 8)) >= 16 else 8
        self.velocity_envelope = self._build_velocity_envelope(bar_contexts, section_length)

        self._build_pad_track(bar_contexts, total_duration_seconds, settings, self.mapped_features)
        self._build_bass_track(bar_contexts, total_duration_seconds, settings, self.mapped_features)
        self._build_motif_track(bar_contexts, total_duration_seconds, settings, self.mapped_features)

        self.mapped_features.sort(key=lambda feature: (feature['start_time'], feature.get('role', ''), feature['pitch']))

        for role in ROLE_TRACK_ORDER:
            self.role_note_counts[role] = sum(1 for feature in self.mapped_features if feature.get('role') == role)

        active_modes = {context['mode'] for context in bar_contexts}
        scale_label = 'Healing Ambient (Major)'
        if 'dorian' in active_modes and 'major' in active_modes:
            scale_label = 'Healing Ambient (Major/Dorian)'
        elif 'dorian' in active_modes:
            scale_label = 'Healing Ambient (Dorian)'
        self.active_scale_label = scale_label

        self.individual_params.update(
            {
                'scale_name': scale_label,
                'track_programs': ROLE_INSTRUMENT_PROGRAMS.copy(),
                'instrument_name': 'Healing Ensemble',
                'instrument_program': ROLE_INSTRUMENT_PROGRAMS['motif'],
                'role_note_counts': self.role_note_counts.copy(),
            }
        )
        return self.mapped_features

    def get_music_parameters(self) -> Dict[str, Any]:
        if not self.mapped_features:
            return {}

        if not any(feature.get('role') for feature in self.mapped_features):
            pitches = [feature['pitch'] for feature in self.mapped_features if not feature.get('is_harmony', False)]
            velocities = [feature['velocity'] for feature in self.mapped_features]
            return {
                'avg_pitch': float(np.mean(pitches)) if pitches else 60.0,
                'avg_velocity': float(np.mean(velocities)) if velocities else 60.0,
                'avg_tempo': self.tempo,
                'pitch_range': (min(pitches), max(pitches)) if pitches else (60, 60),
                'note_count': len(self.mapped_features),
                'bpm': self.tempo,
                'scale': self.individual_params.get('scale_name', 'Major'),
                'instrument': self.individual_params.get('instrument_name', 'Soft Piano'),
                'instrument_program': self.individual_params.get('instrument_program', self.instrument_program),
            }

        pitches = [feature['pitch'] for feature in self.mapped_features if feature.get('role') == 'motif']
        if not pitches:
            pitches = [feature['pitch'] for feature in self.mapped_features]
        velocities = [feature['velocity'] for feature in self.mapped_features]

        track_programs_text = ', '.join(
            f'{role}:{program}' for role, program in ROLE_INSTRUMENT_PROGRAMS.items() if self.role_note_counts.get(role)
        )
        return {
            'avg_pitch': float(np.mean(pitches)) if pitches else 60.0,
            'avg_velocity': float(np.mean(velocities)) if velocities else 60.0,
            'avg_tempo': self.tempo,
            'pitch_range': (min(pitches), max(pitches)) if pitches else (60, 60),
            'note_count': len(self.mapped_features),
            'bpm': self.tempo,
            'scale': self.individual_params.get('scale_name', self.active_scale_label),
            'instrument': 'Healing Ensemble',
            'instrument_program': track_programs_text or str(ROLE_INSTRUMENT_PROGRAMS['motif']),
            'track_programs': ROLE_INSTRUMENT_PROGRAMS.copy(),
            'role_note_counts': self.role_note_counts.copy(),
        }

    def get_total_duration(self) -> float:
        if not self.mapped_features:
            return 0.0
        return max(feature['start_time'] + feature['duration'] for feature in self.mapped_features)

    def get_individual_params(self) -> Dict[str, Any]:
        return self.individual_params


class EEGMusicMapper:

    def __init__(self, config: dict = None):
        self.config = config or MUSIC_MAPPING_CONFIG.copy()
        self.mapped_features: List[Dict[str, Any]] = []
        self.individual_mapper = IndividualizedMusicMapper(self.config)

    def configure_individualized(
        self,
        stats: Dict[str, Any],
        slow_wave_density: float = 3.0,
        user_overrides: Dict[str, Any] = None,
    ):
        self.individual_mapper.configure_from_eeg_stats(stats, slow_wave_density, user_overrides)

    def map_features(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = self.individual_mapper.map_features(eeg_features)
        return self.mapped_features

    def generate_ambient_structure(self, eeg_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.mapped_features = self.individual_mapper.generate_ambient_structure(eeg_features)
        return self.mapped_features

    def generate_structured_ambient_music(
        self,
        eeg_features: List[Dict[str, Any]],
        total_duration_seconds: Optional[float] = None,
        composition_settings: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        self.mapped_features = self.individual_mapper.generate_structured_ambient_music(
            eeg_features,
            total_duration_seconds=total_duration_seconds,
            composition_settings=composition_settings,
        )
        return self.mapped_features

    def get_music_parameters(self) -> Dict[str, Any]:
        return self.individual_mapper.get_music_parameters()

    def get_total_duration(self) -> float:
        return self.individual_mapper.get_total_duration()

    def get_individual_params(self) -> Dict[str, Any]:
        return self.individual_mapper.get_individual_params()
