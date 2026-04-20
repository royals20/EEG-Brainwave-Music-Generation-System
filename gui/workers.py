import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from eeg_processing.feature_extractor import FeatureExtractor, smooth_features
from eeg_processing.sws_detector import SWSDetector
from music.audio_synth import AudioSynthesizer
from music.mapper import EEGMusicMapper
from music.midi_generator import MIDIGenerator
from utils.config import MUSIC_MAPPING_CONFIG, validate_subject_id
from utils.ui_text import format_extracting_features, format_synthesizing_audio


class WorkerCancelled(Exception):
    pass


def _collect_valid_sws_segments(
    eeg_data,
    fs: float,
    sws_results: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_sws = bool(sws_results) and int(sws_results.get('sws_epoch_count', 0) or 0) > 0
    if not valid_sws:
        return []

    min_segment_samples = max(16, int(fs * 0.5))
    segments = []
    for epoch in sws_results.get('sws_epochs', []):
        start_time = float(epoch.get('start_time', 0) or 0)
        end_time = float(epoch.get('end_time', start_time) or start_time)
        start_idx = max(0, int(round(start_time * fs)))
        end_idx = min(len(eeg_data), int(round(end_time * fs)))
        if end_idx - start_idx < min_segment_samples:
            continue

        segments.append(
            {
                'eeg_segment': eeg_data[start_idx:end_idx],
                'epoch_index': int(epoch.get('epoch_index', len(segments))),
                'start_time': start_time,
                'end_time': end_time,
                'duration': max(0.0, end_time - start_time),
                'slow_wave_ratio': epoch.get('slow_wave_ratio', 0),
                'avg_instantaneous_amplitude': epoch.get('avg_instantaneous_amplitude', 0),
                'avg_peak_to_peak_amplitude': epoch.get('avg_peak_to_peak_amplitude', 0),
                'slow_wave_count': epoch.get('slow_wave_count', 0),
                'slow_wave_time': epoch.get('slow_wave_time', 0),
                'wave_durations': epoch.get('wave_durations', []),
                'is_sws': epoch.get('is_sws', True),
            }
        )

    return segments


def prepare_music_source_features(
    eeg_data,
    fs: float,
    sws_results: Optional[Dict[str, Any]],
    feature_extractor: FeatureExtractor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sws_segments = _collect_valid_sws_segments(eeg_data, fs, sws_results)
    if sws_segments:
        raw_features: List[Dict[str, Any]] = []
        for segment in sws_segments:
            feature = feature_extractor.extract_from_segment(segment['eeg_segment'])
            feature['epoch_index'] = segment['epoch_index']
            feature['start_time'] = segment['start_time']
            feature['end_time'] = segment['end_time']
            feature['duration'] = segment['duration']
            feature['slow_wave_ratio'] = segment.get('slow_wave_ratio', feature.get('slow_wave_ratio', 0))
            feature['avg_instantaneous_amplitude'] = segment.get(
                'avg_instantaneous_amplitude',
                feature.get('avg_instantaneous_amplitude', 0),
            )
            feature['avg_peak_to_peak_amplitude'] = segment.get(
                'avg_peak_to_peak_amplitude',
                feature.get('avg_peak_to_peak_amplitude', 0),
            )
            feature['slow_wave_count'] = segment.get('slow_wave_count', 0)
            feature['slow_wave_time'] = segment.get('slow_wave_time', 0)
            feature['wave_durations'] = segment.get('wave_durations', [])
            feature['is_sws'] = segment.get('is_sws', True)
            raw_features.append(feature)

        if raw_features:
            features_list = raw_features
            if len(raw_features) >= feature_extractor.smoothing_window:
                features_list = smooth_features(raw_features, feature_extractor.smoothing_window)

            feature_extractor.raw_features = raw_features
            feature_extractor.features_list = features_list
            return features_list, {
                'source_mode': 'sws_only',
                'effective_duration_seconds': sum(
                    float(feature.get('duration', 0) or 0) for feature in features_list
                ),
                'source_epoch_count': len(features_list),
            }

    features_list = feature_extractor.extract(eeg_data)
    return features_list, {
        'source_mode': 'full_eeg_fallback',
        'effective_duration_seconds': sum(float(feature.get('duration', 0) or 0) for feature in features_list),
        'source_epoch_count': len(features_list),
    }


def prepare_music_window_features(
    eeg_data,
    fs: float,
    sws_results: Optional[Dict[str, Any]],
    feature_extractor: FeatureExtractor,
    window_size: float = 2.0,
    hop_size: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sws_segments = _collect_valid_sws_segments(eeg_data, fs, sws_results)
    if sws_segments:
        music_windows: List[Dict[str, Any]] = []
        compressed_start_time = 0.0
        for segment in sws_segments:
            segment_windows = feature_extractor.extract_music_windows(
                segment['eeg_segment'],
                window_size=window_size,
                hop_size=hop_size,
                start_time=compressed_start_time,
            )
            for window in segment_windows:
                window['source_start_time'] = float(segment['start_time'] + (window['start_time'] - compressed_start_time))
                window['source_end_time'] = float(segment['start_time'] + (window['end_time'] - compressed_start_time))
                window['epoch_index'] = segment['epoch_index']
            music_windows.extend(segment_windows)
            compressed_start_time += segment['duration']

        if music_windows:
            return music_windows, {
                'source_mode': 'sws_only',
                'effective_duration_seconds': compressed_start_time,
                'source_epoch_count': len(sws_segments),
                'music_window_count': len(music_windows),
            }

    music_windows = feature_extractor.extract_music_windows(
        eeg_data,
        window_size=window_size,
        hop_size=hop_size,
        start_time=0.0,
    )
    return music_windows, {
        'source_mode': 'full_eeg_fallback',
        'effective_duration_seconds': len(eeg_data) / fs if fs else 0.0,
        'source_epoch_count': len(feature_extractor.features_list),
        'music_window_count': len(music_windows),
    }


class CancelableWorker(QObject):
    progress = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str, str)
    cancelled = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _check_cancelled(self):
        if self._cancel_requested:
            raise WorkerCancelled()


class SWSDetectionWorker(CancelableWorker):

    def __init__(self, eeg_data, fs: float, config: dict, channel_name: str):
        super().__init__()
        self.eeg_data = eeg_data
        self.fs = fs
        self.config = config
        self.channel_name = channel_name

    @pyqtSlot()
    def run(self):
        try:
            self._check_cancelled()
            self.progress.emit(f'正在检测通道 {self.channel_name} 的 SWS...')
            detector = SWSDetector(self.config)
            detection_result = detector.detect(self.eeg_data, self.fs)
            self._check_cancelled()
            self.result.emit(detection_result)
        except WorkerCancelled:
            self.cancelled.emit('SWS 检测已取消。')
        except Exception as exc:
            self.error.emit('SWS 检测失败', f'错误详情：{exc}\n\n{traceback.format_exc()}')
        finally:
            self.finished.emit()


class MusicGenerationWorker(CancelableWorker):

    def __init__(
        self,
        eeg_data,
        fs: float,
        sws_results: dict,
        user_music_params: dict,
        subject_id: str,
        subject_dirs: dict,
        channel_name: str,
    ):
        super().__init__()
        self.eeg_data = eeg_data
        self.fs = fs
        self.sws_results = sws_results
        self.user_music_params = user_music_params
        self.subject_id = subject_id
        self.subject_dirs = subject_dirs
        self.channel_name = channel_name

    @pyqtSlot()
    def run(self):
        try:
            self._check_cancelled()
            self.progress.emit(format_extracting_features(self.channel_name))
            validated_subject_id = validate_subject_id(self.subject_id)
            feature_extractor = FeatureExtractor(self.fs)
            all_features, source_meta = prepare_music_source_features(
                self.eeg_data,
                self.fs,
                self.sws_results,
                feature_extractor,
            )
            music_windows, window_meta = prepare_music_window_features(
                self.eeg_data,
                self.fs,
                self.sws_results,
                feature_extractor,
                window_size=float(self.user_music_params.get('music_window_seconds', MUSIC_MAPPING_CONFIG['music_window_seconds'])),
                hop_size=float(self.user_music_params.get('music_hop_seconds', MUSIC_MAPPING_CONFIG['music_hop_seconds'])),
            )
            self._check_cancelled()
            if not all_features or not music_windows:
                raise ValueError('无法提取用于音乐生成的 EEG 特征。')

            if source_meta['source_mode'] == 'sws_only':
                self.progress.emit(
                    f'正在使用 {source_meta["source_epoch_count"]} 个 SWS 片段构建音乐特征...'
                )
            else:
                self.progress.emit('未检出可用的 SWS 片段，正在回退到全段 EEG 特征生成...')

            stats = feature_extractor.get_individualized_stats()
            slow_wave_density = self.sws_results.get('slow_wave_density', 0.0) if source_meta['source_mode'] == 'sws_only' else 0.0

            music_mapper = EEGMusicMapper()
            music_mapper.configure_individualized(stats, slow_wave_density, self.user_music_params)
            individual_params = music_mapper.get_individual_params()
            self._check_cancelled()

            keep_original_duration = bool(self.user_music_params.get('keep_original_duration', True))
            target_duration_minutes = self.user_music_params.get('target_duration_minutes')
            if keep_original_duration:
                render_duration_seconds = source_meta['effective_duration_seconds']
            elif target_duration_minutes:
                render_duration_seconds = max(float(target_duration_minutes) * 60.0, 1.0)
            else:
                render_duration_seconds = source_meta['effective_duration_seconds']

            music_features = music_mapper.generate_structured_ambient_music(
                music_windows,
                total_duration_seconds=render_duration_seconds,
                composition_settings=self.user_music_params,
            )
            self._check_cancelled()
            if not music_features:
                raise ValueError('无法生成有效的音乐映射结果。')

            self.progress.emit(f'已生成 {len(music_features)} 个音符，正在写入 MIDI...')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            midi_generator = MIDIGenerator(
                bpm=individual_params['tempo'],
                instrument_program=None,
            )
            midi_path = midi_generator.generate(
                music_features,
                filename=f'{validated_subject_id}_{timestamp}.mid',
                output_dir=self.subject_dirs['midi'],
            )
            self._check_cancelled()

            self.progress.emit(format_synthesizing_audio(render_duration_seconds / 60.0))
            audio_synthesizer = AudioSynthesizer()
            audio_path = audio_synthesizer.synthesize(
                midi_path,
                filename=f'{validated_subject_id}_{timestamp}.wav',
                output_dir=self.subject_dirs['audio'],
                backend=str(self.user_music_params.get('render_backend', 'auto')).lower(),
                soundfont_path=self.user_music_params.get('soundfont_path') or None,
            )
            self._check_cancelled()

            audio_info = audio_synthesizer.get_audio_info()
            music_duration = audio_info.get('duration', 0) if audio_info else 0
            music_params = music_mapper.get_music_parameters()

            self.result.emit(
                {
                    'midi_path': midi_path,
                    'audio_path': audio_path,
                    'music_features': music_features,
                    'music_params': music_params,
                    'music_duration': music_duration,
                    'individual_params': individual_params,
                    'source_mode': source_meta['source_mode'],
                    'source_epoch_count': source_meta['source_epoch_count'],
                    'effective_duration_minutes': source_meta['effective_duration_seconds'] / 60.0,
                    'music_window_count': window_meta['music_window_count'],
                    'render_info': audio_synthesizer.get_render_info(),
                    'used_full_eeg_fallback': source_meta['source_mode'] != 'sws_only',
                }
            )
        except WorkerCancelled:
            self.cancelled.emit('音乐生成已取消。')
        except Exception as exc:
            self.error.emit('音乐生成失败', f'错误详情：{exc}\n\n{traceback.format_exc()}')
        finally:
            self.finished.emit()
