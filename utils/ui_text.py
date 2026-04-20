APP_TITLE = 'EEG 脑波音乐生成系统'
STATUS_READY = '就绪'

SCALE_DISPLAY_NAMES = {
    'Pentatonic': '五声音阶',
    'Pentatonic (五声音阶)': '五声音阶',
    'Pentatonic (浜斿０闊抽樁)': '五声音阶',
    'Major': '大调',
    'Major (大调)': '大调',
    'Major (澶ц皟)': '大调',
    'Minor': '小调',
    'Minor (小调)': '小调',
    'Minor (灏忚皟)': '小调',
}

INSTRUMENT_DISPLAY_NAMES = {
    'Ambient Pad': '氛围铺底',
    'Soft Piano': '柔和钢琴',
    'Warm Strings': '温暖弦乐',
    '姘涘洿闊冲灚': '氛围铺底',
    '鏌斿拰閽㈢惔': '柔和钢琴',
    '娓╂殩寮︿箰': '温暖弦乐',
}


def display_scale_name(scale_name: str) -> str:
    if not scale_name:
        return ''
    return SCALE_DISPLAY_NAMES.get(scale_name, scale_name)


def display_instrument_name(instrument_name: str) -> str:
    if not instrument_name:
        return ''
    return INSTRUMENT_DISPLAY_NAMES.get(instrument_name, instrument_name)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = float(seconds % 60)
    if remaining_seconds.is_integer():
        return f'{minutes}分{int(remaining_seconds)}秒'
    return f'{minutes}分{remaining_seconds:.1f}秒'


def format_subject_selected(subject_id: str, subject_name: str) -> str:
    return f'已选择受试者：{subject_id} - {subject_name}'


def format_history_loaded(subject_id: str, subject_name: str, restored_parts: str, session_count: int) -> str:
    return (
        f'已选择受试者：{subject_id} - {subject_name}；'
        f'已恢复历史：{restored_parts}；会话数={session_count}'
    )


def format_detecting_sws(channel_name: str) -> str:
    return f'正在检测通道 {channel_name} 的 SWS...'


def format_extracting_features(channel_name: str) -> str:
    return f'正在提取通道 {channel_name} 的 EEG 特征...'


def format_synthesizing_audio(duration_minutes: float) -> str:
    return f'正在合成 {duration_minutes:.0f} 分钟音频...'
