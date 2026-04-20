import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.config import CSV_OUTPUT_DIR, ensure_directories


def _resolve_output_path(output_path: Optional[str] = None) -> str:
    ensure_directories()

    if output_path is None:
        output_path = os.path.join(CSV_OUTPUT_DIR, 'experiment_results.csv')

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    return output_path


def _load_csv_rows(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_csv_fieldnames(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return next(reader, [])


def _resolve_fieldnames(existing_rows: List[Dict[str, Any]],
                        new_rows: List[Dict[str, Any]],
                        preferred_fieldnames: Optional[List[str]] = None) -> List[str]:
    fieldnames = []

    for candidate_fields in [preferred_fieldnames or []]:
        for field in candidate_fields:
            if field not in fieldnames:
                fieldnames.append(field)

    for dataset in [existing_rows, new_rows]:
        for row in dataset:
            for field in row.keys():
                if field not in fieldnames:
                    fieldnames.append(field)

    return fieldnames


def _normalize_rows(rows: List[Dict[str, Any]], fieldnames: List[str]) -> List[Dict[str, Any]]:
    return [{field: row.get(field, '') for field in fieldnames} for row in rows]


def _write_csv_rows(rows: List[Dict[str, Any]], output_path: str, fieldnames: List[str]) -> str:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_normalize_rows(rows, fieldnames))

    return output_path


def _append_csv_rows(rows: List[Dict[str, Any]], output_path: str, fieldnames: List[str]) -> str:
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(_normalize_rows(rows, fieldnames))

    return output_path


def export_to_csv(data: List[Dict[str, Any]], output_path: str = None,
                  fieldnames: List[str] = None) -> str:
    output_path = _resolve_output_path(output_path)

    if not data:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return output_path

    resolved_fieldnames = _resolve_fieldnames([], data, fieldnames)
    return _write_csv_rows(data, output_path, resolved_fieldnames)


def append_to_csv(data: List[Dict[str, Any]], output_path: str = None,
                  fieldnames: List[str] = None) -> str:
    output_path = _resolve_output_path(output_path)
    if not data:
        if not os.path.exists(output_path):
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                f.write('')
        return output_path

    existing_fieldnames = _read_csv_fieldnames(output_path)
    if not existing_fieldnames:
        resolved_fieldnames = _resolve_fieldnames([], data, fieldnames)
        return _write_csv_rows(data, output_path, resolved_fieldnames)

    required_fieldnames = _resolve_fieldnames([], data, fieldnames)
    if all(field in existing_fieldnames for field in required_fieldnames):
        return _append_csv_rows(data, output_path, existing_fieldnames)

    existing_data = _load_csv_rows(output_path)
    combined_data = existing_data + data
    resolved_fieldnames = _resolve_fieldnames(existing_data, data, fieldnames)
    return _write_csv_rows(combined_data, output_path, resolved_fieldnames)


def export_subject_results(subject_data: Dict[str, Any], 
                           sws_data: Dict[str, Any],
                           music_data: Dict[str, Any],
                           output_path: str = None) -> str:
    result = {
        'subject_id': subject_data.get('subject_id', ''),
        'name': subject_data.get('name', ''),
        'age': subject_data.get('age', ''),
        'gender': subject_data.get('gender', ''),
        'psqi_score': subject_data.get('psqi_score', ''),
        'group_type': subject_data.get('group_type', ''),
        'SWS_duration': sws_data.get('total_sws_duration', 0),
        'avg_delta_power': sws_data.get('avg_delta_power', 0),
        'sws_segment_count': sws_data.get('sws_segment_count', 0),
        'avg_pitch': music_data.get('avg_pitch', 0),
        'avg_tempo': music_data.get('avg_tempo', 0),
        'music_duration': music_data.get('music_duration', 0),
        'export_time': datetime.now().isoformat()
    }
    
    return append_to_csv([result], output_path, list(result.keys()))


class CSVExporter:
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or CSV_OUTPUT_DIR
        self.results = []
    
    def add_result(self, subject_id: str, sws_duration: float,
                   avg_delta_power: float, avg_pitch: float,
                   avg_tempo: float, **kwargs):
        result = {
            'subject_id': subject_id,
            'SWS_duration': sws_duration,
            'avg_delta_power': avg_delta_power,
            'avg_pitch': avg_pitch,
            'avg_tempo': avg_tempo,
            **kwargs
        }
        result.setdefault('export_time', datetime.now().isoformat())
        self.results.append(result)
    
    def export(self, filename: str = 'experiment_results.csv',
               fieldnames: List[str] = None,
               append: bool = True) -> str:
        output_path = os.path.join(self.output_dir, filename)

        if fieldnames is None and self.results:
            fieldnames = list(self.results[0].keys())

        if append:
            return append_to_csv(self.results, output_path, fieldnames)

        return export_to_csv(self.results, output_path, fieldnames)
    
    def export_summary(self, filename: str = 'summary.csv') -> str:
        if not self.results:
            return ''
        
        summary = {
            'total_subjects': len(self.results),
            'avg_sws_duration': sum(r.get('SWS_duration', 0) for r in self.results) / len(self.results),
            'avg_delta_power': sum(r.get('avg_delta_power', 0) for r in self.results) / len(self.results),
            'avg_pitch': sum(r.get('avg_pitch', 0) for r in self.results) / len(self.results),
            'avg_tempo': sum(r.get('avg_tempo', 0) for r in self.results) / len(self.results),
            'export_time': datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        
        return output_path
    
    def clear(self):
        self.results = []
    
    def load_existing(self, filepath: str) -> List[Dict[str, Any]]:
        return _load_csv_rows(filepath)
    
    def merge_results(self, filepath: str) -> str:
        existing = self.load_existing(filepath)
        
        existing_ids = {r.get('subject_id') for r in existing}
        
        for result in self.results:
            if result.get('subject_id') not in existing_ids:
                existing.append(result)
        
        if existing:
            fieldnames = _resolve_fieldnames([], existing)
            _write_csv_rows(existing, filepath, fieldnames)
        
        return filepath
