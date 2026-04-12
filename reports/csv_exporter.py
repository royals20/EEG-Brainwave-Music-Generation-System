import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.config import CSV_OUTPUT_DIR, ensure_directories


def export_to_csv(data: List[Dict[str, Any]], output_path: str = None,
                  fieldnames: List[str] = None) -> str:
    ensure_directories()
    
    if output_path is None:
        output_path = os.path.join(CSV_OUTPUT_DIR, 'experiment_results.csv')
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not data:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return output_path
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return output_path


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
    
    if output_path is None:
        output_path = os.path.join(CSV_OUTPUT_DIR, 'experiment_results.csv')
    
    existing_data = []
    if os.path.exists(output_path):
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
    
    existing_data.append(result)
    
    fieldnames = list(result.keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
    
    return output_path


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
        self.results.append(result)
    
    def export(self, filename: str = 'experiment_results.csv',
               fieldnames: List[str] = None) -> str:
        output_path = os.path.join(self.output_dir, filename)
        
        if fieldnames is None and self.results:
            fieldnames = list(self.results[0].keys())
        
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
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def merge_results(self, filepath: str) -> str:
        existing = self.load_existing(filepath)
        
        existing_ids = {r.get('subject_id') for r in existing}
        
        for result in self.results:
            if result.get('subject_id') not in existing_ids:
                existing.append(result)
        
        if existing:
            fieldnames = list(existing[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing)
        
        return filepath
