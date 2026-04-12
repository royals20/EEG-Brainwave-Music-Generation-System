import numpy as np
from typing import Tuple, Optional, Dict, Any
import os

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from utils.config import EEG_SIMULATION_CONFIG


class EEGLoader:
    
    def __init__(self):
        self.raw = None
        self.data = None
        self.sample_rate = None
        self.channel_names = None
        self.duration = None
        self.file_path = None
        self.is_simulated = False
    
    def load_edf(self, file_path: str) -> Dict[str, Any]:
        if not MNE_AVAILABLE:
            raise ImportError("MNE library not available. Install with: pip install mne")
        
        self.file_path = file_path
        self.is_simulated = False
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.edf':
            self.raw = mne.io.read_raw_edf(file_path, preload=True)
        elif ext == '.bdf':
            self.raw = mne.io.read_raw_bdf(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        self.data = self.raw.get_data()
        self.sample_rate = self.raw.info['sfreq']
        self.channel_names = self.raw.ch_names
        self.duration = self.raw.times[-1]
        
        return self.get_info()
    
    def load_csv(self, file_path: str, sample_rate: float = 256) -> Dict[str, Any]:
        self.file_path = file_path
        self.is_simulated = False
        
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        self.data = data
        self.sample_rate = sample_rate
        self.channel_names = [f'Channel_{i+1}' for i in range(data.shape[0])]
        self.duration = data.shape[1] / sample_rate
        
        return self.get_info()
    
    def load_simulated(self, duration_minutes: float = None, 
                       sample_rate: int = None,
                       sws_ratio: float = None) -> Dict[str, Any]:
        config = EEG_SIMULATION_CONFIG
        duration_minutes = duration_minutes or config['duration_minutes']
        sample_rate = sample_rate or config['sample_rate']
        sws_ratio = sws_ratio or config['sws_ratio']
        
        self.is_simulated = True
        self.sample_rate = sample_rate
        self.duration = duration_minutes * 60
        self.channel_names = ['Fz', 'Cz', 'Pz', 'Oz']
        
        n_samples = int(self.duration * sample_rate)
        n_channels = len(self.channel_names)
        
        self.data = np.zeros((n_channels, n_samples))
        
        t = np.linspace(0, self.duration, n_samples)
        
        for ch in range(n_channels):
            noise = np.random.randn(n_samples) * 10
            
            background = np.sin(2 * np.pi * 0.5 * t + np.random.rand() * 2 * np.pi) * 5
            
            n_sws_segments = int(self.duration * sws_ratio / 30)
            for _ in range(n_sws_segments):
                start_time = np.random.rand() * (self.duration - 30)
                start_idx = int(start_time * sample_rate)
                end_idx = int((start_time + 30) * sample_rate)
                
                if end_idx > n_samples:
                    end_idx = n_samples
                
                segment_len = end_idx - start_idx
                sws_freq = 0.75 + np.random.rand() * 0.5
                sws_amplitude = 75 + np.random.rand() * 50
                
                sws_signal = np.sin(2 * np.pi * sws_freq * t[start_idx:end_idx]) * sws_amplitude
                self.data[ch, start_idx:end_idx] += sws_signal
            
            self.data[ch, :] += noise + background
        
        return self.get_info()
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'sample_rate': self.sample_rate,
            'channel_count': len(self.channel_names) if self.channel_names else 0,
            'duration': self.duration,
            'channel_names': self.channel_names,
            'file_path': self.file_path,
            'is_simulated': self.is_simulated
        }
    
    def get_channel_data(self, channel_idx: int = 0) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data loaded. Call load_edf() or load_simulated() first.")
        return self.data[channel_idx, :]
    
    def get_all_data(self) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data loaded. Call load_edf() or load_simulated() first.")
        return self.data
    
    def get_time_vector(self) -> np.ndarray:
        if self.sample_rate is None or self.duration is None:
            raise ValueError("No data loaded.")
        return np.linspace(0, self.duration, int(self.duration * self.sample_rate))


def simulate_eeg(duration_minutes: float = 10, sample_rate: int = 256,
                 sws_ratio: float = 0.3) -> Tuple[np.ndarray, float]:
    loader = EEGLoader()
    loader.load_simulated(duration_minutes, sample_rate, sws_ratio)
    return loader.get_channel_data(0), loader.sample_rate
