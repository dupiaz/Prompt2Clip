import librosa
import numpy as np
from src.features.base import BaseFeatureExtractor

class LoudnessExtractor(BaseFeatureExtractor):
    """Extract short-term and long-term loudness energy."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    @property
    def name(self) -> str:
        return "loudness"
        
    @property
    def weight(self) -> float:
        return 0.18  # Combined short (0.10) + long (0.08)
        
    def extract(self, y: np.ndarray, hop_length: int = 512, **kwargs) -> tuple:
        """Returns tuple of (short_term_energy, long_term_energy)"""
        frame_length_short = int(0.03 * self.sr)
        rms_short = librosa.feature.rms(y=y, frame_length=frame_length_short, hop_length=hop_length)[0]
        
        frame_length_long = int(0.4 * self.sr)
        rms_long = librosa.feature.rms(y=y, frame_length=frame_length_long, hop_length=hop_length)[0]
        
        energy_short = librosa.amplitude_to_db(rms_short + 1e-6, ref=np.max)
        energy_long = librosa.amplitude_to_db(rms_long + 1e-6, ref=np.max)
        
        return energy_short, energy_long
