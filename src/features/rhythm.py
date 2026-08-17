import librosa
import numpy as np
from src.features.base import BaseFeatureExtractor

class RhythmExtractor(BaseFeatureExtractor):
    """Temporal Rhythm & Onsets - beats, bursts, fast speech."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    @property
    def name(self) -> str:
        return "rhythm_onsets"
        
    @property
    def weight(self) -> float:
        return 0.15 # 0.08 onset + 0.07 rhythm
        
    def extract(self, y: np.ndarray, hop_length: int = 512, **kwargs) -> tuple:
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr, hop_length=hop_length)[0]
        rhythm_var = np.array([np.std(onset_env[max(0, i-10):i+1]) for i in range(len(onset_env))])
        
        return onset_env, rhythm_var, tempo
