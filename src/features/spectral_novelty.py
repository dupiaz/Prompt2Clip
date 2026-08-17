import librosa
import numpy as np
from src.features.base import BaseFeatureExtractor

class SpectralNoveltyExtractor(BaseFeatureExtractor):
    """Spectral Novelty - captures texture/speaker/music changes."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    @property
    def name(self) -> str:
        return "spectral_novelty"
        
    @property
    def weight(self) -> float:
        return 0.25
        
    def extract(self, y: np.ndarray, hop_length: int = 512, **kwargs) -> np.ndarray:
        n_fft = min(2048, len(y))
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        
        novelty = np.zeros(mfcc.shape[1])
        window = 10
        
        for i in range(window, mfcc.shape[1]):
            curr = mfcc[:, i]
            prev_window = mfcc[:, i-window:i]
            
            prev_mean = np.mean(prev_window, axis=1)
            
            cos_sim = np.dot(curr, prev_mean) / (np.linalg.norm(curr) * np.linalg.norm(prev_mean) + 1e-6)
            novelty[i] = 1 - cos_sim
        
        return novelty
