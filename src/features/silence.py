import librosa
import numpy as np
from scipy.signal import convolve
from src.features.base import BaseFeatureExtractor

class SilenceContrastExtractor(BaseFeatureExtractor):
    """Detects dramatic pauses and silence contrast."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    @property
    def name(self) -> str:
        return "silence_contrast"
        
    @property
    def weight(self) -> float:
        return 0.10
        
    def extract(self, y: np.ndarray, hop_length: int = 512, **kwargs) -> np.ndarray:
        frame_length_short = int(0.03 * self.sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_length_short, hop_length=hop_length)[0]
        
        threshold = np.mean(rms) * 0.5
        is_speech = rms > threshold
        
        transitions = np.abs(np.diff(is_speech.astype(int)))
        transitions = np.pad(transitions, (0, 1), 'constant')
        
        # Weight transitions higher if followed by loud speech
        contrast_score = np.zeros_like(transitions, dtype=float)
        for i in np.where(transitions > 0)[0]:
            # Look ahead a few frames (~0.5s)
            ahead = min(i + int(self.sr/hop_length * 0.5), len(rms)-1)
            contrast_score[i] = max(0, rms[ahead] - rms[i])
            
        # Smooth
        window = 5
        if len(contrast_score) >= window:
            contrast_score = convolve(contrast_score, np.ones(window)/window, mode='same')
            
        return np.maximum(0, contrast_score)
