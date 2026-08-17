import librosa
import numpy as np
import scipy.sparse.csgraph
import scipy.linalg
from src.features.base import BaseFeatureExtractor

class StructuralBoundaryExtractor(BaseFeatureExtractor):
    """Detects structural boundaries using recurrence matrix."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    @property
    def name(self) -> str:
        return "structural_boundary"
        
    @property
    def weight(self) -> float:
        return 0.12
        
    def extract(self, y: np.ndarray, hop_length: int = 512, **kwargs) -> np.ndarray:
        n_fft = min(2048, len(y))
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        
        S = librosa.segment.recurrence_matrix(mfcc, mode='affinity', metric='cosine')
        
        # Add small constant to avoid divide by zero
        S = S + 1e-8
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # normalize laplacian
                L = scipy.sparse.csgraph.laplacian(S, normed=True)
                # Compute eigenvectors
                eigenvals, eigenvecs = scipy.linalg.eigh(L)
                # Ensure we have enough eigenvectors
                k = min(3, eigenvecs.shape[1])
                structural_change = np.sum(np.abs(np.diff(eigenvecs[:, :k], axis=0)), axis=1)
                structural_change = np.pad(structural_change, (0, 1), 'edge')
            except Exception as e:
                print(f"[AUDIO] Boundary detection failed, falling back to zeros: {e}")
                structural_change = np.zeros(S.shape[0])
                
        return structural_change
