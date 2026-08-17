import numpy as np
from src.features.base import BaseFeatureExtractor

class VisualSurpriseExtractor(BaseFeatureExtractor):
    """Calculates visual surprise based on CLIP embeddings."""
    
    @property
    def name(self) -> str:
        return "visual_surprise"
        
    @property
    def weight(self) -> float:
        return 0.20
        
    def extract(self, feat_flat: np.ndarray, prev_feats: list, **kwargs) -> float:
        """
        Extract visual surprise score.
        
        Args:
            feat_flat: Current frame CLIP embedding
            prev_feats: List of recent historical CLIP embeddings
        """
        if len(prev_feats) == 0:
            return 0.0
        
        # Calculate cosine distances to recent history
        distances = []
        feat_norm = np.linalg.norm(feat_flat) + 1e-6
        for prev in prev_feats[-10:]:
            prev_norm = np.linalg.norm(prev) + 1e-6
            cos_sim = np.dot(feat_flat, prev) / (feat_norm * prev_norm)
            distances.append(1 - cos_sim)
            
        # Surprise is the max deviation from recent history
        surprise = np.max(distances)
        return surprise
