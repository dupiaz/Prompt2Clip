from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.features.base import BaseFeatureExtractor

class BaseAnalyzer(ABC):
    """
    Adapter interface for analyzers.
    An analyzer orchestrates multiple feature extractors, normalizes
    their outputs, and produces a unified score timeline.
    """
    
    @abstractmethod
    def analyze(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run full analysis on input file.
        Returns Dict with at minimum: {"timestamps": np.ndarray, "scores": np.ndarray}
        """
        ...
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Analyzer name for logging."""
        ...
        
    @abstractmethod
    def get_feature_extractors(self) -> List[BaseFeatureExtractor]:
        """Return list of feature extractors this analyzer uses."""
        ...
