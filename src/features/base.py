from typing import Any
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    """Abstract base for feature extractors."""
    
    @abstractmethod
    def extract(self, data: Any, **kwargs) -> Any:
        """Extract feature from input data."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable name for the feature."""
        ...
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Default weight in composite scoring."""
        ...
