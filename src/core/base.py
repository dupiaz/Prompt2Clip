from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePipeline(ABC):
    """Abstract base for end-to-end processing pipelines."""
    
    @abstractmethod
    def process(self, video_path: str, user_query: str, **kwargs) -> Dict[str, Any]:
        """Run the full pipeline."""
        ...
