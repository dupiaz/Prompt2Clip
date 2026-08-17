from abc import ABC, abstractmethod
from typing import List

class BaseExporter(ABC):
    """Adapter interface for result exporters."""
    
    @abstractmethod
    def export(self, video_path: str, clips: list, output_dir: str = None, **kwargs) -> List[str]:
        """
        Export clips and return list of output file paths.
        """
        ...
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Exporter name for logging."""
        ...
