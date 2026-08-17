from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """Adapter interface for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        """
        ...
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        ...
