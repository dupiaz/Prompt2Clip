from openai import OpenAI
from src.llm.base import BaseLLMClient
from src.config.settings import AppSettings

class AiBoxLLMClient(BaseLLMClient):
    """AiBox API integration using OpenAI standard client."""
    
    def __init__(self):
        self.settings = AppSettings()
        api_key = self.settings.aibox_api_key
        if not api_key:
            raise ValueError("AIBOX_API_KEY not found in .env")
            
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.ai-box.vn/v1"
        )
        
    @property
    def provider_name(self) -> str:
        return "aibox"
        
    def generate(self, prompt: str, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 2000, temperature: float = 0.3, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Compatibility method for old codebase."""
        return self.generate(prompt, **kwargs)
