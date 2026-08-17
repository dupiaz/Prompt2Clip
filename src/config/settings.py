import os
from pathlib import Path
from dotenv import load_dotenv

class AppSettings:
    """
    Singleton configuration manager.
    Loads all .env files once and provides typed access to settings.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Project root is Prompt2Clip/ (two levels up from src/config)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Load all env files in envs/
        envs_dir = self.project_root / "envs"
        if envs_dir.exists():
            for env_file in envs_dir.glob("*.env"):
                load_dotenv(env_file)
        
        # Also load root .env if exists
        root_env = self.project_root / ".env"
        if root_env.exists():
            load_dotenv(root_env)
            
    @property
    def aibox_api_key(self) -> str:
        return os.getenv("AIBOX_API_KEY", "")
        
    @property
    def output_dir(self) -> str:
        return getattr(self, '_output_dir', os.getenv("OUTPUT_DIR", "output"))
        
    @output_dir.setter
    def output_dir(self, value: str):
        self._output_dir = value
        
    @property
    def cache_dir(self) -> str:
        return os.getenv("CACHE_DIR", ".cache")
        
    @property
    def whisper_env(self) -> str:
        return os.getenv("WHISPER_ENV", "clipz_whisper")
        
    @property
    def vision_env(self) -> str:
        return os.getenv("VISION_ENV", "clipz_vision")
        
    @property
    def audio_env(self) -> str:
        return os.getenv("AUDIO_ENV", "clipz_audio")
        
    @property
    def audio_weight(self) -> float:
        return getattr(self, '_audio_weight', float(os.getenv("AUDIO_WEIGHT", "0.5")))
        
    @audio_weight.setter
    def audio_weight(self, value: float):
        self._audio_weight = value
        
    @property
    def video_weight(self) -> float:
        return getattr(self, '_video_weight', float(os.getenv("VIDEO_WEIGHT", "0.5")))
        
    @video_weight.setter
    def video_weight(self, value: float):
        self._video_weight = value
        
    @property
    def whisper_model_size(self) -> str:
        return os.getenv("WHISPER_MODEL_SIZE", "medium")
        
    @property
    def whisper_device(self) -> str:
        return os.getenv("WHISPER_DEVICE", "cuda")
        
    @property
    def whisper_compute_type(self) -> str:
        return os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        
    @property
    def audio_sample_rate(self) -> int:
        return int(os.getenv("SAMPLE_RATE", "16000"))
