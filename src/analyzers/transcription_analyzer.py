import os
import json
from typing import Dict, Any, List

from src.analyzers.base import BaseAnalyzer
from src.features.base import BaseFeatureExtractor
from src.utils.cache import CacheManager
from src.config.settings import AppSettings

class TranscriptionAnalyzer(BaseAnalyzer):
    """Generates transcriptions using faster-whisper."""
    
    def __init__(self):
        self.settings = AppSettings()
        self.cache = CacheManager(namespace="transcription")
        
    @property
    def name(self) -> str:
        return "transcription_analyzer"
        
    def get_feature_extractors(self) -> List[BaseFeatureExtractor]:
        return []
        
    def transcribe_with_timestamps(self, audio_path: str, model_size: str = "medium", verbose: bool = False, use_cache: bool = True) -> list:
        """Compatibility method for old workers."""
        res = self.analyze(audio_path, model_size=model_size, verbose=verbose, use_cache=use_cache)
        return res["segments"]
        
    def analyze(self, input_path: str, model_size: str = "medium", verbose: bool = False, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Run transcription.
        Returns: Dict containing list of segments.
        """
        model_size = model_size or self.settings.whisper_model_size
        cache_key = f"{input_path}_{model_size}"
        
        if use_cache:
            cached = self.cache.load_json(cache_key)
            if cached:
                print(f"[TRANSCRIPTION] Using cached transcription")
                return {"segments": cached}
                
        print(f"[TRANSCRIPTION] Transcribing audio (this may take a while)...")
        print(f"[TRANSCRIPTION] Using faster-whisper on {self.settings.whisper_device}...")
        
        # Load DLLs for Windows CUDA
        if os.name == 'nt':
            import site
            for site_pkg in site.getsitepackages():
                for pkg in ['cublas', 'cudnn', 'cuda_nvrtc', 'cuda_runtime']:
                    path = os.path.join(site_pkg, 'nvidia', pkg, 'bin')
                    if os.path.exists(path):
                        try:
                            os.add_dll_directory(path)
                        except Exception:
                            pass
                        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
                        
        from faster_whisper import WhisperModel
        
        model = WhisperModel(
            model_size, 
            device=self.settings.whisper_device, 
            compute_type=self.settings.whisper_compute_type
        )
        
        segments_generator, info = model.transcribe(input_path, task="transcribe")
        
        sentences = []
        for segment in segments_generator:
            sentences.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
        if use_cache:
            self.cache.save_json(cache_key, sentences)
            print(f"[TRANSCRIPTION] Cached transcription for future use")
            
        return {"segments": sentences}
