import numpy as np
from scipy.interpolate import interp1d
from src.config.settings import AppSettings

class SignalFusion:
    """Fuses audio and video signals."""
    def __init__(self):
        self.settings = AppSettings()
        
    def fuse(self, audio_data: dict, video_data: dict) -> tuple:
        print("=" * 60)
        print("PHASE 2: SIGNAL FUSION")
        print("=" * 60 + "\n")
        
        max_time = max(audio_data["timestamps"][-1], video_data["timestamps"][-1])
        unified_timestamps = np.arange(0, max_time, 1.0)
        
        audio_interp = interp1d(
            audio_data["timestamps"], audio_data["scores"],
            kind='linear', bounds_error=False, fill_value=0
        )
        audio_unified = audio_interp(unified_timestamps)
        
        video_interp = interp1d(
            video_data["timestamps"], video_data["scores"],
            kind='linear', bounds_error=False, fill_value=0
        )
        video_unified = video_interp(unified_timestamps)
        
        combined_scores = (
            self.settings.audio_weight * audio_unified +
            self.settings.video_weight * video_unified
        )
        
        return unified_timestamps, combined_scores, audio_unified, video_unified
