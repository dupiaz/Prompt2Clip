import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List

from src.analyzers.base import BaseAnalyzer
from src.features.base import BaseFeatureExtractor
from src.features.loudness import LoudnessExtractor
from src.features.spectral_novelty import SpectralNoveltyExtractor
from src.features.rhythm import RhythmExtractor
from src.features.silence import SilenceContrastExtractor
from src.features.semantic_events import SemanticEventExtractor

from src.utils.normalization import robust_normalize, ema_filter, multi_scale_window
from src.utils.cache import CacheManager
from src.config.settings import AppSettings

class AudioAnalyzer(BaseAnalyzer):
    """Orchestrates audio feature extraction and scoring."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.settings = AppSettings()
        self.cache = CacheManager(namespace="audio")
        
        print("[INIT] Loading audio models...")
        self.extractors = [
            LoudnessExtractor(sr),
            SpectralNoveltyExtractor(sr),
            RhythmExtractor(sr),
            SilenceContrastExtractor(sr),
            SemanticEventExtractor(sr)
        ]
        
    @property
    def name(self) -> str:
        return "audio_analyzer"
        
    def get_feature_extractors(self) -> List[BaseFeatureExtractor]:
        return self.extractors
        
    def compute_audio_scores(self, audio_path: str, use_cache: bool = True) -> tuple:
        """Compatibility method for old workers."""
        res = self.analyze(audio_path, use_cache=use_cache)
        return res["timestamps"], res["scores"]
        
    def analyze(self, input_path: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        if use_cache:
            cached = self.cache.load_numpy(input_path)
            if cached:
                print(f"[AUDIO] Using cached analysis for {input_path}")
                return cached
                
        print(f"[AUDIO] Starting deep audio analysis: {input_path}")
        
        # Load audio once
        y, _ = librosa.load(input_path, sr=self.sr, mono=True)
        hop_length = 512
        
        n_frames = len(y) // hop_length + 1
        timestamps = librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=hop_length)
        
        # Extract features
        loudness_ex = next(e for e in self.extractors if e.name == "loudness")
        e_short, e_long = loudness_ex.extract(y, hop_length=hop_length)
        
        novelty_ex = next(e for e in self.extractors if e.name == "spectral_novelty")
        novelty = novelty_ex.extract(y, hop_length=hop_length)
        
        rhythm_ex = next(e for e in self.extractors if e.name == "rhythm_onsets")
        onset_env, rhythm_var, tempo = rhythm_ex.extract(y, hop_length=hop_length)
        
        silence_ex = next(e for e in self.extractors if e.name == "silence_contrast")
        silence_contrast = silence_ex.extract(y, hop_length=hop_length)
        
        semantic_ex = next(e for e in self.extractors if e.name == "semantic_events")
        tensor_y = torch.from_numpy(y).unsqueeze(0)
        semantic = semantic_ex.extract(tensor_y, n_frames_target=n_frames)
        
        # Cross-scale consistency
        window_sizes = [0.5, 1.0, 2.0, 5.0]
        ms_features = multi_scale_window(e_short, window_sizes, hop_length, self.sr)
        
        consistency = np.ones(n_frames)
        if ms_features:
            smoothed = np.array([ms_features[f'{ws}s'] for ws in window_sizes])
            consistency = 1.0 / (np.std(smoothed, axis=0) + 1e-6)
            
        # Ensure all arrays are same length
        arrays = [e_short, e_long, novelty, onset_env, rhythm_var, silence_contrast, semantic, consistency]
        min_len = min(len(a) for a in arrays)
        min_len = min(min_len, len(timestamps))
        
        e_short, e_long, novelty, onset_env, rhythm_var, silence_contrast, semantic, consistency, timestamps = (
            arr[:min_len] for arr in (e_short, e_long, novelty, onset_env, rhythm_var, silence_contrast, semantic, consistency, timestamps)
        )
        
        # Normalize
        norm_arrays = [robust_normalize(arr) for arr in (e_short, e_long, novelty, onset_env, rhythm_var, silence_contrast, semantic, consistency)]
        e_short, e_long, novelty, onset_env, rhythm_var, silence_contrast, semantic, consistency = norm_arrays
        
        # Scoring (distributed 0.12 boundary weight: +0.05 novelty, +0.07 silence)
        scores = (
            0.10 * e_short + 0.08 * e_long + 0.30 * novelty + 0.08 * onset_env +
            0.07 * rhythm_var + 0.17 * silence_contrast +
            0.18 * semantic + 0.02 * consistency
        )
        
        # Expectation violation
        dE = np.diff(scores, prepend=scores[0])
        d2E = np.diff(dE, prepend=dE[0])
        ev = robust_normalize(np.abs(d2E))
        
        scores += 0.15 * ev
        
        # Smoothing
        scores = ema_filter(scores, alpha=0.3)
        scores = robust_normalize(scores)
        
        # Min-max scale to 0-1
        scores = (scores - np.min(scores)) / (np.ptp(scores) + 1e-6)
        
        results = {
            "timestamps": timestamps,
            "scores": scores
        }
        
        if use_cache:
            self.cache.save_numpy(input_path, **results)
            
        return results
