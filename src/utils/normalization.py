import numpy as np
import scipy.signal as signal
from scipy.signal import convolve

def robust_normalize(x: np.ndarray) -> np.ndarray:
    """Robust normalization using median absolute deviation."""
    x = np.array(x)
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return np.clip((x - med) / mad, -3, 3)

def ema_filter(signal_data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average for temporal smoothing."""
    if len(signal_data) == 0:
        return np.array([])
    b = [alpha]
    a = [1, -(1 - alpha)]
    ema = signal.lfilter(b, a, signal_data)
    return ema

def temporal_envelope(signal_data: np.ndarray, window: int) -> np.ndarray:
    """Compute slow-moving RMS envelope for perceived excitement."""
    if len(signal_data) == 0:
        return np.array([])
    squared = signal_data ** 2
    rms = np.sqrt(convolve(squared, np.ones(window)/window, mode='same'))
    return rms

def multi_scale_window(signal_data: np.ndarray, window_sizes_sec: list, hop_length: int, sr: int) -> dict:
    """Compute multi-scale features across different time windows."""
    features = {}
    if len(signal_data) == 0:
        return features
        
    for ws in window_sizes_sec:
        ws_frames = int(ws * sr / hop_length)
        if ws_frames > 1:
            kernel = np.ones(ws_frames) / ws_frames
            smoothed = convolve(signal_data, kernel, mode='same')
            features[f'{ws}s'] = smoothed
    return features
