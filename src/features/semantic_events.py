import torch
import numpy as np
from src.features.base import BaseFeatureExtractor

try:
    from torch_vggish_yamnet import yamnet as torch_yamnet
    from torch_vggish_yamnet.input_proc import WaveformToInput
    HAS_YAMNET = True
except ImportError:
    HAS_YAMNET = False


class SemanticEventExtractor(BaseFeatureExtractor):
    """Semantic Audio Events - laughter, applause, screams (using YAMNet)."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.yamnet_model = None
        self.yamnet_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if HAS_YAMNET:
            try:
                self.yamnet_model = torch_yamnet.yamnet(pretrained=True)
                self.yamnet_model.eval()
                self.yamnet_model.to(self.yamnet_device)
                self.yamnet_converter = WaveformToInput()
            except Exception as e:
                print(f"[WARNING] Failed to load YAMNet: {e}")
                self.yamnet_model = None
                
    @property
    def name(self) -> str:
        return "semantic_events"
        
    @property
    def weight(self) -> float:
        return 0.18
        
    def extract(self, y_16k_tensor: torch.Tensor, n_frames_target: int, **kwargs) -> np.ndarray:
        """
        Extracts semantic events.
        
        Args:
            y_16k_tensor: PyTorch tensor of shape [1, samples]
            n_frames_target: Expected output size to interpolate to
        """
        if self.yamnet_model is None:
            return np.zeros(n_frames_target)
            
        try:
            waveform_tensor = y_16k_tensor.squeeze(0).float()
            in_tensor = self.yamnet_converter(waveform_tensor, 16000)
            in_tensor = in_tensor.to(self.yamnet_device)
            
            with torch.no_grad():
                scores, embeddings = self.yamnet_model(in_tensor)
            scores = scores.cpu().numpy()
            
            excitement_classes = {
                321: 2.0,   # Laughter
                138: 1.8,   # Applause
                139: 1.8,   # Cheering
                422: 1.5,   # Screaming
                137: 1.0    # Music
            }
            
            excitement = np.zeros(scores.shape[0])
            for class_idx, weight in excitement_classes.items():
                excitement += scores[:, class_idx] * weight
                
            excitement_interp = np.interp(
                np.linspace(0, len(excitement), n_frames_target),
                np.arange(len(excitement)),
                excitement
            )
            return excitement_interp
        except Exception as e:
            print(f"[WARNING] Semantic event extraction failed: {e}")
            return np.zeros(n_frames_target)
