import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from src.analyzers.base import BaseAnalyzer
from src.features.base import BaseFeatureExtractor
from src.features.motion import MotionExtractor
from src.features.visual_surprise import VisualSurpriseExtractor
from src.features.composition import CompositionExtractor
from src.features.thumbnailability import ThumbnailabilityExtractor

from src.utils.normalization import robust_normalize, ema_filter
from src.utils.cache import CacheManager
from src.config.settings import AppSettings

class VideoAnalyzer(BaseAnalyzer):
    """Orchestrates video feature extraction and scoring."""
    
    def __init__(self):
        self.settings = AppSettings()
        self.cache = CacheManager(namespace="video")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[INIT] Loading vision models... (device: {self.device})")
        
        # Load YOLO
        base_dir = self.settings.project_root
        model_path = base_dir / "models" / "yolov8n.pt"
        if model_path.exists():
            self.yolo = YOLO(str(model_path))
        else:
            self.yolo = YOLO("yolov8n.pt")
            os.makedirs(base_dir / "models", exist_ok=True)
            
        # Load CLIP
        hf_cache_dir = base_dir / ".cache" / "huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(hf_cache_dir)).to(self.device)
        if self.device == "cuda":
            self.clip_model = self.clip_model.half()
            
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True, cache_dir=str(hf_cache_dir)
        )
        
        # Extractors
        self.extractors = [
            MotionExtractor(),
            VisualSurpriseExtractor(),
            CompositionExtractor(),
            ThumbnailabilityExtractor()
        ]
        
    @property
    def name(self) -> str:
        return "video_analyzer"
        
    def get_feature_extractors(self) -> List[BaseFeatureExtractor]:
        return self.extractors
        
    def compute_visual_scores(self, video_path: str, target_fps: int = 2, use_cache: bool = True) -> tuple:
        """Compatibility method for old workers."""
        res = self.analyze(video_path, target_fps=target_fps, use_cache=use_cache)
        return res["timestamps"], res["scores"]
        
    def analyze(self, input_path: str, target_fps: int = 2, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        cache_key = f"{input_path}_{target_fps}"
        if use_cache:
            cached = self.cache.load_numpy(cache_key)
            if cached:
                print(f"[VIDEO] Using cached analysis for {input_path}")
                return cached
                
        print(f"[VIDEO] Starting deep visual analysis: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / target_fps))
        
        M, S, C, R, I, T, SB, AP = [], [], [], [], [], [], [], []
        timestamps = []
        flow_signal = []
        dominant_history = []
        prev_feats = []
        
        motion_ex = next(e for e in self.extractors if e.name == "motion")
        surprise_ex = next(e for e in self.extractors if e.name == "visual_surprise")
        comp_ex = next(e for e in self.extractors if e.name == "composition")
        thumb_ex = next(e for e in self.extractors if e.name == "thumbnailability")
        
        batch_size = 16
        frames_batch, grays_batch, ts_batch = [], [], []
        
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Video is empty")
            
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        current_frame = 0
        pbar = tqdm(total=total_frames // frame_interval, desc="[VIDEO] Processing frames")
        
        prev_dominant = None
        
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                frames_batch.append(frame)
                grays_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                ts_batch.append(current_frame / fps)
                
                if len(frames_batch) >= batch_size:
                    prev_dominant, prev_gray = self._process_batch(
                        frames_batch, grays_batch, ts_batch,
                        M, S, C, I, T, SB, AP, timestamps, flow_signal,
                        dominant_history, prev_feats, prev_gray, prev_dominant,
                        motion_ex, surprise_ex, comp_ex, thumb_ex
                    )
                    frames_batch, grays_batch, ts_batch = [], [], []
                    pbar.update(batch_size)
                    
            current_frame += 1
            
        if len(frames_batch) > 0:
            self._process_batch(
                frames_batch, grays_batch, ts_batch,
                M, S, C, I, T, SB, AP, timestamps, flow_signal,
                dominant_history, prev_feats, prev_gray, prev_dominant,
                motion_ex, surprise_ex, comp_ex, thumb_ex
            )
            pbar.update(len(frames_batch))
            
        cap.release()
        pbar.close()
        
        R = [np.std(np.abs(np.diff(flow_signal))[-10:]) if i >= 10 else 0 for i in range(len(flow_signal))]
        
        AP = ema_filter(np.array(dominant_history), alpha=0.3)
        AP = 1 - AP
        
        M = robust_normalize(M)
        S = robust_normalize(S)
        C = robust_normalize(C)
        R = robust_normalize(R)
        I = robust_normalize(I)
        T = robust_normalize(T)
        SB = robust_normalize(SB)
        AP = robust_normalize(AP)
        
        raw_scores = np.array([
            0.20 * M[i] + 0.20 * S[i] + 0.12 * C[i] + 
            0.08 * R[i] + 0.08 * I[i] + 0.12 * T[i] + 
            0.05 * AP[i] + 0.05 * SB[i]
            for i in range(len(M))
        ])
        
        dE = np.diff(raw_scores, prepend=raw_scores[0])
        d2E = np.diff(dE, prepend=dE[0])
        EV = robust_normalize(np.abs(d2E))
        
        scores = raw_scores + 0.15 * EV
        scores = ema_filter(scores, alpha=0.3)
        scores = robust_normalize(scores)
        scores = (scores - np.min(scores)) / (np.ptp(scores) + 1e-6)
        
        timestamps = np.array(timestamps)
        results = {"timestamps": timestamps, "scores": scores}
        
        if use_cache:
            self.cache.save_numpy(cache_key, **results)
            
        return results

    def _process_batch(self, frames, grays, ts, M, S, C, I, T, SB, AP, timestamps, flow_signal, dominant_history, prev_feats, prev_gray, prev_dominant, motion_ex, surprise_ex, comp_ex, thumb_ex):
        m_scores, current_prev_gray, sb_scores = motion_ex.extract(grays, prev_gray)
        M.extend(m_scores)
        flow_signal.extend(m_scores)
        SB.extend(sb_scores)
        
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        inputs = self.clip_processor(images=pil_images, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device, dtype=torch.float16) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            feats_batch = (outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs).float().cpu().numpy()
            
        for feat in feats_batch:
            feat_flat = feat.flatten()
            S.append(surprise_ex.extract(feat_flat, prev_feats))
            prev_feats.append(feat_flat)
            
        yolo_results = self.yolo(frames, verbose=False, device=self.device)
        
        current_prev_dominant = prev_dominant if len(dominant_history) > 0 else None
        
        for i, res in enumerate(yolo_results):
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            classes = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
            frame_shape = frames[i].shape
            
            C.append(comp_ex.extract(boxes, frame_shape))
            I.append(len(boxes))
            T.append(thumb_ex.extract(frames[i], boxes, classes))
            
            # Simple dominant tracking
            if len(boxes) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                idx = np.argmax(areas)
                curr_dominant = boxes[idx]
                if current_prev_dominant is not None:
                    # IoU approximation
                    xA = max(curr_dominant[0], current_prev_dominant[0])
                    yA = max(curr_dominant[1], current_prev_dominant[1])
                    xB = min(curr_dominant[2], current_prev_dominant[2])
                    yB = min(curr_dominant[3], current_prev_dominant[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    persistence = interArea / areas[idx]
                else:
                    persistence = 1.0
            else:
                curr_dominant = None
                persistence = 0.0
                
            dominant_history.append(persistence)
            current_prev_dominant = curr_dominant
            timestamps.append(ts[i])
            
        return current_prev_dominant, current_prev_gray
