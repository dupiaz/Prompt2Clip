import cv2
import numpy as np
from src.features.base import BaseFeatureExtractor

class MotionExtractor(BaseFeatureExtractor):
    """Calculates optical flow magnitude between consecutive frames."""
    
    @property
    def name(self) -> str:
        return "motion"
        
    @property
    def weight(self) -> float:
        return 0.20
        
    def extract(self, frames: list, prev_gray: np.ndarray = None, **kwargs) -> tuple:
        """
        Extracts motion scores and shot boundary scores.
        
        Args:
            frames: list of grayscale frames.
            prev_gray: The last frame from the previous batch.
            
        Returns:
            Tuple of (motion_scores, last_gray_frame, shot_boundary_scores)
        """
        if not frames:
            return np.array([]), prev_gray, np.array([])
            
        if prev_gray is None:
            prev_gray = frames[0]
            
        M = []
        SB = []
        
        current_prev_gray = prev_gray
        for gray in frames:
            # Shot boundary diff
            frame_diff = np.mean(np.abs(gray.astype(float) - current_prev_gray.astype(float)))
            SB.append(frame_diff)
            
            # Optical Flow
            small_prev = cv2.resize(current_prev_gray, (160, 90), interpolation=cv2.INTER_LINEAR)
            small_curr = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_LINEAR)
            
            flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None,
                                                0.5, 1, 8, 2, 5, 1.1, 0)
            mag = np.linalg.norm(flow, axis=2).mean()
            M.append(mag)
            current_prev_gray = gray
            
        return np.array(M), current_prev_gray, np.array(SB)
