import cv2
import numpy as np
from src.features.base import BaseFeatureExtractor

try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False


class ThumbnailabilityExtractor(BaseFeatureExtractor):
    """Scores frame for potential thumbnail usage (faces, clarity)."""
    
    def __init__(self):
        if HAS_DLIB:
            try:
                self.detector = dlib.get_frontal_face_detector()
            except Exception as e:
                print(f"[WARNING] dlib face detector init failed: {e}")
                self.detector = None
        else:
            self.detector = None
            
    @property
    def name(self) -> str:
        return "thumbnailability"
        
    @property
    def weight(self) -> float:
        return 0.12
        
    def extract(self, frame: np.ndarray, boxes: np.ndarray, classes: np.ndarray, **kwargs) -> float:
        """
        Extract thumbnailability score.
        
        Args:
            frame: Raw BGR frame
            boxes: YOLO boxes
            classes: YOLO classes
        """
        score = 0.0
        
        # 1. Face detection (high weight for thumbnails)
        if self.detector is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 1)
                if len(faces) > 0:
                    score += 0.4 + min(0.2, len(faces) * 0.05)
                    
                    # Check face size
                    max_face_area = max((f.right() - f.left()) * (f.bottom() - f.top()) for f in faces)
                    frame_area = frame.shape[0] * frame.shape[1]
                    if max_face_area > frame_area * 0.05:  # Prominent face
                        score += 0.2
            except Exception as e:
                pass
        
        # 2. Object detection (person = 0 in COCO)
        if len(classes) > 0:
            persons = sum(1 for c in classes if c == 0)
            if persons > 0:
                score += 0.2 + min(0.1, persons * 0.02)
                
        # 3. Contrast / Clarity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        clarity_score = min(1.0, std_dev / 80.0)  # Normalize
        score += 0.2 * clarity_score
        
        return min(1.0, score)
