import numpy as np
from src.features.base import BaseFeatureExtractor

class CompositionExtractor(BaseFeatureExtractor):
    """Rule of thirds and central subject scoring."""
    
    @property
    def name(self) -> str:
        return "composition"
        
    @property
    def weight(self) -> float:
        return 0.12
        
    def extract(self, boxes: np.ndarray, frame_shape: tuple, **kwargs) -> float:
        """
        Extract composition score from YOLO bounding boxes.
        
        Args:
            boxes: Bounding boxes from YOLO
            frame_shape: (height, width, channels)
        """
        if len(boxes) == 0:
            return 0.0
            
        h, w = frame_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        third_w, third_h = w / 3, h / 3
        power_points = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        
        best_score = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Rule of thirds score (distance to nearest power point)
            min_dist = min(np.sqrt((box_cx - px)**2 + (box_cy - py)**2) for px, py in power_points)
            rot_score = 1 - (min_dist / (np.sqrt(w**2 + h**2) / 2))
            
            # Center score
            center_dist = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
            center_score = 1 - (center_dist / (np.sqrt(w**2 + h**2) / 2))
            
            # Size score
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            size_score = min(1.0, box_area / (frame_area * 0.4))
            
            combined = 0.5 * rot_score + 0.3 * center_score + 0.2 * size_score
            best_score = max(best_score, combined)
            
        return best_score
