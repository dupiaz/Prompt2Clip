import numpy as np
from scipy.signal import find_peaks

class CandidateClipGenerator:
    """Generates candidate clips using peak detection and transcript boundaries."""
    
    def generate(self, timestamps, scores, transcript_segments, min_duration=5, max_duration=60, prominence=0.5, min_distance=10):
        print("=" * 60)
        print("PHASE 3: CANDIDATE CLIP GENERATION")
        print("=" * 60 + "\n")
        
        peaks, _ = find_peaks(scores, prominence=prominence, distance=min_distance)
        candidates = []
        
        for idx, peak_idx in enumerate(peaks):
            peak_time = timestamps[peak_idx]
            initial_start = max(0, peak_time - min_duration / 2)
            initial_end = min(timestamps[-1], peak_time + min_duration / 2)
            
            aligned_clip = self._align_to_transcript_boundaries(
                initial_start, initial_end, transcript_segments, min_duration, max_duration
            )
            
            if aligned_clip:
                start_idx = np.searchsorted(timestamps, aligned_clip["start"])
                end_idx = np.searchsorted(timestamps, aligned_clip["end"])
                
                candidates.append({
                    "clip_id": idx + 1,
                    "start": aligned_clip["start"],
                    "end": aligned_clip["end"],
                    "duration": aligned_clip["end"] - aligned_clip["start"],
                    "transcript": aligned_clip["transcript"],
                    "peak_time": peak_time,
                    "avg_score": np.mean(scores[start_idx:end_idx]),
                    "peak_score": scores[peak_idx]
                })
                
        candidates.sort(key=lambda x: x["peak_score"], reverse=True)
        return candidates
        
    def _align_to_transcript_boundaries(self, start_time, end_time, transcript_segments, min_duration, max_duration):
        overlapping = [
            seg for seg in transcript_segments
            if not (seg["end"] < start_time or seg["start"] > end_time)
        ]
        
        if not overlapping:
            return None
            
        aligned_start = overlapping[0]["start"]
        aligned_end = overlapping[-1]["end"]
        duration = aligned_end - aligned_start
        
        if duration < min_duration:
            prev_idx = transcript_segments.index(overlapping[0]) - 1
            if prev_idx >= 0:
                aligned_start = transcript_segments[prev_idx]["start"]
                
        if duration > max_duration:
            cumulative = 0
            valid = []
            for seg in overlapping:
                seg_dur = seg["end"] - seg["start"]
                if cumulative + seg_dur <= max_duration:
                    valid.append(seg)
                    cumulative += seg_dur
                else:
                    break
            if valid:
                aligned_start = valid[0]["start"]
                aligned_end = valid[-1]["end"]
            else:
                return None
                
        transcript_text = " ".join([seg["text"] for seg in overlapping])
        
        return {
            "start": aligned_start,
            "end": aligned_end,
            "transcript": transcript_text
        }
