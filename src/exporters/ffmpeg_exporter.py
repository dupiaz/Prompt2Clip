import os
import subprocess
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List

from src.exporters.base import BaseExporter
from src.config.settings import AppSettings
from src.utils.cache import CacheManager

class FFmpegExporter(BaseExporter):
    """Exports video clips using FFmpeg subprocess."""
    
    def __init__(self):
        self.settings = AppSettings()
        self.cache = CacheManager(namespace="metadata")
        
    @property
    def name(self) -> str:
        return "ffmpeg"
        
    def export(self, video_path: str, clips: list, output_dir: str = None, **kwargs) -> List[str]:
        if not clips:
            print("[EXPORT] No clips to export")
            return []
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_dir:
            output_dir = self.settings.project_root / self.settings.output_dir / f"clips_{timestamp}"
            
        clips_dir = Path(output_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        failed_clips = []
        
        print(f"[EXPORT] Extracting {len(clips)} clips from video...")
        
        ffmpeg_path = self.settings.project_root / 'bin' / 'ffmpeg.exe'
        if not ffmpeg_path.exists():
            raise RuntimeError(f"Standalone FFmpeg not found at {ffmpeg_path}")
            
        for idx, clip in enumerate(tqdm(clips, desc="Exporting clips"), 1):
            output_file = clips_dir / f"clip_{idx:03d}.mp4"
            
            try:
                cmd = [
                    str(ffmpeg_path),
                    '-i', str(os.path.abspath(video_path)),
                    '-ss', str(clip['start']),
                    '-to', str(clip['end']),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-y',
                    str(os.path.abspath(output_file))
                ]
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if output_file.exists() and output_file.stat().st_size > 0:
                    exported_files.append(str(output_file))
                    
                    clip_metadata = {
                        "clip_number": idx,
                        "video_file": str(output_file),
                        "start_time": clip['start'],
                        "end_time": clip['end'],
                        "duration": clip['duration'],
                        "transcript": clip.get('transcript', ''),
                        "interest_score": clip.get('llm_interest_score', 0),
                        "reason": clip.get('reason', ''),
                        "tags": clip.get('tags', [])
                    }
                    self.cache.save_json(f"clip_{idx:03d}_{timestamp}", clip_metadata)
                else:
                    failed_clips.append(idx)
                    print(f"\n✗ Clip {idx} failed: File not created or empty")
                    
            except subprocess.CalledProcessError as e:
                failed_clips.append(idx)
                print(f"\n✗ Clip {idx} failed: {e.stderr[-500:] if e.stderr else str(e)}")
            except Exception as e:
                failed_clips.append(idx)
                print(f"\n✗ Clip {idx} failed: {str(e)}")
                
        print(f"\n{'=' * 60}")
        print(f"[EXPORT] ✓ Successfully exported {len(exported_files)} clips")
        if failed_clips:
            print(f"[EXPORT] ✗ Failed clips: {failed_clips}")
        print(f"[EXPORT] Output folder: {clips_dir}")
        print(f"{'=' * 60}\n")
        
        return exported_files
