import os
import subprocess
from pathlib import Path
from src.config.settings import AppSettings

class VideoIO:
    """Helper class for video file operations."""
    
    @staticmethod
    def validate_video_path(video_path: str) -> str:
        """Validate and correct video path, checking videos/ folder if needed."""
        settings = AppSettings()
        if not os.path.exists(video_path):
            videos_path = settings.project_root / 'videos' / video_path
            if videos_path.exists():
                print(f"[INFO] Found video in videos/ folder: {videos_path}")
                return str(videos_path)
            else:
                raise FileNotFoundError(
                    f"Video file not found: {video_path}\n"
                    f"Also checked: {videos_path}\n"
                    f"Please provide full path or place video in videos/ folder"
                )
        return video_path
        
    @staticmethod
    def extract_audio_from_video(video_path: str, force: bool = False) -> str:
        """Extract audio track from video using FFmpeg."""
        settings = AppSettings()
        audio_path = str(Path(video_path).with_suffix('.wav'))
        
        if os.path.exists(audio_path) and not force:
            print(f"[AUDIO] Audio file already exists: {audio_path}")
            return audio_path
            
        print(f"[AUDIO] Extracting audio from video...")
        
        ffmpeg_path = settings.project_root / 'bin' / 'ffmpeg.exe'
        if not ffmpeg_path.exists():
            raise RuntimeError(f"Standalone FFmpeg not found at {ffmpeg_path}. Please install it.")
            
        cmd = [
            str(ffmpeg_path), '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(settings.audio_sample_rate),
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            print(f"[AUDIO] Audio extracted to: {audio_path}\n")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg failed to extract audio: {error_msg}")
            
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio extraction failed: {audio_path} was not created")
            
        return audio_path
