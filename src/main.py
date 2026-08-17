import argparse
from datetime import datetime
from pathlib import Path

from src.analyzers.worker_clients import AudioWorkerClient, VideoWorkerClient, TranscriptionWorkerClient
from src.llm.aibox_client import AiBoxLLMClient
from src.exporters.ffmpeg_exporter import FFmpegExporter
from src.core.pipeline import ClipExtractor
from src.config.settings import AppSettings

def main():
    parser = argparse.ArgumentParser(description="Extract viral clips from video using AI analysis (OOP refactored)")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--query", default="give me 10 interesting clips", help="Query for clip selection")
    parser.add_argument("--audio-weight", type=float, default=0.5, help="Weight for audio scores (0-1)")
    parser.add_argument("--video-weight", type=float, default=0.5, help="Weight for video scores (0-1)")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS for video analysis")
    parser.add_argument("--min-duration", type=int, default=5, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum clip duration in seconds")
    parser.add_argument("--output-dir", default="output", help="Output directory for clips")
    parser.add_argument("--no-export", action="store_true", help="Skip exporting video files")
    
    args = parser.parse_args()
    
    # Update settings
    settings = AppSettings()
    settings.audio_weight = args.audio_weight
    settings.video_weight = args.video_weight
    if args.output_dir != "output":
        settings.output_dir = args.output_dir
        
    # Wire dependencies
    audio_analyzer = AudioWorkerClient()
    video_analyzer = VideoWorkerClient()
    transcription_analyzer = TranscriptionWorkerClient()
    llm_client = AiBoxLLMClient()
    exporter = FFmpegExporter()
    
    extractor = ClipExtractor(
        audio_analyzer=audio_analyzer,
        video_analyzer=video_analyzer,
        transcription_analyzer=transcription_analyzer,
        llm_client=llm_client,
        exporter=exporter
    )
    
    # Process
    try:
        results = extractor.process(
            video_path=args.video_path,
            user_query=args.query,
            target_fps=args.fps,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            export=not args.no_export
        )
        
        print("\n📊 FINAL CLIPS:")
        for i, clip in enumerate(results["final_clips"], 1):
            print(f"\nClip {i}:")
            print(f"  Time: {clip['start']:.1f}s - {clip['end']:.1f}s ({clip['duration']:.1f}s)")
            print(f"  Transcript: {clip.get('transcript', 'N/A')[:100]}...")
            if 'llm_interest_score' in clip:
                print(f"  Interest Score: {clip['llm_interest_score']}/10")
            if 'reason' in clip:
                print(f"  Reason: {clip['reason']}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
