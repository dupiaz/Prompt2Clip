"""
Audio Worker — Isolated audio analysis process for clipz_audio environment.
Receives audio path via CLI, outputs audio excitement scores as JSON.

Usage:
    conda run -n clipz_audio python workers/audio_worker.py <audio_path> [--output <json_path>]
"""

import sys
import os
import json
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment config
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'envs', 'audio.env'))


def main():
    parser = argparse.ArgumentParser(description='Audio analysis worker')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--output', '-o', default=None, help='Output JSON path')
    parser.add_argument('--use-cache', action='store_true', help='Use cached results if available')
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.audio_path):
        print(f"[ERROR] Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        cache_dir = os.path.join('.cache', 'audio_worker')
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_hash = hashlib.md5(args.audio_path.encode()).hexdigest()
        output_path = os.path.join(cache_dir, f"audio_{cache_hash}.json")

    # Check cache
    if args.use_cache and os.path.exists(output_path):
        print(f"[AUDIO_WORKER] Using cached results: {output_path}")
        sys.exit(0)

    print(f"[AUDIO_WORKER] Starting audio analysis...")
    print(f"[AUDIO_WORKER] Audio: {args.audio_path}")

    sr = int(os.getenv('SAMPLE_RATE', '16000'))

    from src.analyzers.audio_analyzer import AudioAnalyzer as ClipAudio

    detector = ClipAudio(sr=sr)
    timestamps, scores = detector.compute_audio_scores(
        args.audio_path,
        use_cache=args.use_cache
    )

    # Convert numpy arrays to lists for JSON serialization
    result = {
        "timestamps": timestamps.tolist() if isinstance(timestamps, np.ndarray) else list(timestamps),
        "scores": scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
    }

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    print(f"[AUDIO_WORKER] Analysis complete: {len(result['timestamps'])} frames")
    print(f"[AUDIO_WORKER] Output saved to: {output_path}")


if __name__ == '__main__':
    main()
