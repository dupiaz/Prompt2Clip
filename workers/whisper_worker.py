"""
Whisper Worker — Isolated transcription process for clipz_whisper environment.
Receives audio path via CLI, outputs transcription as JSON.

Usage:
    conda run -n clipz_whisper python workers/whisper_worker.py <audio_path> [--output <json_path>] [--model-size <size>]
"""

import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment config
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'envs', 'whisper.env'))


def main():
    parser = argparse.ArgumentParser(description='Whisper transcription worker')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--output', '-o', default=None, help='Output JSON path (default: .cache/transcription/result.json)')
    parser.add_argument('--model-size', default=None, help='Whisper model size (default: from whisper.env)')
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.audio_path):
        print(f"[ERROR] Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # Get config from env
    model_size = args.model_size or os.getenv('WHISPER_MODEL_SIZE', 'medium')
    device = os.getenv('WHISPER_DEVICE', 'cuda')
    compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        cache_dir = os.path.join('.cache', 'transcription')
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_key = f"{args.audio_path}_{model_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        output_path = os.path.join(cache_dir, f"transcript_{cache_hash}.json")

    # Check cache
    if os.path.exists(output_path):
        print(f"[WHISPER_WORKER] Using cached transcription: {output_path}")
        sys.exit(0)

    print(f"[WHISPER_WORKER] Starting transcription...")
    print(f"[WHISPER_WORKER] Model: {model_size} | Device: {device} | Compute: {compute_type}")

    # Load DLLs for CUDA on Windows
    if os.name == 'nt':
        import site
        for site_pkg in site.getsitepackages():
            for pkg in ['cublas', 'cudnn', 'cuda_nvrtc', 'cuda_runtime']:
                path = os.path.join(site_pkg, 'nvidia', pkg, 'bin')
                if os.path.exists(path):
                    try:
                        os.add_dll_directory(path)
                    except Exception:
                        pass
                    os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')

    from src.analyzers.transcription_analyzer import TranscriptionAnalyzer

    analyzer = TranscriptionAnalyzer()
    analyzer.settings.whisper_device = device
    analyzer.settings.whisper_compute_type = compute_type
    
    sentences = analyzer.transcribe_with_timestamps(
        args.audio_path,
        model_size=model_size,
        use_cache=False
    )

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)

    print(f"[WHISPER_WORKER] Transcription complete: {len(sentences)} segments")
    print(f"[WHISPER_WORKER] Output saved to: {output_path}")


if __name__ == '__main__':
    main()
