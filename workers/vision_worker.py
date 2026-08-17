"""
Vision Worker — Isolated video analysis process for clipz_vision environment.
Receives video path via CLI, outputs visual excitement scores as JSON.

Usage:
    conda run -n clipz_vision python workers/vision_worker.py <video_path> --query <query> [--output <json_path>] [--target-fps <fps>]
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment config
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'envs', 'vision.env'))


def main():
    parser = argparse.ArgumentParser(description='Vision analysis worker')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--query', required=True, help='Semantic search query for CLIP')
    parser.add_argument('--output', '-o', default=None, help='Output JSON path')
    parser.add_argument('--target-fps', type=int, default=2, help='Analysis FPS (default: 2)')
    parser.add_argument('--use-cache', action='store_true', help='Use cached results if available')
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        cache_dir = os.path.join('.cache', 'vision_worker')
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_key = f"{args.video_path}_{args.target_fps}_{args.query}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        output_path = os.path.join(cache_dir, f"vision_{cache_hash}.json")

    # Check cache
    if args.use_cache and os.path.exists(output_path):
        print(f"[VISION_WORKER] Using cached results: {output_path}")
        sys.exit(0)

    print(f"[VISION_WORKER] Starting video analysis...")
    print(f"[VISION_WORKER] Video: {args.video_path} | FPS: {args.target_fps} | Query: {args.query}")

    from transformers import CLIPProcessor, CLIPModel
    
    model_name = os.getenv('CLIP_MODEL', 'openai/clip-vit-base-patch32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VISION_WORKER] Loading CLIP model ({model_name}) on {device}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Pre-compute text features
    dummy_image = Image.new("RGB", (224, 224))
    text_inputs = processor(text=[args.query], images=dummy_image, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model(**text_inputs).text_embeds
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"[VISION_WORKER] Processing video frames...")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = int(round(fps / args.target_fps))
    if frame_interval < 1:
        frame_interval = 1

    timestamps = []
    scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            
            # Convert BGR (cv2) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Extract image features
            image_inputs = processor(text=["dummy"], images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model(**image_inputs).image_embeds
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = (text_features @ image_features.T).item()
                # CLIP similarities are typically in range ~ 0.15 - 0.35. Normalize loosely to [0,1] 
                # or just pass raw score. We'll pass raw score and let pipeline normalize.
                # But to meet contract [0,1], we clamp. Actually, dot product of normalized vectors is [-1, 1].
                score = max(0.0, min(1.0, similarity))
                
            timestamps.append(timestamp)
            scores.append(score)

        frame_count += 1

    cap.release()

    result = {
        "timestamps": timestamps,
        "scores": scores,
        "query": args.query,
        "model": model_name
    }

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    print(f"[VISION_WORKER] Analysis complete: {len(result['timestamps'])} frames")
    print(f"[VISION_WORKER] Output saved to: {output_path}")


if __name__ == '__main__':
    main()
