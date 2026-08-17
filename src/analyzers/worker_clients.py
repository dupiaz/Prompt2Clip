import os
import sys
import json
import uuid
import tempfile
import subprocess
import numpy as np
from typing import Dict, Any, List

from src.analyzers.base import BaseAnalyzer
from src.features.base import BaseFeatureExtractor

class WorkerExecutionError(Exception):
    """Custom exception for worker failures with detailed stderr."""
    pass

class BaseWorkerClient(BaseAnalyzer):
    """
    Base client for delegating analysis to a subprocess running in an isolated Conda environment.
    Prevents zombie processes, handles silent failures, and prevents memory leaks.
    """
    
    def __init__(self, env_name: str, worker_script: str):
        self.env_name = env_name
        self.worker_script = worker_script
        
    def get_feature_extractors(self) -> List[BaseFeatureExtractor]:
        # Feature extraction is handled by the worker process
        return []
        
    def _run_subprocess(self, cmd: List[str]):
        """Run the command safely and capture all stderr for debugging."""
        try:
            # Using shell=True for Windows compatibility with conda
            if os.name == 'nt' and isinstance(cmd, list):
                cmd_str = subprocess.list2cmdline(cmd)
            else:
                cmd_str = cmd
                
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                shell=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Print stdout to the main console for transparency
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"[{self.name.upper()}] {line}")
                        
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error (No stderr output)"
                raise WorkerExecutionError(
                    f"Worker '{self.name}' in env '{self.env_name}' failed with code {result.returncode}.\n"
                    f"--- STDERR ---\n{error_msg}\n--------------"
                )
        except Exception as e:
            if not isinstance(e, WorkerExecutionError):
                raise WorkerExecutionError(f"Failed to execute worker '{self.name}': {e}")
            raise

class AudioWorkerClient(BaseWorkerClient):
    @property
    def name(self) -> str:
        return "audio_analyzer"
        
    def __init__(self):
        super().__init__(env_name="clipz_audio", worker_script="workers/audio_worker.py")
        
    def analyze(self, input_path: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        output_file = os.path.join(tempfile.gettempdir(), f"audio_worker_{uuid.uuid4().hex}.json")
        
        cmd = [
            "conda", "run", "-n", self.env_name, 
            "python", self.worker_script, 
            input_path, 
            "--output", output_file
        ]
        
        # Audio worker handles its own internal cache, so we just let it run
        if use_cache:
            cmd.append("--use-cache")
            
        print(f"[{self.name.upper()}] Spawning isolated process in {self.env_name}...")
        self._run_subprocess(cmd)
        
        # Validate data contract
        if not os.path.exists(output_file):
            raise WorkerExecutionError(f"Worker did not produce output file: {output_file}")
            
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Clean up temporary file to prevent disk bloat
        try:
            os.remove(output_file)
        except OSError:
            pass
            
        # Enforce contract
        if "timestamps" not in data or "scores" not in data:
            raise WorkerExecutionError(f"Invalid data contract from {self.name}: missing keys")
            
        return {
            "timestamps": np.array(data["timestamps"], dtype=np.float32),
            "scores": np.array(data["scores"], dtype=np.float32)
        }

class VideoWorkerClient(BaseWorkerClient):
    @property
    def name(self) -> str:
        return "video_analyzer"
        
    def __init__(self):
        super().__init__(env_name="clipz_vision", worker_script="workers/vision_worker.py")
        
    def analyze(self, input_path: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        output_file = os.path.join(tempfile.gettempdir(), f"video_worker_{uuid.uuid4().hex}.json")
        
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("VideoWorkerClient requires 'query' kwarg for CLIP model")

        cmd = [
            "conda", "run", "-n", self.env_name, 
            "python", self.worker_script, 
            input_path, 
            "--output", output_file,
            "--query", query
        ]
        
        if use_cache:
            cmd.append("--use-cache")
            
        print(f"[{self.name.upper()}] Spawning isolated process in {self.env_name}...")
        self._run_subprocess(cmd)
        
        if not os.path.exists(output_file):
            raise WorkerExecutionError(f"Worker did not produce output file: {output_file}")
            
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        try:
            os.remove(output_file)
        except OSError:
            pass
            
        if "timestamps" not in data or "scores" not in data:
            raise WorkerExecutionError(f"Invalid data contract from {self.name}: missing keys")
            
        return {
            "timestamps": np.array(data["timestamps"], dtype=np.float32),
            "scores": np.array(data["scores"], dtype=np.float32)
        }

class TranscriptionWorkerClient(BaseWorkerClient):
    @property
    def name(self) -> str:
        return "transcription_analyzer"
        
    def __init__(self):
        super().__init__(env_name="clipz_whisper", worker_script="workers/whisper_worker.py")
        
    def analyze(self, input_path: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        output_file = os.path.join(tempfile.gettempdir(), f"whisper_worker_{uuid.uuid4().hex}.json")
        
        cmd = [
            "conda", "run", "-n", self.env_name, 
            "python", self.worker_script, 
            input_path, 
            "--output", output_file
        ]
        
        if use_cache:
            cmd.append("--use-cache")
            
        print(f"[{self.name.upper()}] Spawning isolated process in {self.env_name}...")
        self._run_subprocess(cmd)
        
        if not os.path.exists(output_file):
            raise WorkerExecutionError(f"Worker did not produce output file: {output_file}")
            
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        try:
            os.remove(output_file)
        except OSError:
            pass
            
        if "segments" not in data:
            raise WorkerExecutionError(f"Invalid data contract from {self.name}: missing segments")
            
        return {
            "segments": data["segments"]
        }
