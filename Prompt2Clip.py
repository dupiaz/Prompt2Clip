"""
Backward compatibility wrapper for legacy imports.
"""
from src.main import main
from src.analyzers.worker_clients import AudioWorkerClient as ClipAudio
from src.analyzers.worker_clients import VideoWorkerClient as ClipVideo
from src.analyzers.worker_clients import TranscriptionWorkerClient as Transcriber
from src.llm.aibox_client import AiBoxLLMClient as LLM
from src.core.pipeline import ClipExtractor

if __name__ == "__main__":
    main()
