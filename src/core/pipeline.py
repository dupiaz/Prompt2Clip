from typing import Dict, Any

from src.core.base import BasePipeline
from src.core.signal_fusion import SignalFusion
from src.core.clip_generator import CandidateClipGenerator

from src.analyzers.base import BaseAnalyzer
from src.llm.base import BaseLLMClient
from src.exporters.base import BaseExporter
from src.llm.prompt_builder import PromptBuilder
from src.llm.response_parser import LLMResponseParser
from src.utils.video_io import VideoIO

class ClipExtractor(BasePipeline):
    """Main pipeline orchestrator via Dependency Injection."""
    def __init__(self, 
                 audio_analyzer: BaseAnalyzer,
                 video_analyzer: BaseAnalyzer,
                 transcription_analyzer: BaseAnalyzer,
                 llm_client: BaseLLMClient,
                 exporter: BaseExporter):
        self.audio_analyzer = audio_analyzer
        self.video_analyzer = video_analyzer
        self.transcription_analyzer = transcription_analyzer
        self.llm_client = llm_client
        self.exporter = exporter
        
        self.fusion = SignalFusion()
        self.generator = CandidateClipGenerator()
        
    def process(self, video_path: str, user_query: str = "give me 5 interesting clips", **kwargs) -> Dict[str, Any]:
        video_path = VideoIO.validate_video_path(video_path)
        audio_path = VideoIO.extract_audio_from_video(video_path)
        
        # 1. Feature Extraction
        print("=" * 60)
        print("PHASE 1: FEATURE EXTRACTION")
        print("=" * 60 + "\n")
        
        audio_data = self.audio_analyzer.analyze(audio_path)
        video_data = self.video_analyzer.analyze(video_path)
        transcript_data = self.transcription_analyzer.analyze(audio_path)
        transcript_segments = transcript_data["segments"]
        
        # 2. Fusion
        unified_ts, combined_scores, _, _ = self.fusion.fuse(audio_data, video_data)
        
        # 3. Candidate Generation
        candidates = self.generator.generate(unified_ts, combined_scores, transcript_segments)
        
        # 4. LLM Analysis
        print("=" * 60)
        print("PHASE 4: LLM ANALYSIS")
        print("=" * 60 + "\n")
        
        context = PromptBuilder.prepare_context(candidates)
        prompt = PromptBuilder.create_prompt(context, user_query)
        
        try:
            response = self.llm_client.generate(prompt)
            final_clips = LLMResponseParser.parse_response(response, candidates)
        except Exception as e:
            print(f"[LLM] Failed, falling back to heuristic ranking: {e}")
            final_clips = LLMResponseParser.fallback_ranking(candidates, user_query)
            
        # 5. Export
        exported_files = self.exporter.export(video_path, final_clips)
        
        return {
            "candidates": candidates,
            "final_clips": final_clips,
            "exported_files": exported_files
        }
