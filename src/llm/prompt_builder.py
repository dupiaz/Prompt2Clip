import json

class PromptBuilder:
    """Builds prompts for LLM selection."""
    
    @staticmethod
    def prepare_context(candidates: list, audio_scores: dict = None, video_scores: dict = None) -> list:
        """Prepare structured context for LLM."""
        context = []
        for clip in candidates:
            context.append({
                "clip_id": clip["clip_id"],
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "duration": round(clip["duration"], 2),
                "transcript": clip["transcript"][:200],  # Truncate for context
                "excitement_score": round(clip["avg_score"], 3)
            })
        return context
        
    @staticmethod
    def create_prompt(context: list, user_query: str) -> str:
        """Create prompt for LLM clip selection."""
        prompt = f"""You are a viral video clip curator. Analyze these video segments and select the best clips based on the user's request.

        USER REQUEST: {user_query}

        CANDIDATE CLIPS:
        {json.dumps(context, indent=2)}

        INSTRUCTIONS:
        1. Analyze each clip's transcript for interestingness, emotion, humor, or viral potential
        2. Identify clips that should be MERGED (if they're adjacent and part of the same story)
        3. Score each clip or merged clip for interest (0-10)
        4. Return the requested number of clips, ranked by interest

        MERGING RULES:
        - Merge clips if they're within 5 seconds of each other
        - Merge if they continue the same topic/story
        - Merged clips should have combined start/end times

        OUTPUT FORMAT (JSON only, no other text):
        [
            {{
                "merged_clip_ids": [1, 2],
                "final_start": 10.5,
                "final_end": 45.0,
                "reason": "Why this clip is interesting",
                "interest_score": 9.5,
                "tags": ["emotional", "funny"]
            }}
        ]

        CRITICAL: Make sure the JSON is perfectly valid. Do not use unescaped double quotes inside the reason string. Do not leave trailing commas.
        Return ONLY valid JSON, nothing else."""
        
        return prompt
