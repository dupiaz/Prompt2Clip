import json
import re

class LLMResponseParser:
    """Parses LLM responses and handles fallbacks."""
    
    @staticmethod
    def parse_response(response: str, candidates: list) -> list:
        """Parse LLM JSON response and apply merges."""
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            json_str = response[json_start:json_end]
            
            llm_clips = json.loads(json_str)
            final_clips = []
            
            for llm_clip in llm_clips:
                # Find original candidate clips
                merged_ids = llm_clip.get("merged_clip_ids", [])
                
                if merged_ids:
                    # Use LLM's merged timestamps
                    clip_data = {
                        "start": llm_clip["final_start"],
                        "end": llm_clip["final_end"],
                        "duration": llm_clip["final_end"] - llm_clip["final_start"],
                        "transcript": " ".join([
                            c["transcript"] for c in candidates if c["clip_id"] in merged_ids
                        ]),
                        "llm_interest_score": llm_clip.get("interest_score", 0),
                        "reason": llm_clip.get("reason", ""),
                        "tags": llm_clip.get("tags", []),
                        "merged_from": merged_ids
                    }
                else:
                    clip_data = llm_clip
                
                final_clips.append(clip_data)
            
            return final_clips
            
        except Exception as e:
            print(f"[LLM] Failed to parse response: {e}")
            raise e
            
    @staticmethod
    def fallback_ranking(candidates: list, user_query: str) -> list:
        """Fallback to simple score-based ranking if LLM fails."""
        # Default number of clips
        n_clips = 5
        
        # Extract number from query if present
        match = re.search(r'\d+', user_query)
        if match:
            n_clips = int(match.group())
        else:
            # Fallback to word matching
            lower_query = user_query.lower()
            word_map = {
                'single': 1, 'one': 1, 'two': 2, 'three': 3,
                'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10
            }
            for word, num in word_map.items():
                if re.search(r'\b' + word + r'\b', lower_query):
                    n_clips = num
                    break
        
        return candidates[:n_clips]
