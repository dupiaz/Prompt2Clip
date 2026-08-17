import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
AIBOX_API_KEY = os.getenv("AIBOX_API_KEY")

class LLM:
    def __init__(self):
        if not AIBOX_API_KEY:
            raise ValueError("AIBOX_API_KEY not found in .env")
        # Sử dụng chuẩn tương thích OpenAI của API AI-BOX
        self.client = OpenAI(
            api_key=AIBOX_API_KEY, 
            base_url="https://api.ai-box.vn/v1"
        )

    def generate_text(
        self,
        prompt,
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0.3
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            # Bắt buộc trả về JSON vì Prompt2Clip mong đợi định dạng này để parse
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content