# LLM Integration

Tích hợp LLM để phân tích clip ứng viên và chọn đoạn viral.

- `aibox_client.py`: Class tương tác API.
- `prompt_builder.py`: Khung prompt gửi lên mô hình.
- `response_parser.py`: Xử lý đầu ra, map với danh sách clip.
- Muốn đổi provider AI (như OpenAI, Anthropic)? Tạo file mới kế thừa `BaseLLMClient`.
