# Core Pipeline

Trái tim của hệ thống: kết hợp mọi thứ lại thành luồng chạy hoàn chỉnh.

- `signal_fusion.py`: Đồng bộ dòng thời gian giữa Audio và Video.
- `clip_generator.py`: Thuật toán tìm đỉnh (peaks) và đề xuất clip theo câu chữ.
- `pipeline.py`: Lớp `ClipExtractor` chứa toàn bộ flow từ đầu đến cuối qua Dependency Injection.
