# Workers

Các process chạy ngầm trong môi trường Conda độc lập để phân tích AI.

- Để tận dụng GPU và tránh xung đột thư viện, hệ thống dùng `subprocess` gọi các file trong này thông qua các conda environment (`clipz_audio`, `clipz_vision`, `clipz_whisper`).
- Worker không chứa logic tính toán (tất cả nằm ở `src/`), worker chỉ import từ `src/` và chạy trong môi trường thích hợp.
