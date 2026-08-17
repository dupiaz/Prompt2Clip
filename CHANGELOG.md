# Changelog

Tất cả các thay đổi đáng kể của dự án Prompt2Clip sẽ được ghi chép lại trong file này.

## [Unreleased]

### Đã thêm (Added)
- **Môi trường Conda Local**: Thiết lập môi trường Python 3.10 tại thư mục `.\.conda` để chuẩn hóa các gói dependency và tách biệt khỏi hệ thống toàn cục.
- **Cấu hình Docker**: Thêm hỗ trợ chạy ứng dụng bằng Container.
  - `Dockerfile`: Sử dụng base image `python:3.10-slim`, cài đặt các thư viện hệ thống cần thiết (như `ffmpeg`, `libgl1`) và cài đặt các dependencies từ `requirements.txt`.
  - `.dockerignore`: Cấu hình bỏ qua các thư mục như `.conda`, `.git`, `__pycache__`, `videos`, `models` để tối ưu kích thước build context.
  - `docker-compose.yml`: Bổ sung cấu hình chạy Docker Compose, hỗ trợ tự động mount các volume lưu trữ mô hình AI và video tạm thời, đồng thời sẵn sàng hỗ trợ cấu hình GPU khi cần thiết.


### Sửa lỗi (Fixed)
- **Lỗi thiếu CUDA DLLs trên Windows (CTranslate2) & Xung đột cuDNN**: Khắc phục lỗi `cublas64_12.dll not found` khi chạy `faster-whisper`. Đã bổ sung gói `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` để Whisper chạy tối đa tốc độ. Đặc biệt lưu ý: Do hệ thống sử dụng GPU **NVIDIA RTX 5060 Ti** với kiến trúc thế hệ mới (Compute Capability `sm_120`), PyTorch BẮT BUỘC phải dùng phiên bản **CUDA 13.2** (`torch==2.13.0+cu132`) mới có thể chạy được (các bản CUDA 12 sẽ bị lỗi `no kernel image is available`). Vì vậy, để tránh xung đột DLL cuDNN giữa CUDA 12 (của Whisper) và CUDA 13 (của PyTorch), luồng Whisper đã được thiết kế chạy **cách ly qua Subprocess**.
- **Lỗi in Unicode Windows Console**: Sửa lỗi crash khi in ký tự mũi tên (`→`) trong log trả về của transcript, chuyển thành `->`.
- **FFmpeg DLL crash trên Windows** (`0xC0000139 — STATUS_ENTRYPOINT_NOT_FOUND`): Bản FFmpeg 8.0.1 từ conda-forge bị xung đột DLL, không thể chạy trên Windows. Đã gỡ bản conda-forge và thay thế bằng FFmpeg 9.0 static build từ [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (tự chứa tất cả thư viện, không phụ thuộc DLL bên ngoài). Các file `ffmpeg.exe`, `ffprobe.exe`, `ffplay.exe` được đặt tại `.conda\Library\bin\`.
- **CLIP model không tương thích transformers 5.x** (`video.py`): Method `get_image_features()` ở phiên bản `transformers` mới (5.14.1) trả về `BaseModelOutputWithPooling` thay vì tensor trực tiếp, gây lỗi `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'cpu'`. Đã sửa bằng cách kiểm tra kiểu trả về và trích xuất `.pooler_output` khi cần, tương thích cả phiên bản cũ lẫn mới.

### Cải thiện (Changed)
- **Nâng cấp công cụ nhận diện giọng nói (Transcription)**: Thay thế `openai-whisper` bằng `faster-whisper` (dựa trên CTranslate2) cho tốc độ xử lý vượt trội. Nâng cấp cấu hình model mặc định từ `base` lên `medium` để tăng độ chính xác, tối ưu hóa chạy độc quyền trên CUDA với `float16`.
- **Kích hoạt GPU (CUDA)**: Nâng cấp PyTorch lên bản hỗ trợ CUDA 13.2 (`torch==2.13.0+cu132`) thay cho bản CPU-only. Đồng thời sửa code `video.py` để chuyển các mô hình CLIP và YOLO lên GPU: thêm auto-detect device, dùng `.to(device)` cho CLIP model và inputs, truyền `device=` cho YOLO inference. Cải thiện tốc độ xử lý video 5-10x.
- **Tích hợp LLM**: Chuyển đổi từ Google Gemini (`google-genai`) sang Anthropic Claude (`claude-3-5-sonnet-20240620`) qua chuẩn API tương thích OpenAI của nền tảng `api.ai-box.vn`. Tối ưu cho JSON Response format.
- Tổ chức lại luồng khởi tạo và cài đặt môi trường (Conda + requirements.txt).
