FROM python:3.10-slim

# Ngăn chặn Python tạo ra file .pyc và bật unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cài đặt các thư viện hệ thống cần thiết (như ffmpeg, OpenCV dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép file requirements để tận dụng bộ nhớ cache của Docker
COPY requirements.txt .

# Cài đặt thư viện Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Sao chép mã nguồn dự án
COPY . .

# Lệnh khởi chạy mặc định (có thể bị ghi đè bởi docker-compose)
CMD ["python", "Prompt2Clip.py"]
