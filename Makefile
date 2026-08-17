.PHONY: setup run ui health clean test-import

# --- Lệnh phổ biến ---

# Cài đặt toàn bộ môi trường Conda và thư viện
setup:
	setup\setup_all.bat

# Chạy giao diện Desktop UI
ui:
	python -m src.ui.app

# Chạy CLI Pipeline (cần truyền tham số video_path)
# Ví dụ: make run VIDEO=videos/sample.mp4
run:
	python Prompt2Clip.py $(VIDEO)

# --- Tiện ích ---

# Kiểm tra tình trạng môi trường Conda
health:
	setup\healthcheck.bat

# Dọn dẹp các file cache sinh ra trong quá trình chạy
clean:
	python -c "import shutil, os; shutil.rmtree('.cache', ignore_errors=True) if os.path.exists('.cache') else None"
	@echo "Đã xóa thư mục cache."

# Chạy test import để đảm bảo code không lỗi cú pháp
test-import:
	python -m py_compile src/ui/app.py
	python Prompt2Clip.py --help
	@echo "Test import thành công!"
