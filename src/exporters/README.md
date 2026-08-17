# Exporters

Chịu trách nhiệm cắt video và xuất file.

- `ffmpeg_exporter.py`: Gọi FFmpeg để xử lý media.
- Kế thừa `BaseExporter` nếu muốn xuất định dạng khác (GIF, Audio only...).
