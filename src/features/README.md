# Features - Trích xuất đặc trưng

Thư mục này chứa các class trích xuất đặc trưng độc lập từ audio/video.

- Muốn thêm đặc trưng mới? Kế thừa `BaseFeatureExtractor` ở `base.py`.
- Âm thanh: loudness, spectral, rhythm, silence, boundary, events.
- Hình ảnh: motion, surprise, composition, thumbnailability.
