# Lộ trình triển khai

## Giai đoạn 1 — Chuẩn bị dữ liệu

- [x] Khảo sát các dataset ứng viên (Kaggle, Roboflow Universe) theo tiêu chí ở [`docs/dataset.md`](dataset.md)
- [x] Chốt 1 dataset cụ thể, ghi rõ nguồn/link/license vào `docs/dataset.md` — đã chọn `bomaich/vnlicenseplate`
- [x] Viết `src/data/download.py` để tải dataset về `data/raw/`
- [x] Chạy `download.py`, kiểm tra trực quan vài ảnh + annotation để xác nhận format đúng như mô tả — đúng format, train 381 / valid 109 / test 8 ảnh
- [x] Viết `data.yaml` trỏ tới split train/valid/test sẵn có trong `data/raw/vnlicenseplate/` (không cần convert vì annotation đã ở YOLO format) — xem `configs/data.yaml`

## Giai đoạn 2 — Huấn luyện baseline

- [x] Viết `configs/train_config.yaml` với tham số baseline
- [x] Viết `configs/data.yaml` (path tương đối, dùng khi train/inference local)
- [x] Viết `notebooks/train_colab.ipynb` (tự tải dataset + tạo data.yaml riêng cho Colab + train + tải `best.pt` về)
- [x] Chạy notebook trên Colab
- [x] Ghi lại kết quả (mAP, precision, recall) vào `docs/pipeline.md` — precision 0.997, recall 1.0, mAP50 0.995, mAP50-95 0.911
- [x] Tải `best.pt` về `models/`

## Giai đoạn 3 — Tinh chỉnh model

**Bỏ qua ở thời điểm này** — kết quả baseline đã rất tốt (xem `docs/pipeline.md`), không cần tinh chỉnh thêm. Sẽ quay lại nếu inference thực tế (Giai đoạn 4) cho thấy vấn đề.

## Giai đoạn 4 — Inference & OCR

- [x] Viết `src/plate_detector/detect.py`
- [x] Viết `src/plate_detector/ocr.py`, xử lý riêng biển 1 dòng và 2 dòng (gom nhóm theo hàng dựa trên toạ độ y, sort theo x trong hàng)
- [x] Test pipeline detect + OCR trên tập ảnh test (8/8 ảnh) — đọc đúng biển chính 7/8 frame, xem chi tiết `docs/pipeline.md`
- [ ] (Tùy chọn) Lọc bbox phụ false-positive (confidence thấp, OCR ra chuỗi rác) — chưa cần thiết cho demo, để sau nếu ảnh hưởng trải nghiệm

## Giai đoạn 5 — Demo & hoàn thiện

- [x] Viết `src/app/demo.py` bằng Gradio
- [x] Test demo qua Gradio API thật (không chỉ đọc code) — upload ảnh, nhận đúng ảnh vẽ bbox + text `51A19222 (0.92)`; xác nhận trực quan bbox phụ là sticker đại lý Toyota, không phải biển số thật
- [x] Test thêm với ảnh đa dạng hơn — quét tập `train` tìm & xác nhận 2 ảnh biển 2 dòng (xe máy) thật, logic ghép dòng hoạt động đúng thứ tự (xem `docs/pipeline.md`)
- [ ] (Tùy chọn) Thêm FastAPI endpoint nếu cần API
- [x] Cập nhật `README.md` với trạng thái hiện tại
- [x] Viết `requirements.txt` (kagglehub, ultralytics, opencv-python, easyocr, gradio)

## Giai đoạn 6 (mở rộng, không bắt buộc)

- [ ] Export model sang ONNX để tăng tốc inference
- [x] Multi-frame tracking + voting (`src/plate_detector/track.py`) — ghép detection qua các frame bằng IoU, vote consensus text theo ký tự; thực nghiệm trên 8 frame test: per-frame 87.5% đúng → sau voting 100%. Xem `docs/pipeline.md`
- [x] Evaluation script (`src/eval/evaluate.py` + `data/eval/ground_truth.json`) — đo exact-match accuracy (7/9 = 77.8%) và character accuracy (94.6%) trên 9 ảnh gán nhãn thủ công; phát hiện & sửa 1 bug matching (chọn theo IoU thay vì confidence) khi ảnh có nhiều biển số
- [x] Viết unit test cơ bản cho `src/plate_detector/` và `src/eval/` (`tests/test_ocr.py`, `tests/test_track.py`, `tests/test_evaluate.py` — 16 test case, chạy bằng `pytest`)

## Giai đoạn 7 — Nâng cấp kiến trúc code + UI demo (2026-08-19)

- [x] Đóng gói `src/inference/{detect,ocr,track}.py` → `src/plate_detector/` (package pip-installable qua `pyproject.toml`, `pip install -e .`), bỏ toàn bộ `sys.path.insert` hack ở `demo.py`, `evaluate.py`, và các file test
- [x] Thêm `plate_detector/pipeline.py` (class `PlateReader`) gom logic "detect + OCR từng bbox" từng bị lặp lại ở `demo.py`, `evaluate.py`, `track.py`
- [x] Xác nhận refactor không đổi hành vi: 16/16 test vẫn pass, `evaluate.py` vẫn ra đúng kết quả cũ (7/9 exact match, 94.6% char accuracy)
- [x] Làm lại `src/app/demo.py` bằng `gr.Blocks`: custom theme, slider chỉnh confidence threshold, ảnh mẫu bấm thử nhanh, kết quả dạng Markdown, accordion giải thích cách hoạt động

## Giai đoạn 8 — Classical image processing (2026-08-19)

- [x] Thêm `src/plate_detector/preprocess.py`: `enhance_contrast()` (CLAHE), `deskew()` (minAreaRect + warpAffine), `binarize()` (Otsu) — có unit test riêng bằng ảnh tổng hợp (`tests/test_preprocess.py`, 6 test case)
- [x] Tích hợp `preprocess_plate()` (CLAHE + deskew) vào `ocr.py`, thay cho việc chỉ chuyển grayscale
- [x] Đo thực nghiệm 3 biến thể qua `evaluate.py`: gray-only vs +CLAHE+deskew vs +binarize — phát hiện **binarize làm giảm accuracy** (77.8%→55.6%) do EasyOCR là OCR deep learning, không phải OCR cổ điển; quyết định không dùng binarize mặc định. Xem chi tiết `docs/pipeline.md`
- [x] Xác nhận không phá vỡ gì: 22/22 test pass sau khi thêm module mới
