# VN License Plate Detector

Phát hiện và nhận diện biển số xe Việt Nam: fine-tune YOLOv8 để xác định vị trí biển số, sau đó dùng OCR để đọc nội dung.

Tài liệu thiết kế chi tiết: [`docs/architecture.md`](docs/architecture.md), [`docs/dataset.md`](docs/dataset.md), [`docs/pipeline.md`](docs/pipeline.md), [`docs/roadmap.md`](docs/roadmap.md).

## Cấu trúc thư mục

```
configs/          File cấu hình train/inference
data/
  raw/             Dataset gốc tải về (không commit)
  processed/       Dataset đã xử lý/chia split (không commit)
notebooks/         Notebook train trên Colab (GPU)
src/
  data/            Script tải & xử lý dataset
  train/           Script train YOLOv8
  inference/       Pipeline detect + OCR + multi-frame tracking/voting
  app/             Demo app (Gradio/FastAPI)
  eval/            Script đánh giá độ chính xác (so với ground truth gán tay)
models/            Trọng số model đã train (không commit)
tests/             Unit test
docs/              Tài liệu kiến trúc & kế hoạch
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Dùng virtual environment riêng (`.venv/`, đã có trong `.gitignore`) để tránh xung đột phiên bản package (numpy, v.v.) với các project Python khác trên máy.

## Tải dataset

```bash
python src/data/download.py
```

Script tải dataset [`bomaich/vnlicenseplate`](https://www.kaggle.com/datasets/bomaich/vnlicenseplate) qua `kagglehub`, cần có Kaggle API credentials (`~/.kaggle/kaggle.json`, lấy tại Kaggle → Account → Create New API Token). Dataset được lưu vào `data/raw/vnlicenseplate/`.

## Trạng thái

- [x] Khung thư mục + tài liệu kiến trúc
- [x] Chọn dataset, tải dataset về `data/raw/vnlicenseplate/` (381 train / 109 valid / 8 test ảnh)
- [x] Viết `configs/data.yaml`, `configs/train_config.yaml`, `notebooks/train_colab.ipynb`
- [x] Train baseline trên Colab — precision 0.997, recall 1.0, mAP50 0.995 (xem `docs/pipeline.md`)
- [x] Inference pipeline (`src/inference/detect.py` + `ocr.py`), test trên 8 ảnh — đọc đúng biển 7/8 frame
- [x] Demo app (`src/app/demo.py`, Gradio) — đã test qua API thật, hoạt động đúng
- [x] Xác nhận logic đọc biển 2 dòng (xe máy) đúng trên ảnh thật
- [x] Multi-frame tracking + voting (`src/inference/track.py`) — bù lỗi OCR từng frame bằng cách vote đa số qua nhiều frame: per-frame 87.5% đúng → sau voting 100%
- [x] Evaluation script (`src/eval/evaluate.py`) — exact-match 77.8%, character accuracy 94.6% trên 9 ảnh gán nhãn tay
- [x] Unit test cho toàn bộ logic thuần (`tests/`, 16 test case, chạy `pytest`)

## Chạy demo

```bash
python src/app/demo.py
```

Mở `http://127.0.0.1:7860` trên trình duyệt, upload ảnh xe để xem kết quả phát hiện + đọc biển số.

## Đánh giá độ chính xác

```bash
python src/eval/evaluate.py
```

## Chạy tracking + voting trên nhiều frame

```bash
python src/inference/track.py <thư mục ảnh chứa các frame liên tiếp>
```

Xem chi tiết lộ trình tại [`docs/roadmap.md`](docs/roadmap.md).
