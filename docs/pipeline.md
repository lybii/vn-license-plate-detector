# Pipeline Training & Inference

## 1. Training trên Google Colab

### Vì sao dùng Colab thay vì máy local

Máy local có GPU NVIDIA nhưng bản PyTorch đang cài là bản CPU-only (chưa cấu hình CUDA), nên trước mắt dùng Colab (GPU miễn phí, sẵn CUDA/PyTorch) để không mất thời gian setup môi trường. Có thể chuyển sang train local sau nếu cấu hình lại được CUDA.

### Các bước

1. Mở `notebooks/train_colab.ipynb` trên Google Colab, chọn Runtime → GPU.
2. Cài đặt thư viện: `pip install ultralytics`.
3. Tải `data/processed/` lên Colab (upload trực tiếp, hoặc tải từ Google Drive/Roboflow API — sẽ quyết định khi có dataset cụ thể).
4. Load model pretrained: bắt đầu bằng `yolov8n.pt` (bản nhỏ nhất, train nhanh, phù hợp free-tier GPU/thời gian giới hạn của Colab).
5. Fine-tune bằng lệnh tương đương:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)
   ```
   Các tham số cụ thể (epochs, imgsz, batch, learning rate, augmentation) được quản lý tập trung trong `configs/train_config.yaml` thay vì hard-code trong notebook, để dễ tái lập và so sánh giữa các lần thử.
6. Sau khi train, chạy `model.val()` để lấy chỉ số đánh giá trên tập validation.
7. Tải file trọng số tốt nhất (`runs/detect/train/weights/best.pt`) về máy local, lưu vào `models/`.

### Chỉ số đánh giá

- **mAP@0.5** — chỉ số chính để đánh giá độ chính xác detection tổng thể.
- **Precision / Recall** — theo dõi riêng để biết model đang bỏ sót biển số (recall thấp) hay báo sai vị trí không phải biển số (precision thấp), từ đó quyết định hướng tinh chỉnh (thêm dữ liệu, đổi ngưỡng confidence, đổi model size).

### Chiến lược tinh chỉnh nếu kết quả baseline chưa đạt

- Nếu recall thấp (bỏ sót biển số): tăng số epoch, thêm augmentation, hoặc chuyển sang `yolov8s`/`yolov8m` (model lớn hơn, chính xác hơn nhưng chậm hơn).
- Nếu overfit (train tốt, val kém): tăng augmentation, giảm epoch, hoặc bổ sung thêm dữ liệu.
- Nếu ảnh biển số quá nhỏ trong khung hình gốc (xe ở xa): cân nhắc tăng `imgsz` khi train/inference để giữ chi tiết.

## 2. Inference pipeline (chạy local, không cần GPU)

### `src/inference/detect.py`

- Load model: `YOLO("models/best.pt")`.
- Input: đường dẫn ảnh hoặc ảnh dạng numpy array.
- Output: danh sách `{bbox: [x1,y1,x2,y2], confidence: float}` cho mỗi biển số phát hiện được.
- Áp dụng ngưỡng confidence tối thiểu (VD: 0.4) để lọc bớt kết quả nhiễu, giá trị cụ thể sẽ tinh chỉnh dựa trên kết quả train thực tế.

### `src/inference/ocr.py`

- Nhận ảnh gốc + bbox từ bước trên, crop vùng biển số.
- Tiền xử lý ảnh crop trước khi đưa vào OCR: resize về kích thước chuẩn, chuyển grayscale, có thể áp dụng threshold/contrast enhancement nếu ảnh mờ.
- Chạy EasyOCR (`reader.readtext(cropped_image)`).
- Hậu xử lý chuỗi kết quả:
  - Loại bỏ ký tự không thuộc bảng chữ cái/số hợp lệ trên biển số VN.
  - Với biển 2 dòng (xe máy): ghép kết quả OCR của 2 vùng trên/dưới theo đúng thứ tự (dựa vào tọa độ y của từng dòng text mà EasyOCR trả về).
  - Chuẩn hóa định dạng hiển thị (VD: thêm dấu `-` và `.` đúng vị trí theo quy chuẩn biển số VN) — mức độ chuẩn hóa này sẽ quyết định khi có dữ liệu thực tế để test.

### Kết quả trả về của pipeline tổng hợp

```json
[
  {
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.93,
    "plate_text": "30A-123.45"
  }
]
```

## 3. Demo app — `src/app/demo.py`

- Dùng **Gradio** làm giao diện chính (ưu tiên vì dựng nhanh, phù hợp quay video/chụp ảnh demo cho portfolio):
  - Input: upload ảnh hoặc chọn ảnh mẫu có sẵn.
  - Output: ảnh gốc có vẽ bounding box + text biển số nhận diện được, kèm confidence score.
- Có thể bổ sung **FastAPI** sau nếu muốn có REST API riêng (`POST /detect`, nhận ảnh trả về JSON) để tích hợp với ứng dụng khác hoặc dùng cho việc test tự động.

## 4. File cấu hình — `configs/train_config.yaml`

Tập trung các tham số có thể thay đổi giữa các lần train để không phải sửa trực tiếp trong code/notebook:

```yaml
model: yolov8n.pt
epochs: 50
imgsz: 640
batch: 16
lr0: 0.01
patience: 20
augment: true
```

(Giá trị cụ thể sẽ điều chỉnh sau khi có dataset thật và chạy thử baseline đầu tiên.)

## Kết quả baseline (2026-08-19)

Train xong trên Colab với tham số baseline ở trên (`yolov8n`, 50 epochs, imgsz 640), đánh giá trên tập `valid` (109 ảnh, 125 đối tượng):

| Metric | Giá trị |
|---|---|
| Precision | 0.997 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.911 |

Kết quả rất tốt ngay từ baseline — nhờ bài toán chỉ có 1 class (`plate`) và transfer learning tốt từ pretrained COCO. Vì các chỉ số đã cao, **quyết định bỏ qua Giai đoạn 3 (tinh chỉnh sâu)** ở thời điểm này và chuyển sang inference/demo; có thể quay lại tinh chỉnh sau nếu inference thực tế cho thấy model detect sai nhiều (VD: ảnh góc lạ, thiếu sáng — điều tập test 8 ảnh chưa phản ánh hết).

Trọng số: `models/best.pt` (~6MB, yolov8n).

## Kết quả test inference + OCR (2026-08-19)

Chạy `src/inference/detect.py` + `src/inference/ocr.py` trên toàn bộ 8 ảnh tập `test` (đều là frame từ cùng 1 clip, cùng 1 xe/biển số):

- **Detection**: đúng vị trí biển số chính ở cả 8/8 ảnh, confidence 0.90–0.93.
- **OCR trên biển chính**: đọc đúng `51A19222` ở 7/8 frame; 1 frame thiếu 1 ký tự do ảnh mờ/góc xấu — chấp nhận được, không phải lỗi logic.
- **Bug đã sửa**: ban đầu `read_plate_text()` chỉ sort theo toạ độ y, khiến biển 1 dòng bị EasyOCR tách nhiều đoạn có y gần bằng nhau bị **đảo thứ tự ký tự** (VD: `51A92221` thay vì `51A19222`, cùng bộ ký tự nhưng sai vị trí). Đã sửa bằng cách gom nhóm các đoạn text theo hàng (dựa vào chênh lệch y so với chiều cao trung bình đoạn text), sort theo x trong từng hàng, rồi mới ghép các hàng theo thứ tự trên-dưới — logic này đúng cho cả biển 1 dòng lẫn 2 dòng.
- **Giới hạn còn tồn tại**: ở 5/8 frame, model phát hiện thêm 1 bbox phụ (confidence 0.41–0.84) không phải biển số thật — đã xác nhận trực quan qua demo Gradio đây là **sticker đại lý Toyota** trên xe, không phải biển số. Chưa xử lý lọc false-positive này; nếu ảnh hưởng demo, có thể tăng `CONF_THRESHOLD` trong `detect.py` (hiện 0.4) hoặc lọc thêm theo tỉ lệ khung hình (aspect ratio) đặc trưng của biển số.

## Xác nhận riêng cho biển 2 dòng (xe máy)

Tập `test` (8 ảnh) chỉ có 1 xe/biển 1 dòng, nên logic ghép dòng cho **biển 2 dòng chưa được validate qua tập test**. Đã tự kiểm tra bằng cách quét 40 ảnh ngẫu nhiên trong tập `train` để tìm bbox có tỉ lệ khung hình gần vuông (đặc trưng biển 2 dòng), tìm được 2 ảnh biển 2 dòng thật:

| Ảnh | Biển thật (đọc bằng mắt) | OCR trả về | Nhận xét |
|---|---|---|---|
| `clip18_new_14.jpg` | `48-H1` / `163.26` | `48HT16326` | Thứ tự dòng trên→dưới **đúng**; sai 1 ký tự (`1`→`T`, lỗi nhận diện font, không phải lỗi logic ghép dòng) |
| `9.jpg` | `2-D1` / `60.97` (ảnh bị cắt lề trái) | `2D15097` | Thứ tự dòng đúng; sai 1 ký tự (`6`→`5`) |

**Kết luận**: logic gom nhóm theo hàng + sort theo x (viết ở `order_segments()` trong `src/inference/ocr.py`) hoạt động đúng cho cả biển 1 dòng và 2 dòng — lỗi còn lại chỉ là nhận diện sai ký tự đơn lẻ của EasyOCR (giới hạn của model OCR, không phải bug logic). Có unit test cho hàm này ở `tests/test_ocr.py`.

## Multi-frame tracking & voting (2026-08-19)

Vì lỗi OCR còn lại chủ yếu là sai/mất 1 ký tự đơn lẻ ở từng frame riêng lẻ, và input thực tế thường là video (nhiều frame liên tiếp của cùng 1 xe) chứ không chỉ 1 ảnh, mình viết thêm `src/inference/track.py` để tận dụng thông tin từ nhiều frame:

1. Ghép detection cùng 1 biển số qua các frame bằng IoU giữa bbox frame trước/sau (`iou()`).
2. Với mỗi track, vote ra text đồng thuận theo từng vị trí ký tự (`vote_text()`) — dùng độ dài phổ biến nhất trong các lần đọc để loại các lần đọc thiếu ký tự, rồi vote ký tự đa số ở từng vị trí.

**Kết quả trên 8 frame test** (chạy `python src/inference/track.py <thư mục ảnh>`):

| | Per-frame (riêng lẻ) | Sau voting |
|---|---|---|
| Track biển chính | 7/8 frame đúng (87.5%) | **8/8 — voted text = `51A19222` (100%)** |
| Track phụ (sticker Toyota) | chuỗi rác/rỗng ở các frame | vẫn ra chuỗi rác (`TOTOTATAVOKA`) — càng xác nhận đây không phải biển thật |

Voting giúp bù được frame bị OCR đọc thiếu ký tự (frame 17: `51A9222` — thiếu 1 chữ số) bằng cách kết hợp thông tin từ các frame còn lại, mà không cần model OCR tốt hơn. Có unit test cho `iou()`, `vote_text()`, `track_and_vote()` ở `tests/test_track.py`.

## Evaluation script (2026-08-19)

`src/eval/evaluate.py` đo độ chính xác pipeline (detect + OCR) trên tập ảnh đã gán nhãn thủ công (`data/eval/ground_truth.json`, hiện có 9 ảnh: 8 frame biển 1 dòng + 1 ảnh biển 2 dòng, gán nhãn dựa trên xác minh trực quan ở các bước trên).

**Lưu ý quan trọng đã phát hiện khi viết script**: ảnh `clip18_new_14.jpg` có 3 biển số/bbox khác nhau trong cùng 1 ảnh — nếu chọn detection theo confidence cao nhất để so khớp với ground truth sẽ **chọn nhầm biển khác**, cho kết quả sai lệch hoàn toàn dù pipeline hoạt động đúng. Đã sửa: match detection với ground truth bằng IoU giữa bbox, không dùng confidence.

**Kết quả** (chạy `python src/eval/evaluate.py`):

| Metric | Giá trị |
|---|---|
| Exact-match accuracy | 7/9 = 77.8% |
| Mean character accuracy | 94.6% |

2 trường hợp sai đều là lỗi OCR đọc thiếu/nhầm 1 ký tự (không phải lỗi detect hay lỗi logic ghép dòng) — khớp với phân tích ở các mục trên. Ground truth hiện còn nhỏ (9 ảnh, gán nhãn thủ công); có thể mở rộng dần khi có thêm ảnh xác minh được.
