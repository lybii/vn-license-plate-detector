# Kế hoạch Dataset

## Đặc điểm biển số xe Việt Nam cần lưu ý

- **Biển 1 dòng**: phổ biến ở ô tô (VD: `30A-123.45`).
- **Biển 2 dòng**: phổ biến ở xe máy (VD: dòng 1 `29-N1`, dòng 2 `123.45`). Khi OCR đọc biển 2 dòng, cần logic ghép 2 dòng text theo đúng thứ tự trên-dưới, không thể chỉ đọc thẳng như biển 1 dòng.
- **Biển nền trắng** (xe cá nhân), **nền vàng** (xe kinh doanh vận tải), **nền xanh** (xe cơ quan nhà nước), **nền đỏ** (xe quân đội/công an) — nếu dataset có đa dạng loại nền thì model sẽ tổng quát hóa tốt hơn, nhưng ở phiên bản đầu chỉ cần detect được vị trí biển (1 class `plate`), chưa cần phân loại nền màu.
- Ký tự trên biển số VN chỉ gồm chữ số (0-9) và một số chữ cái Latin in hoa cố định (không dùng hết bảng chữ cái) — cần lưu ý khi hậu xử lý OCR để lọc bỏ ký tự OCR đọc nhầm (VD: nhầm `0` với `O`, `8` với `B`).

## Dataset đã chọn

**`VNLicensePlate_yolov7`** — [kaggle.com/datasets/bomaich/vnlicenseplate](https://www.kaggle.com/datasets/bomaich/vnlicenseplate)

- Gồm **cả biển 1 dòng và 2 dòng** (khớp yêu cầu xử lý biển xe máy đã nêu ở trên)
- Annotation **đã ở định dạng YOLO** (txt, `class x_center y_center width height`) — đã tải và kiểm tra thực tế, đúng như mô tả, không cần viết script convert từ VOC/COCO
- **Đã chia sẵn** train/valid/test — dùng nguyên split này, không tự chia lại
- Số lượng ảnh thực tế sau khi tải (2026-08-19): **train 381 / valid 109 / test 8** (tổng 498) — ít hơn con số ~1000 ghi nhận ban đầu lúc khảo sát, và tập test chỉ có 8 ảnh nên khá nhỏ để đánh giá tin cậy; cân nhắc dùng thêm `valid` làm test bổ sung, hoặc bổ sung dataset dự phòng bên dưới nếu baseline cho thấy cần thêm dữ liệu.
- Class: 1 class duy nhất `plate`, khớp thiết kế trong `docs/architecture.md`

Vì annotation đã đúng format cần dùng, `src/data/download.py` chỉ cần tải dataset về `data/raw/`, không cần thêm bước convert riêng như dự tính ban đầu (mục "Chuyển đổi annotation" bên dưới vẫn giữ lại để tham khảo cho trường hợp dùng thêm dataset khác sau này).

### Dataset dự phòng (chưa dùng, cân nhắc nếu cần thêm dữ liệu)

- **Roboflow — "Vietnamese Car License Plate" by Cuong Ta** ([universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate](https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate)) — ~8.255 ảnh, lớn hơn nhiều nhưng thiên về biển ô tô, có thể thiếu biển 2 dòng xe máy.
- **Roboflow — "Vietnam License plate" by Traffic Camera** ([universe.roboflow.com/traffic-camera/vietnam-license-plate-hayn8](https://universe.roboflow.com/traffic-camera/vietnam-license-plate-hayn8)) — ~885 ảnh, license CC BY 4.0, góc chụp camera giao thông (bổ sung tốt cho case góc xa/camera an ninh).

### Tiêu chí đã dùng để chọn

### Tiêu chí chọn dataset

- Có annotation dạng bounding box cho vị trí biển số (không phải chỉ classification toàn ảnh).
- Số lượng ảnh đủ để fine-tune (lý tưởng ≥ 1000 ảnh; YOLOv8 vẫn có thể fine-tune tốt với vài trăm ảnh nhờ transfer learning từ pretrained COCO).
- Đa dạng điều kiện chụp: góc nghiêng, khoảng cách, ánh sáng, xe máy lẫn ô tô — để model không bị overfit vào 1 kiểu ảnh.
- Giấy phép sử dụng cho phép mục đích học tập/portfolio (không dùng dataset có điều khoản cấm chia sẻ lại).

## Chuyển đổi annotation

Hầu hết dataset public dùng định dạng Pascal VOC (XML) hoặc COCO (JSON). YOLOv8 yêu cầu định dạng YOLO txt, mỗi ảnh 1 file `.txt` cùng tên, mỗi dòng là 1 object:

```
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó toạ độ được chuẩn hóa về khoảng [0, 1] theo kích thước ảnh. Với dự án này chỉ có **1 class**: `plate` (class_id = 0).

`src/data/download.py` sẽ đảm nhiệm tải dataset thô về `data/raw/`; một script/notebook riêng (sẽ viết ở bước sau) đảm nhiệm convert annotation sang format YOLO và ghi ra `data/processed/`.

## Chiến lược chia tập

- Tỉ lệ mặc định: **80% train / 10% validation / 10% test**.
- Nếu dataset gốc đã có sẵn split (train/val/test thư mục riêng) thì ưu tiên giữ nguyên để dễ so sánh với các kết quả benchmark công khai (nếu có).
- Chia theo **ảnh**, không chia theo annotation riêng lẻ, để tránh rò rỉ dữ liệu (data leakage) giữa các tập.

## Tăng cường dữ liệu (data augmentation)

Áp dụng augmentation ngay trong lúc train qua Ultralytics (có sẵn, cấu hình trong `configs/train_config.yaml`) thay vì tạo file ảnh augment riêng:

- Mosaic, random flip ngang, thay đổi độ sáng/tương phản (mô phỏng điều kiện ánh sáng khác nhau)
- Random scale/crop nhẹ (mô phỏng khoảng cách chụp khác nhau)
- **Không** áp dụng flip dọc hoặc xoay góc lớn — biển số luôn có hướng cố định, augmentation kiểu này sẽ tạo dữ liệu phi thực tế

## Nơi lưu trữ

- `data/raw/` — dataset gốc y như tải về, không chỉnh sửa (gitignore, không commit vì dung lượng lớn)
- `data/processed/` — ảnh + label định dạng YOLO đã convert, cùng file `data.yaml` khai báo class và đường dẫn train/val/test (gitignore, không commit)
- Repo chỉ commit **script tạo ra dữ liệu**, không commit **bản thân dữ liệu**

## Việc cần làm tiếp theo

- Chạy `src/data/download.py` để tải dataset về `data/raw/` (cần cấu hình Kaggle API credentials, xem README).
- Kiểm tra trực quan vài ảnh + annotation sau khi tải để xác nhận format đúng như mô tả trước khi bắt đầu train.
