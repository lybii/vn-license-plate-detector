from pathlib import Path

import cv2
import gradio as gr

from plate_detector.pipeline import PlateReader

reader = PlateReader()

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "data" / "raw" / "vnlicenseplate" / "test" / "images"
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.jpg"))[:4] if EXAMPLES_DIR.exists() else []


def process(image, conf_threshold: float):
    if image is None:
        return None, "_Chưa có ảnh đầu vào._"

    reader.conf_threshold = conf_threshold
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = reader.read(bgr)
    annotated = cv2.cvtColor(reader.annotate(bgr, detections), cv2.COLOR_BGR2RGB)

    if detections:
        result_md = "\n".join(
            f"- **`{d['plate_text'] or '(không đọc được)'}`** — confidence {d['confidence']:.0%}"
            for d in detections
        )
    else:
        result_md = "_Không phát hiện biển số nào._"
    return annotated, result_md


with gr.Blocks(title="VN License Plate Detector") as demo:
    gr.Markdown(
        "# 🚗 VN License Plate Detector\n"
        "Phát hiện và đọc biển số xe Việt Nam bằng YOLOv8 (fine-tuned) + EasyOCR — hỗ trợ cả biển 1 dòng và 2 dòng."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Ảnh đầu vào")
            confidence_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Ngưỡng confidence")
            detect_button = gr.Button("Phát hiện biển số", variant="primary")
            if EXAMPLE_PATHS:
                gr.Examples(examples=[[str(p)] for p in EXAMPLE_PATHS], inputs=input_image, label="Ảnh mẫu")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Kết quả")
            output_text = gr.Markdown(label="Biển số nhận diện")

    with gr.Accordion("Cách hoạt động", open=False):
        gr.Markdown(
            "1. **YOLOv8** (fine-tuned trên dataset biển số VN) phát hiện vị trí biển số trong ảnh.\n"
            "2. Vùng biển được cắt ra, đưa qua **EasyOCR** để đọc ký tự.\n"
            "3. Kết quả OCR được ghép theo hàng (hỗ trợ đúng cả biển 1 dòng và 2 dòng xe máy).\n\n"
            "Chi tiết kiến trúc & kết quả thực nghiệm: "
            "[github.com/lybii/vn-license-plate-detector](https://github.com/lybii/vn-license-plate-detector)"
        )

    detect_button.click(process, inputs=[input_image, confidence_slider], outputs=[output_image, output_text])
    input_image.change(process, inputs=[input_image, confidence_slider], outputs=[output_image, output_text])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"))
