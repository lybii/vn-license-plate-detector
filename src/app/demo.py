import sys
from pathlib import Path

import cv2
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from detect import detect_plates
from ocr import read_plate


def process(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = detect_plates(bgr)

    for det in detections:
        det["plate_text"] = read_plate(bgr, det["bbox"])
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['plate_text'] or '?'} ({det['confidence']:.2f})"
        cv2.putText(bgr, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if detections:
        summary = "\n".join(
            f"{d['plate_text'] or '(không đọc được)'} — conf {d['confidence']:.2f}" for d in detections
        )
    else:
        summary = "Không phát hiện biển số."
    return annotated, summary


demo = gr.Interface(
    fn=process,
    inputs=gr.Image(type="numpy", label="Ảnh đầu vào"),
    outputs=[gr.Image(type="numpy", label="Kết quả"), gr.Textbox(label="Biển số nhận diện")],
    title="VN License Plate Detector",
    description="Upload ảnh xe để phát hiện và đọc biển số xe Việt Nam.",
)

if __name__ == "__main__":
    demo.launch()
