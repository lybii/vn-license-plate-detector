import cv2

from plate_detector.detect import detect_plates
from plate_detector.ocr import read_plate


class PlateReader:
    """Combines detection + OCR into one call; used by the demo app, eval script, and track.py."""

    def __init__(self, conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold

    def read(self, image) -> list[dict]:
        detections = detect_plates(image, conf=self.conf_threshold)
        for det in detections:
            det["plate_text"] = read_plate(image, det["bbox"])
        return detections

    def annotate(self, image, detections: list[dict]):
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['plate_text'] or '?'} ({det['confidence']:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return annotated
