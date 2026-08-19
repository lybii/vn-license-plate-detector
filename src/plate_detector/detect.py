from pathlib import Path

from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"
CONF_THRESHOLD = 0.4

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def detect_plates(image, conf: float | None = None) -> list[dict]:
    model = _get_model()
    results = model.predict(image, conf=conf if conf is not None else CONF_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(box.conf[0]),
            }
        )
    return detections


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    for det in detect_plates(image_path):
        print(det)
