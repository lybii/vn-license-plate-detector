import re
from pathlib import Path

import cv2
import easyocr

_reader = None
_INVALID_CHARS = re.compile(r"[^A-Z0-9]")


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _load_image(image):
    if isinstance(image, (str, Path)):
        return cv2.imread(str(image))
    return image


def crop_plate(image, bbox: list[int]):
    img = _load_image(image)
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def order_segments(segments: list[dict]) -> str:
    """Group OCR text segments into rows (by y-proximity) and order each row left-to-right.

    Handles both 1-line plates (segments split across the same row by EasyOCR) and
    2-line plates (segments genuinely on separate rows), using each segment's height
    to judge whether the next segment belongs to the same row.
    """
    if not segments:
        return ""

    segments = sorted(segments, key=lambda s: s["cy"])
    avg_height = sum(s["h"] for s in segments) / len(segments)

    rows = [[segments[0]]]
    for seg in segments[1:]:
        if seg["cy"] - rows[-1][-1]["cy"] < avg_height * 0.6:
            rows[-1].append(seg)
        else:
            rows.append([seg])

    lines = []
    for row in rows:
        row.sort(key=lambda s: s["cx"])
        lines.append("".join(_INVALID_CHARS.sub("", s["text"].upper()) for s in row))
    return "".join(lines)


def read_plate_text(plate_crop) -> str:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    results = _get_reader().readtext(gray)

    segments = []
    for box, text, _ in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        segments.append({"cx": sum(xs) / 4, "cy": sum(ys) / 4, "h": max(ys) - min(ys), "text": text})

    return order_segments(segments)


def read_plate(image, bbox: list[int]) -> str:
    return read_plate_text(crop_plate(image, bbox))


if __name__ == "__main__":
    import sys

    from plate_detector.detect import detect_plates

    image_path = sys.argv[1]
    for det in detect_plates(image_path):
        text = read_plate(image_path, det["bbox"])
        print({**det, "plate_text": text})
