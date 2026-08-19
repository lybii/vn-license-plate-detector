import cv2
import numpy as np


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE (adaptive histogram equalization) — helps plates that are underlit or glared."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the plate upright using the minimum-area rectangle of its foreground pixels.

    Skips rotation if the estimated angle is negligible or implausibly large (>15deg) --
    a plate crop from YOLO is already close to axis-aligned, so a large angle usually
    means the foreground mask picked up noise rather than genuine skew.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray

    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu binarization -- produces clean black-on-white text."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def preprocess_plate(gray: np.ndarray) -> np.ndarray:
    return deskew(enhance_contrast(gray))
