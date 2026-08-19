import cv2
import numpy as np

from plate_detector.preprocess import binarize, deskew, enhance_contrast, preprocess_plate


def _rotated_text_blob(angle_deg: float, size: int = 200) -> np.ndarray:
    """White background with a dark rotated rectangle, simulating a skewed plate's text."""
    gray = np.full((size, size), 255, dtype=np.uint8)
    box = cv2.boxPoints(((size / 2, size / 2), (size * 0.6, size * 0.2), angle_deg))
    box = box.astype(np.int32)
    cv2.fillPoly(gray, [box], color=0)
    return gray


def _blob_angle(gray: np.ndarray) -> float:
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    return angle


def test_deskew_straightens_a_rotated_blob():
    skewed = _rotated_text_blob(angle_deg=10)
    assert abs(_blob_angle(skewed)) > 5

    corrected = deskew(skewed)
    assert abs(_blob_angle(corrected)) < 2


def test_deskew_leaves_already_straight_image_unchanged():
    straight = _rotated_text_blob(angle_deg=0)
    result = deskew(straight)
    assert np.array_equal(result, straight)


def test_deskew_ignores_implausibly_large_angles():
    # a nearly-blank crop (a handful of stray foreground pixels) shouldn't be "corrected"
    almost_blank = np.full((100, 100), 255, dtype=np.uint8)
    almost_blank[10:12, 10:12] = 0
    result = deskew(almost_blank)
    assert np.array_equal(result, almost_blank)


def test_enhance_contrast_increases_dynamic_range_of_a_flat_image():
    low_contrast = np.random.randint(100, 121, size=(50, 50), dtype=np.uint8)
    enhanced = enhance_contrast(low_contrast)

    assert enhanced.shape == low_contrast.shape
    assert enhanced.dtype == np.uint8
    assert int(enhanced.max()) - int(enhanced.min()) > int(low_contrast.max()) - int(low_contrast.min())


def test_binarize_produces_only_two_values():
    bimodal = np.full((50, 50), 50, dtype=np.uint8)
    bimodal[:, 25:] = 200
    result = binarize(bimodal)
    assert set(np.unique(result)).issubset({0, 255})


def test_preprocess_plate_returns_same_shape_and_dtype():
    gray = _rotated_text_blob(angle_deg=8)
    result = preprocess_plate(gray)
    assert result.shape == gray.shape
    assert result.dtype == gray.dtype
