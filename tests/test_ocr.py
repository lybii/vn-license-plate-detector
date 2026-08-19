from plate_detector.ocr import order_segments


def _seg(cx, cy, h, text):
    return {"cx": cx, "cy": cy, "h": h, "text": text}


def test_empty_segments_returns_empty_string():
    assert order_segments([]) == ""


def test_single_line_plate_orders_by_x_even_if_y_is_noisy():
    # EasyOCR sometimes splits a 1-line plate into segments with slightly
    # different y, and may return them out of left-to-right order.
    segments = [
        _seg(cx=120, cy=42, h=40, text="9222"),
        _seg(cx=20, cy=40, h=40, text="51A1"),
    ]
    assert order_segments(segments) == "51A19222"


def test_two_line_plate_orders_top_row_before_bottom_row():
    segments = [
        _seg(cx=30, cy=120, h=30, text="123.45"),
        _seg(cx=20, cy=20, h=30, text="29-N1"),
    ]
    assert order_segments(segments) == "29N112345"


def test_strips_invalid_characters_and_uppercases():
    segments = [_seg(cx=10, cy=10, h=20, text="51a-192.22")]
    assert order_segments(segments) == "51A19222"
