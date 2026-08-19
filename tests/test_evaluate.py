from evaluate import char_accuracy


def test_char_accuracy_exact_match_is_one():
    assert char_accuracy("51A19222", "51A19222") == 1.0


def test_char_accuracy_partial_mismatch():
    assert char_accuracy("51A1T222", "51A19222") == 7 / 8


def test_char_accuracy_shorter_prediction_counts_missing_chars_as_wrong():
    assert char_accuracy("51A9222", "51A19222") == 5 / 8


def test_char_accuracy_empty_truth_is_zero():
    assert char_accuracy("anything", "") == 0.0
