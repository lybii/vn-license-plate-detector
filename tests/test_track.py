from plate_detector.track import iou, track_and_vote, vote_text


def test_iou_identical_boxes_is_one():
    box = [10, 10, 50, 50]
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_iou_partial_overlap():
    # 10x10 boxes overlapping in a 5x10 region -> inter=50, union=150
    score = iou([0, 0, 10, 10], [5, 0, 15, 10])
    assert abs(score - 50 / 150) < 1e-6


def test_vote_text_empty_returns_empty_string():
    assert vote_text([]) == ""
    assert vote_text(["", "", ""]) == ""


def test_vote_text_picks_majority_char_per_position():
    # 2/3 frames agree on "51A19222"; 1 frame misread one character
    texts = ["51A19222", "51A19222", "51A1T222"]
    assert vote_text(texts) == "51A19222"


def test_vote_text_ignores_shorter_partial_reads():
    # a frame that dropped a character shouldn't corrupt the vote
    texts = ["51A19222", "51A19222", "51A9222"]
    assert vote_text(texts) == "51A19222"


def test_track_and_vote_associates_same_object_across_frames_by_iou():
    frame_detections = [
        [{"bbox": [100, 100, 300, 200], "plate_text": "51A19222"}],
        [{"bbox": [102, 101, 301, 199], "plate_text": "51A1T222"}],
        [{"bbox": [101, 100, 300, 201], "plate_text": "51A19222"}],
    ]
    tracks = track_and_vote(frame_detections)
    assert len(tracks) == 1
    assert tracks[0]["voted_text"] == "51A19222"


def test_track_and_vote_keeps_unrelated_boxes_as_separate_tracks():
    frame_detections = [
        [
            {"bbox": [0, 0, 100, 50], "plate_text": "51A19222"},
            {"bbox": [500, 500, 600, 550], "plate_text": ""},
        ],
    ]
    tracks = track_and_vote(frame_detections)
    assert len(tracks) == 2
