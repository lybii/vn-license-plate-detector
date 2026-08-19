from collections import Counter
from pathlib import Path

IOU_MATCH_THRESHOLD = 0.3


def iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def vote_text(texts: list[str]) -> str:
    """Majority-vote a consensus string from OCR readings of the same plate across frames.

    Individual frames often disagree on 1-2 characters (OCR misreads) or drop a
    character entirely (partial detection). Voting per character position, among
    readings that share the most common length, cancels out those per-frame errors.
    """
    candidates = [t for t in texts if t]
    if not candidates:
        return ""

    mode_len = Counter(len(t) for t in candidates).most_common(1)[0][0]
    same_len = [t for t in candidates if len(t) == mode_len]
    if not same_len:
        return max(candidates, key=len)

    return "".join(Counter(t[i] for t in same_len).most_common(1)[0][0] for i in range(mode_len))


def track_and_vote(frame_detections: list[list[dict]]) -> list[dict]:
    """Greedily associate detections across frames by IoU, then vote a consensus text per track.

    frame_detections: one list of {"bbox": [...], "plate_text": str} per frame, in order.
    """
    tracks: list[dict] = []

    for detections in frame_detections:
        for det in detections:
            best_track, best_score = None, IOU_MATCH_THRESHOLD
            for track in tracks:
                score = iou(track["bbox"], det["bbox"])
                if score > best_score:
                    best_track, best_score = track, score

            if best_track is not None:
                best_track["bbox"] = det["bbox"]
                best_track["texts"].append(det.get("plate_text", ""))
            else:
                tracks.append({"bbox": det["bbox"], "texts": [det.get("plate_text", "")]})

    for track in tracks:
        track["voted_text"] = vote_text(track["texts"])
    return tracks


if __name__ == "__main__":
    import sys

    from plate_detector.pipeline import PlateReader

    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if frames_dir is None:
        raise SystemExit("Usage: python track.py <thư mục ảnh chứa các frame liên tiếp>")

    reader = PlateReader()
    image_paths = sorted(frames_dir.glob("*.jpg"))
    frame_detections = []
    for path in image_paths:
        detections = reader.read(str(path))
        frame_detections.append(detections)
        print(path.name, detections)

    tracks = track_and_vote(frame_detections)
    print("\n--- Kết quả voting theo track ---")
    for i, track in enumerate(tracks):
        print(f"Track {i}: các frame đọc được {track['texts']} -> voted: '{track['voted_text']}'")
