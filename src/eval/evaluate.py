import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "eval" / "ground_truth.json"

sys.path.insert(0, str(REPO_ROOT / "src" / "inference"))
from detect import detect_plates
from ocr import read_plate
from track import iou


def char_accuracy(pred: str, truth: str) -> float:
    if not truth:
        return 0.0
    matches = sum(1 for p, t in zip(pred, truth) if p == t)
    return matches / len(truth)


def evaluate() -> list[dict]:
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    results = []
    for entry in ground_truth:
        image_path = REPO_ROOT / entry["image"]
        detections = detect_plates(str(image_path))
        # match by IoU against the labeled bbox, not just highest confidence -- some
        # images contain multiple plates/false positives, so confidence alone can
        # pick the wrong one
        best = max(detections, key=lambda d: iou(d["bbox"], entry["bbox"]), default=None)
        pred = read_plate(str(image_path), best["bbox"]) if best else ""

        results.append(
            {
                "image": entry["image"],
                "truth": entry["text"],
                "pred": pred,
                "exact_match": pred == entry["text"],
                "char_accuracy": round(char_accuracy(pred, entry["text"]), 3),
            }
        )
    return results


if __name__ == "__main__":
    results = evaluate()

    print(f"{'Image':<58} {'Truth':<12} {'Pred':<12} {'Exact':<7} CharAcc")
    for r in results:
        print(f"{r['image']:<58} {r['truth']:<12} {r['pred']:<12} {str(r['exact_match']):<7} {r['char_accuracy']}")

    n = len(results)
    exact = sum(r["exact_match"] for r in results)
    mean_char_acc = sum(r["char_accuracy"] for r in results) / n
    print()
    print(f"Exact-match accuracy: {exact}/{n} = {exact / n:.1%}")
    print(f"Mean char accuracy:   {mean_char_acc:.1%}")
