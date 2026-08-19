import shutil
from pathlib import Path

import kagglehub

DATASET = "bomaich/vnlicenseplate"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "vnlicenseplate"


def download() -> Path:
    if RAW_DIR.exists():
        print(f"Already downloaded at {RAW_DIR}, skipping.")
        return RAW_DIR

    cache_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, RAW_DIR)
    print(f"Dataset copied to {RAW_DIR}")
    return RAW_DIR


if __name__ == "__main__":
    download()
