import sys
from pathlib import Path

# src/eval/ is a standalone script directory, not part of the installable
# plate_detector package, so it needs an explicit path for tests to import it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))
