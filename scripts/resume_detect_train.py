"""Resume YOLO26n detector training from its last checkpoint.

Windows multiprocessing (spawn) requires the entry point to be a real
module guarded by `if __name__ == "__main__":`, not an inline `python -c`
script — running it inline caused worker processes to mis-spawn and crash
with a spurious MemoryError.
"""

from pathlib import Path

from ultralytics import YOLO

LAST_CKPT = "runs/detect/runs/detect/taco_single_class/weights/last.pt"


def main() -> None:
    ckpt = Path(LAST_CKPT)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found.")
    model = YOLO(str(ckpt))
    model.train(resume=True)


if __name__ == "__main__":
    main()
