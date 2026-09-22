"""Continue YOLO26n detector training from a checkpoint's weights.

Not `resume=True`: two resume attempts both crashed with Windows
ERROR_COMMITMENT_LIMIT (tiny allocations failing despite free RAM) traced to
`resume`'s locked-in workers=8 from the original run's args.yaml, which
overloads Windows' paging-file commit charge with too many DataLoader IPC
shared-memory mappings. This instead starts a fresh (shorter) training
schedule warm-started from the checkpoint's weights, with workers explicitly
capped low.
"""

from pathlib import Path

from ultralytics import YOLO

CKPT = "runs/detect/runs/detect/taco_single_class/weights/last.pt"


def main() -> None:
    ckpt = Path(CKPT)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found.")
    model = YOLO(str(ckpt))
    model.train(
        data="data/detect/taco/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        workers=2,
        project="runs/detect",
        name="taco_single_class_cont",
    )


if __name__ == "__main__":
    main()
