"""Train the 18-class TACO litter detector.

Supersedes the single-class 'waste_item' localizer: the original TACO YOLO
mirror ships real per-object litter categories (cigarette, bottle, can,
straw, styrofoam piece, ...) which the old single-class prep threw away by
forcing every label to index 0. Keeping them gives object-level identity in
one pass, which is what a dense scene (a beach survey frame with dozens of
mixed items) actually needs — the material classifier then types each crop,
so the two stages are complementary rather than redundant.

workers is capped low: Windows hits ERROR_COMMITMENT_LIMIT (paging-file
commit charge, not RAM) with the ultralytics default of 8.
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "data/detect/taco_raw/data_multiclass.yaml"


def main() -> None:
    data = Path(DATA_YAML)
    if not data.exists():
        raise SystemExit(f"{data} not found.")
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(data),
        epochs=60,
        imgsz=640,
        batch=16,
        workers=2,
        project="runs/detect",
        name="taco_multiclass",
    )


if __name__ == "__main__":
    main()
