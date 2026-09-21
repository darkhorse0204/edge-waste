"""Per-detection CSV logging for live pipeline sessions.

Every detection (box, class, confidences, attention split, OCI/decision if
computed) gets one flushed CSV row, so a session directory becomes a
self-contained, analysable record — the raw material for the live-session
evidence a report needs (mean confidence, class distribution, attention
drift) without re-running anything.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class DetectionRecord:
    timestamp: str
    frame_idx: int
    class_name: str
    cls_conf: float
    det_conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    attn_convnext: float = float("nan")
    attn_vit: float = float("nan")
    uncertainty: float = float("nan")
    oci_score: float = float("nan")
    decision: str = ""


class PredictionLogger:
    """Appends `DetectionRecord`s to a CSV, flushing after every row."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = [f.name for f in fields(DetectionRecord)]
        is_new = not self.csv_path.exists()
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, record: DetectionRecord) -> None:
        self._writer.writerow(asdict(record))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "PredictionLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
