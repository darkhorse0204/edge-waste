"""Structured per-detection logging. Additive — importing this doesn't change
run_camera/run_images behavior unless wired in via --log-csv."""

from __future__ import annotations

import csv
import time
from pathlib import Path


class PredictionLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.log_path.exists()
        self._file = open(self.log_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            # raw_* are the pre-smoothing, single-frame values; class_name /
            # cls_conf are what was displayed. Logging both means the flicker
            # reduction can be measured after the fact — count label changes
            # in raw_class_name vs class_name over the same session.
            self._writer.writerow([
                "timestamp", "track_id", "class_name", "cls_conf",
                "raw_class_name", "raw_cls_conf", "det_conf",
                "object_name", "object_conf", "decided_by",
                "x1", "y1", "x2", "y2", "attn_convnext", "attn_vit",
                "votes", "window",
            ])

    def log(self, det) -> None:
        x1, y1, x2, y2 = det.box
        a_cnx, a_vit = det.attn if det.attn else ("", "")
        # getattr defaults keep this working with any Detection that predates
        # the stability fields.
        self._writer.writerow([
            time.time(), getattr(det, "track_id", -1),
            det.class_name, f"{det.cls_conf:.4f}",
            getattr(det, "raw_class_name", ""),
            f"{getattr(det, 'raw_cls_conf', 0.0):.4f}",
            f"{det.det_conf:.4f}",
            getattr(det, "object_name", ""),
            f"{getattr(det, 'object_conf', 0.0):.4f}",
            getattr(det, "decided_by", ""),
            x1, y1, x2, y2, a_cnx, a_vit,
            getattr(det, "votes", ""), getattr(det, "window", ""),
        ])
        self._file.flush()

    def close(self) -> None:
        self._file.close()