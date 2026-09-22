"""Video survey pipeline: detect, track, type and inventory litter in footage.

The camera pipeline in `pipeline.py` answers "what is in front of me right
now". This module answers a different question: "given a video of a scene —
a beach survey walk, a conveyor run, a drone pass — what litter is actually
*there*, and how much of it?"

The hard part is not detection but *counting*. A single physical item
appears in hundreds of consecutive frames; naively summing per-frame
detections overcounts by orders of magnitude. So each detection is assigned
a persistent track ID (ByteTrack, via ultralytics), and the inventory is
aggregated per track, not per frame — one physical item contributes exactly
one inventory row no matter how long it stays in view.

Per track we keep:
  * the object class the detector is most confident about across its life
    (majority vote weighted by confidence — robust to a few bad frames),
  * the material class from the Stage-1 classifier on its best-quality crop,
  * MC-Dropout uncertainty and the OCI contamination score for that crop,
  * first/last frame and how many frames it was visible.

Usage:
    edgewaste-video --source beach.mp4 --det-ckpt ... --cls-ckpt ...
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .confidence import mc_dropout_predict
from .decision import decide
from .identity_prior import apply_identity_prior, explain as explain_prior, is_consistent
from .infer import load_for_inference
from .oci import compute_oci
from .sensors import fit_demo_oci_model, reading_to_features, simulate_sensors
from .utils import pick_device

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class Track:
    """Everything we learn about one physical item over its time on screen."""

    track_id: int
    first_frame: int
    last_frame: int = 0
    n_frames: int = 0
    # class_name -> summed detector confidence, for a confidence-weighted vote
    object_votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    best_crop_area: int = 0
    best_crop: Image.Image | None = None
    material: str = ""
    material_conf: float = 0.0
    material_raw: str = ""  # classifier's own call, before the identity prior
    prior_corrected: bool = False
    consistent: bool = True
    uncertainty: float = float("nan")
    oci_score: float | None = None
    route: str = ""

    @property
    def object_class(self) -> str:
        if not self.object_votes:
            return "unknown"
        return max(self.object_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def object_conf(self) -> float:
        if not self.object_votes or self.n_frames == 0:
            return 0.0
        return self.object_votes[self.object_class] / self.n_frames


def _classify_track_crops(tracks: dict[int, Track], classifier, tfm, class_names, device,
                           oci_model, mc_passes: int, rng) -> None:
    """Run the material classifier once per track, on its best crop.

    Deliberately deferred to the end rather than run per frame: the
    classifier is the expensive stage (50.7M params vs the detector's 2.4M),
    and only one crop per physical item actually matters. Using the largest
    crop the track ever produced picks the frame where the item was closest
    to the camera, i.e. the most informative view.
    """
    for track in tracks.values():
        if track.best_crop is None:
            continue
        x = tfm(track.best_crop).unsqueeze(0).to(device)
        mean_probs, uncertainty = mc_dropout_predict(classifier, x, n_passes=mc_passes)
        raw_idx = int(mean_probs[0].argmax())
        track.material_raw = class_names[raw_idx]

        # Cross-validate the classifier against what the detector says the
        # object *is* — see identity_prior.py.
        adjusted = apply_identity_prior(mean_probs[0], track.object_class, class_names)
        conf, idx = adjusted.max(0)
        track.material = class_names[int(idx)]
        track.material_conf = float(conf)
        track.prior_corrected = track.material != track.material_raw
        track.consistent = is_consistent(track.object_class, track.material)
        track.uncertainty = float(uncertainty[0])

        if oci_model is not None:
            reading = simulate_sensors(track.material, rng=rng)
            f_m, f_g = reading_to_features(reading)
            track.oci_score = compute_oci(oci_model, f_m, f_g)

        decision = decide(track.material, track.material_conf, track.uncertainty,
                           track.oci_score)
        # A surviving object/material contradiction means one of the two
        # stages is wrong on this item; don't sort it on a coin flip.
        track.route = "manual_review" if not track.consistent else decision.route


def run_video(
    cfg: Config, source: str, det_ckpt: str, cls_ckpt: str, out_dir: str,
    conf_threshold: float | None = None, mc_passes: int = 8,
    save_video: bool = True, max_frames: int | None = None,
) -> dict[int, Track]:
    import cv2
    from ultralytics import YOLO

    device = pick_device()
    classifier, class_names, tfm = load_for_inference(cls_ckpt, cfg, device)
    detector = YOLO(det_ckpt)
    conf = conf_threshold if conf_threshold is not None else cfg.detect.conf_threshold
    rng = np.random.default_rng(cfg.data.seed)

    oci_model = fit_demo_oci_model()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    print(f"Source: {source}  ({width}x{height} @ {fps:.1f}fps, {total} frames)")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), fourcc, fps, (width, height))

    tracks: dict[int, Track] = {}
    # persist=True keeps ByteTrack's state across calls so IDs stay stable.
    stream = detector.track(source=source, stream=True, persist=True,
                             conf=conf, tracker="bytetrack.yaml", verbose=False)

    det_names = detector.names
    frame_idx = 0
    for result in stream:
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break
        frame = result.orig_img
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            for xyxy, tid, cls_idx, det_conf in zip(
                boxes.xyxy.tolist(), boxes.id.tolist(),
                boxes.cls.tolist(), boxes.conf.tolist(),
            ):
                tid = int(tid)
                x1, y1, x2, y2 = (max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                                   min(width, int(xyxy[2])), min(height, int(xyxy[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                name = det_names[int(cls_idx)]
                track = tracks.get(tid)
                if track is None:
                    track = Track(track_id=tid, first_frame=frame_idx)
                    tracks[tid] = track
                track.last_frame = frame_idx
                track.n_frames += 1
                track.object_votes[name] += float(det_conf)

                area = (x2 - x1) * (y2 - y1)
                if area > track.best_crop_area:
                    track.best_crop_area = area
                    track.best_crop = pil.crop((x1, y1, x2, y2))

                if writer is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"#{tid} {name} {det_conf*100:.0f}%",
                                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if writer is not None:
            writer.write(frame)
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}  |  {len(tracks)} unique items so far")

    if writer is not None:
        writer.release()

    print(f"\nTracked {len(tracks)} unique items across {frame_idx} frames.")
    print("Classifying material for each tracked item...")
    _classify_track_crops(tracks, classifier, tfm, class_names, device,
                           oci_model, mc_passes, rng)

    _write_reports(tracks, out, source, frame_idx)
    return tracks


def _write_reports(tracks: dict[int, Track], out: Path, source: str, n_frames: int) -> None:
    rows_path = out / "tracks.csv"
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "object_class", "object_conf", "material",
                          "material_raw", "prior_corrected", "consistent",
                          "material_conf", "uncertainty", "oci_score", "route",
                          "first_frame", "last_frame", "frames_visible"])
        for t in sorted(tracks.values(), key=lambda t: t.first_frame):
            writer.writerow([
                t.track_id, t.object_class, f"{t.object_conf:.3f}", t.material,
                t.material_raw, int(t.prior_corrected), int(t.consistent),
                f"{t.material_conf:.3f}", f"{t.uncertainty:.3f}",
                "" if t.oci_score is None else f"{t.oci_score:.3f}",
                t.route, t.first_frame, t.last_frame, t.n_frames,
            ])

    by_object: dict[str, int] = defaultdict(int)
    by_material: dict[str, int] = defaultdict(int)
    by_route: dict[str, int] = defaultdict(int)
    contaminated = 0
    corrected = 0
    contradictions: list[str] = []
    for t in tracks.values():
        by_object[t.object_class] += 1
        if t.material:
            by_material[t.material] += 1
        if t.route:
            by_route[t.route] += 1
        if t.oci_score is not None and t.oci_score >= 0.5:
            contaminated += 1
        if t.prior_corrected:
            corrected += 1
        if t.material and not t.consistent:
            contradictions.append(f"  #{t.track_id:<5} "
                                   + explain_prior(t.object_class, t.material))

    lines = [
        "LITTER INVENTORY REPORT",
        "=" * 46,
        f"Source           : {source}",
        f"Generated        : {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Frames processed : {n_frames}",
        f"Unique items     : {len(tracks)}   (de-duplicated across frames by track ID)",
        f"Flagged contaminated (OCI >= 0.50): {contaminated}",
        f"Material corrected by identity prior: {corrected}",
        f"Unresolved object/material contradictions: {len(contradictions)}",
        "",
        "BY OBJECT TYPE",
        "-" * 46,
    ]
    for name, n in sorted(by_object.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<28} {n:>5}")
    lines += ["", "BY MATERIAL", "-" * 46]
    for name, n in sorted(by_material.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<28} {n:>5}")
    lines += ["", "BY ROUTING DECISION", "-" * 46]
    for name, n in sorted(by_route.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:<28} {n:>5}")
    if contradictions:
        lines += ["", "OBJECT/MATERIAL CONTRADICTIONS (routed to manual review)", "-" * 46]
        lines += contradictions

    report = "\n".join(lines)
    (out / "inventory.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {rows_path}, {out / 'inventory.txt'}"
          + (f", {out / 'annotated.mp4'}" if (out / 'annotated.mp4').exists() else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Video litter survey with tracking + inventory.")
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--source", required=True, help="Video file path (or a camera index).")
    # Ultralytics nests project/ under its own runs_dir, hence the doubled path.
    ap.add_argument("--det-ckpt",
                     default="runs/detect/runs/detect/taco_multiclass/weights/best.pt")
    ap.add_argument("--cls-ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--out-dir", default=None, help="Defaults to runs/video/<timestamp>.")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--mc-passes", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="Skip writing annotated.mp4.")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or f"runs/video/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg = Config.load(args.config)
    run_video(cfg, args.source, args.det_ckpt, args.cls_ckpt, out_dir,
               conf_threshold=args.conf, mc_passes=args.mc_passes,
               save_video=not args.no_video, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
