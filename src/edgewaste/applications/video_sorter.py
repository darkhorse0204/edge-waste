# video_sorter.py - video or live camera: finds, segments, tracks, classifies and sorts every waste item, and draws the result
"""One program for a video file, a folder of ordered frames, or a live camera.

For every frame:
  1. a segmentation model trained on real conveyor video (ZeroWaste) finds each item, gives its exact outline (mask) and a coarse
     material (cardboard, metal, rigid plastic, soft plastic). ByteTrack gives the item a number that stays the same between frames.
     (if only a detection model is given, the Segment Anything Model can draw the outline: --sam)
  2. the 33-class hybrid classifier looks at the masked crop of each item and gives fine class probabilities. the probabilities of an
     item are averaged over its frames (temporal smoothing), and Monte Carlo dropout gives its uncertainty.
  3. decision, with the project's safety rule kept:
       - if the classifier's calibrated family set contains "hazardous", the item is never sent to a recycling bin
         (hazardous bin, or priority review if unsure), whatever the segmenter says;
       - otherwise, if the video-trained segmenter is confident, its material gives the bin (the 33-class classifier was trained on
         photos, so on conveyor video the video-trained model is the better judge of material); the classifier gives the fine class;
       - otherwise the project's decision engine decides (usually manual review).
  4. the outline of each item is filled with the colour of its bin and labelled; an annotated video and csv files are written.

usage:  edgewaste-video-sorter --source clip.mp4 --out runs/video_sorter/clip
        edgewaste-video-sorter --source 0                      # live camera, window closes with q
        edgewaste-video-sorter --source data/video/zerowaste/test/images --frames-pattern 01_frame_*.jpg"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from edgewaste.classification.run_inference import load_for_inference
from edgewaste.common_utils import pick_device
from edgewaste.config import Config
from edgewaste.decision_engine.conformal_routing import prediction_sets
from edgewaste.decision_engine.routing_rules import decide
from edgewaste.decision_engine.startup_calibration import calibrate_from_cache, mc_head_uncertainty
from edgewaste.taxonomy import CLASS_TO_FAMILY, FAMILIES, FAMILY_TO_CLASSES

# video-trained material classes (ZeroWaste) -> project families
FAMILY_OF_SEG = {"rigid_plastic": "plastic", "soft_plastic": "plastic", "cardboard": "cardboard", "metal": "metal"}
# bin colours (bgr)
COLOUR = {"plastic": (230, 160, 40), "paper": (60, 160, 250), "cardboard": (60, 120, 190), "glass": (200, 220, 100), "metal": (190, 190, 190),
          "organic": (80, 170, 80), "styrofoam": (220, 120, 220), "textile": (150, 90, 200), "hazardous": (40, 40, 220), "review": (0, 215, 255)}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class TrackState:
    first_frame: int
    frames: int = 0
    p_ema: np.ndarray | None = None           # smoothed class probabilities
    u_ema: float = 0.0                        # smoothed uncertainty
    seg_votes: dict = field(default_factory=lambda: defaultdict(float))
    seg_conf_sum: float = 0.0
    routes: Counter = field(default_factory=Counter)
    last: dict = field(default_factory=dict)


class VideoSorter:
    def __init__(self, seg_ckpt: str, cls_ckpt: str = "runs/stage1/best.pt", config: str = "configs/classifier_convnext_vit.yaml",
                 conf: float = 0.25, seg_trust: float = 0.40, haz_mass: float = 0.90, smooth: float = 0.6, mask_crop: bool = True, use_sam: bool = False,
                 alpha: float = 0.10, alpha_h: float = 0.05, review_budget: float = 0.10, imgsz: int = 640):
        from ultralytics import YOLO
        self.dev = pick_device()
        self.seg = YOLO(seg_ckpt)
        cfg = Config.load(config)
        cfg.model.pretrained = False                      # the checkpoint holds every weight, no download needed
        self.model, self.names, self.tfm = load_for_inference(cls_ckpt, cfg, self.dev)
        self.head = self.model.head
        self.q, self.tau_u, self.M = calibrate_from_cache(self.names, self.head, self.dev, alpha, alpha_h, review_budget)
        self.haz_mass = haz_mass
        self.conf, self.seg_trust, self.smooth, self.mask_crop, self.imgsz = conf, seg_trust, smooth, mask_crop, imgsz
        self.sam = None
        if use_sam:
            from edgewaste.detection.sam_segmentation import load_sam
            self.sam = load_sam()
        self.tracks: dict[int, TrackState] = {}
        self.frame_no = 0
        self._fake_id = 0
        self.fam_index = {f: i for i, f in enumerate(FAMILIES)}
        self.haz = self.fam_index["hazardous"]

    # ---------------------------------------------------------------------------------------------- one frame
    def _outline(self, r, i, frame):
        """polygon of detection i in pixel coordinates (from the segmentation model, or from the Segment Anything Model)."""
        if r.masks is not None:
            return r.masks.xy[i].astype(np.int32)
        if self.sam is not None:
            from edgewaste.detection.sam_segmentation import segment_box
            x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[i].tolist())
            seg = segment_box(self.sam, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (x1, y1, x2, y2))
            cs, _ = cv2.findContours(seg.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                return max(cs, key=cv2.contourArea)[:, 0, :].astype(np.int32)
        return None

    def _crop(self, frame, box, poly):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pw, ph = int(0.10 * (x2 - x1)), int(0.10 * (y2 - y1))
        x1, y1, x2, y2 = max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)
        crop = frame[y1:y2, x1:x2].copy()
        if self.mask_crop and poly is not None and crop.size:
            m = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillPoly(m, [poly - np.array([x1, y1])], 255)
            crop[m == 0] = (124, 116, 104)                  # grey: removes the belt and neighbouring items from the classifier input
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def _classify(self, crops):
        x = torch.stack([self.tfm(c) for c in crops]).to(self.dev)
        self.model.eval()
        emb = self.model.feature_vector(x)
        probs = torch.softmax(self.head(emb), 1).cpu().numpy().astype(np.float64)
        _, u = mc_head_uncertainty(self.head, emb)
        return probs, u.cpu().numpy()

    def process(self, frame):
        """returns (annotated frame, list of one dict per item)."""
        self.frame_no += 1
        r = self.seg.track(frame, persist=True, conf=self.conf, tracker="bytetrack.yaml", imgsz=self.imgsz, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        out, crops, keep = [], [], []
        for i in range(n):
            box = tuple(int(v) for v in r.boxes.xyxy[i].tolist())
            poly = self._outline(r, i, frame)
            if r.boxes.id is not None:
                tid = int(r.boxes.id[i])
            else:
                self._fake_id += 1
                tid = -self._fake_id
            crop = self._crop(frame, box, poly)
            if min(crop.size) < 8:
                continue
            keep.append((i, tid, box, poly)); crops.append(crop)
        if crops:
            probs, us = self._classify(crops)
            for (i, tid, box, poly), p, u in zip(keep, probs, us):
                out.append(self._decide(r, i, tid, box, poly, p, float(u)))
        return self._draw(frame, out), out

    # ---------------------------------------------------------------------------------------------- decision per item
    def _decide(self, r, i, tid, box, poly, p, u):
        t = self.tracks.setdefault(tid, TrackState(first_frame=self.frame_no))
        t.frames += 1
        t.p_ema = p if t.p_ema is None else self.smooth * t.p_ema + (1 - self.smooth) * p
        t.u_ema = u if t.frames == 1 else self.smooth * t.u_ema + (1 - self.smooth) * u
        seg_name = r.names[int(r.boxes.cls[i])]
        seg_conf = float(r.boxes.conf[i])
        t.seg_votes[seg_name] += seg_conf
        t.seg_conf_sum += seg_conf
        voted = max(t.seg_votes, key=t.seg_votes.get)
        voted_conf = t.seg_votes[voted] / t.frames
        seg_family = FAMILY_OF_SEG.get(voted)

        fam_mass = t.p_ema @ self.M
        fam_set = {FAMILIES[k] for k in np.where(prediction_sets(fam_mass[None, :], self.q)[0])[0]}
        top = int(t.p_ema.argmax())
        dec = decide(self.names[top], float(t.p_ema[top]), t.u_ema, None, uncertainty_threshold=self.tau_u, family_set=fam_set)

        if "hazardous" in fam_set and fam_mass[FAMILIES.index("hazardous")] >= self.haz_mass:                         # safety rule: never overridden by the segmenter
            route, family, why = dec.route, "hazardous", "hazard in the calibrated set: " + dec.reason
        elif seg_family is not None and voted_conf >= self.seg_trust:
            route, family, why = seg_family, seg_family, f"video-trained segmenter: {voted} ({voted_conf:.2f})"
        else:
            route, family, why = dec.route, dec.family, dec.reason
        # fine class: best class of the chosen family according to the classifier
        if family in self.fam_index and family != "review":
            members = [self.names.index(c) for c in FAMILY_TO_CLASSES[family]]
            fine = self.names[members[int(np.argmax(t.p_ema[members]))]]
        else:
            fine = self.names[top]
        t.routes[route] += 1
        rec = {"frame": self.frame_no, "track": tid, "segmenter_class": seg_name, "segmenter_conf": round(seg_conf, 3), "route": route, "family": family,
               "fine_class": fine, "classifier_top": self.names[top], "classifier_family": CLASS_TO_FAMILY[self.names[top]], "segmenter_family": seg_family, "uncertainty": round(t.u_ema, 3), "family_set": "|".join(sorted(fam_set)),
               "box": box, "poly": poly, "why": why}
        t.last = rec
        return rec

    # ---------------------------------------------------------------------------------------------- drawing
    def _draw(self, frame, items):
        canvas = frame.copy()
        layer = frame.copy()
        for it in items:
            col = COLOUR["review"] if it["route"] in ("manual_review", "contaminated_reject") else COLOUR.get(it["family"], (200, 200, 200))
            if it["poly"] is not None:
                cv2.fillPoly(layer, [it["poly"]], col)
        canvas = cv2.addWeighted(layer, 0.45, canvas, 0.55, 0)
        for it in items:
            col = COLOUR["review"] if it["route"] in ("manual_review", "contaminated_reject") else COLOUR.get(it["family"], (200, 200, 200))
            if it["poly"] is not None:
                cv2.polylines(canvas, [it["poly"]], True, col, 2)
            x1, y1, _, _ = it["box"]
            label = f"#{abs(it['track'])} {it['fine_class'].replace('_', ' ')} > {it['route'].replace('_', ' ')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(canvas, label, (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        bins = Counter(it["route"] for it in items)
        hud = f"frame {self.frame_no}   items {len(items)}   " + "  ".join(f"{k}:{v}" for k, v in bins.most_common(4))
        cv2.putText(canvas, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # ---------------------------------------------------------------------------------------------- sources
    def run(self, source: str, out_dir: str | None, show: bool = False, max_frames: int | None = None, fps: float = 5.0, pattern: str = "*"):
        src = Path(source) if not str(source).isdigit() else None
        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source)); frames = None
        elif src.is_dir():
            frames = sorted(p for p in src.glob(pattern) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}); cap = None
        else:
            cap = cv2.VideoCapture(str(src)); frames = None
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
        if cap is not None and not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        out = Path(out_dir) if out_dir else None
        writer, log_rows, i = None, [], 0
        t0 = time.time()
        while True:
            if frames is not None:
                if i >= len(frames):
                    break
                frame = cv2.imread(str(frames[i]))
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            i += 1
            canvas, items = self.process(frame)
            for it in items:
                log_rows.append({k: v for k, v in it.items() if k not in ("poly",)})
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            if show:
                cv2.imshow("video sorter (q to quit)", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and i >= max_frames:
                break
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        secs = max(time.time() - t0, 1e-6)
        print(f"{i} frames in {secs:.1f} s ({i / secs:.1f} frames per second), {len(self.tracks)} tracked items")
        if out is not None and i > 0:
            self._write_reports(out, log_rows)
        elif i == 0:
            print("no frames were read: check --source and --frames-pattern")
        return log_rows

    def _write_reports(self, out: Path, rows):
        if rows:
            with open(out / "detections.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        with open(out / "inventory.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track", "first_frame", "frames_seen", "final_route", "family", "fine_class", "segmenter_class", "mean_uncertainty"])
            for tid, t in sorted(self.tracks.items()):
                voted = max(t.seg_votes, key=t.seg_votes.get)
                w.writerow([abs(tid), t.first_frame, t.frames, t.routes.most_common(1)[0][0], t.last.get("family"), t.last.get("fine_class"), voted, round(t.u_ema, 3)])
        routes = Counter(t.routes.most_common(1)[0][0] for t in self.tracks.values())
        (out / "summary.txt").write_text(f"items tracked: {len(self.tracks)}\nfinal route per item: {dict(routes)}\n")
        print("wrote", out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, folder of frames, or a camera index such as 0")
    ap.add_argument("--seg-ckpt", default="runs/video_seg/zerowaste_seg/weights/best.pt")
    ap.add_argument("--cls-ckpt", default="runs/stage1/best.pt")
    ap.add_argument("--out", default=None, help="folder for annotated.mp4 and the csv files")
    ap.add_argument("--show", action="store_true", help="open a window (always use it with a live camera)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--haz-mass", type=float, default=0.90, help="classifier probability of the hazardous family needed (besides the conformal set) to bar recycling bins")
    ap.add_argument("--seg-trust", type=float, default=0.40, help="how sure the video-trained segmenter must be to decide the bin")
    ap.add_argument("--no-mask-crop", action="store_true")
    ap.add_argument("--sam", action="store_true", help="draw outlines with the Segment Anything Model when the detector gives only boxes")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=5.0, help="frame rate of the output when the source is a folder of frames")
    ap.add_argument("--frames-pattern", default="*", help="file pattern when the source is a folder, for example 01_frame_*.jpg")
    a = ap.parse_args(argv)
    s = VideoSorter(a.seg_ckpt, a.cls_ckpt, conf=a.conf, seg_trust=a.seg_trust, haz_mass=a.haz_mass, mask_crop=not a.no_mask_crop, use_sam=a.sam)
    s.run(a.source, a.out, show=a.show or str(a.source).isdigit(), max_frames=a.max_frames, fps=a.fps, pattern=a.frames_pattern)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
