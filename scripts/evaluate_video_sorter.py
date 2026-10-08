# evaluate_video_sorter.py - measures the video sorter on the zerowaste test videos: segmentation scores, material accuracy per item, false hazard calls, tracking
"""usage: python scripts/evaluate_video_sorter.py [--split test] [--seg-ckpt runs/video_seg/zerowaste_seg/weights/best.pt]
writes reports/video/evaluation.json and reports/video/evaluation.md.

what is measured
  A. box and mask scores of the segmentation model on the split (ultralytics validation: mAP50, mAP50-95, per class)
  B. the whole pipeline, run on every video of the split in frame order. each detection is matched to a ground-truth item by mask overlap (iou >= 0.5);
     for matched items the material family chosen by (1) the segmenter alone, (2) the 33-class classifier alone, (3) the fused decision is compared with the label.
     also: share of ground-truth items found, false hazard routes (the dataset has no hazardous items, so every hazard route is a false alarm), tracks per video."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edgewaste.applications.video_sorter import FAMILY_OF_SEG, VideoSorter  # noqa: E402

NAMES = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]


def read_gt(label_path: Path, w: int, h: int):
    """list of (family, class name, mask) for one frame."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().split("\n"):
        v = line.split()
        if len(v) < 7:
            continue
        pts = (np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        out.append((FAMILY_OF_SEG[NAMES[int(v[0])]], NAMES[int(v[0])], m))
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seg-ckpt", default=str(ROOT / "runs/video_seg/zerowaste_seg/weights/best.pt"))
    ap.add_argument("--cls-ckpt", default=str(ROOT / "runs/stage1/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/video/zerowaste/data.yaml"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--haz-mass", type=float, default=0.90)
    ap.add_argument("--seg-trust", type=float, default=0.40)
    ap.add_argument("--max-frames", type=int, default=None, help="per video, for a quick test")
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports/video"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    if not a.skip_val:
        from ultralytics import YOLO
        m = YOLO(a.seg_ckpt).val(data=a.data, split=a.split, imgsz=640, batch=8, workers=2, verbose=False, plots=False)
        res["segmentation_scores"] = {"box_mAP50": float(m.box.map50), "box_mAP50_95": float(m.box.map), "mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map),
                                      "per_class_mask_mAP50": {NAMES[int(c)]: float(v) for c, v in zip(m.ap_class_index, m.seg.ap50)}}
        print(json.dumps(res["segmentation_scores"], indent=1))

    img_dir = ROOT / "data/video/zerowaste" / a.split / "images"
    lab_dir = ROOT / "data/video/zerowaste" / a.split / "labels"
    videos = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        videos[p.name.split("_frame_")[0]].append(p)
    tot = Counter()
    per_video = {}
    conf_fused = Counter()
    for vid, files in sorted(videos.items()):
        if a.max_frames:
            files = files[: a.max_frames]
        sorter = VideoSorter(a.seg_ckpt, a.cls_ckpt, conf=a.conf, seg_trust=a.seg_trust, haz_mass=a.haz_mass)       # fresh tracker for every video
        n_gt = n_found = 0
        for f in files:
            frame = cv2.imread(str(f))
            h, w = frame.shape[:2]
            _, items = sorter.process(frame)
            gts = read_gt(lab_dir / (f.stem + ".txt"), w, h)
            n_gt += len(gts)
            used = set()
            for it in items:
                tot["detections"] += 1
                if "hazardous" in it["family_set"].split("|"):
                    tot["false_hazard_flags"] += 1
                if it["route"] == "hazardous":
                    tot["false_hazard_bin"] += 1
                if it["poly"] is None:
                    continue
                dm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(dm, [it["poly"]], 1)
                best, bi = 0.0, -1
                for gi, (fam, name, gm) in enumerate(gts):
                    v = iou(dm, gm)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5:
                    fam, name, _ = gts[bi]
                    tot["matched"] += 1
                    if bi not in used:
                        used.add(bi)
                        n_found += 1
                    tot["seg_correct"] += int(it["segmenter_family"] == fam)
                    tot["clf_correct"] += int(it["classifier_family"] == fam)
                    tot["fused_correct"] += int(it["family"] == fam)
                    conf_fused[(fam, it["family"])] += 1
        trk = sorter.tracks
        per_video[vid] = {"frames": len(files), "gt_items": n_gt, "gt_items_found": n_found, "tracks": len(trk),
                          "mean_track_length": float(np.mean([t.frames for t in trk.values()])) if trk else 0.0}
        tot["gt_items"] += n_gt
        tot["gt_found"] += n_found
        print(vid, per_video[vid])
    m = max(tot["matched"], 1)
    res["pipeline"] = {"frames": sum(v["frames"] for v in per_video.values()), "detections": tot["detections"], "matched_to_ground_truth": tot["matched"],
                       "ground_truth_items_found_share": tot["gt_found"] / max(tot["gt_items"], 1),
                       "family_accuracy_segmenter_only": tot["seg_correct"] / m, "family_accuracy_33class_classifier_only": tot["clf_correct"] / m,
                       "family_accuracy_fused_decision": tot["fused_correct"] / m,
                       "detections_with_hazardous_in_set": tot["false_hazard_flags"], "detections_sent_to_hazardous_bin": tot["false_hazard_bin"],
                       "per_video": per_video, "fused_confusion": {f"{k[0]} -> {k[1]}": v for k, v in conf_fused.most_common()}}
    (out / "evaluation.json").write_text(json.dumps(res, indent=1))
    p = res["pipeline"]
    md = ["# video sorter evaluation (zerowaste " + a.split + " videos)", "", "| quantity | value |", "|---|---|"]
    if "segmentation_scores" in res:
        s = res["segmentation_scores"]
        md += [f"| box mAP50 / mAP50-95 | {s['box_mAP50']:.3f} / {s['box_mAP50_95']:.3f} |", f"| mask mAP50 / mAP50-95 | {s['mask_mAP50']:.3f} / {s['mask_mAP50_95']:.3f} |"]
    md += [f"| frames, detections | {p['frames']}, {p['detections']} |", f"| ground-truth items found | {p['ground_truth_items_found_share'] * 100:.1f}% |",
           f"| family accuracy, segmenter only | {p['family_accuracy_segmenter_only'] * 100:.1f}% |", f"| family accuracy, 33-class classifier only | {p['family_accuracy_33class_classifier_only'] * 100:.1f}% |",
           f"| family accuracy, fused decision | {p['family_accuracy_fused_decision'] * 100:.1f}% |", f"| detections with hazardous in the family set | {p['detections_with_hazardous_in_set']} |",
           f"| detections sent to the hazardous bin (all false) | {p['detections_sent_to_hazardous_bin']} |"]
    (out / "evaluation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
