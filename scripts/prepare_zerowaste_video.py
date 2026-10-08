# prepare_zerowaste_video.py - streams the zerowaste conveyor-video frames from zenodo and writes small jpeg images with yolo segmentation labels
"""ZeroWaste-f (Bashkirova et al., 2022): frames of real conveyor-belt video from a waste sorting plant, with a polygon mask for every
cardboard, soft plastic, rigid plastic and metal object. The zip on zenodo is 7.5 GB, so the files are read straight from the web
(range requests), shrunk to 640 pixels wide and saved as jpeg. The big png files are never stored.

usage:  python scripts/prepare_zerowaste_video.py                    # all three splits (train, val, test)
        python scripts/prepare_zerowaste_video.py --splits test      # only the test split
output: data/video/zerowaste/<split>/images, .../labels (yolo segmentation format), data/video/zerowaste/data.yaml
frame names look like 01_frame_001160.PNG: the number before _frame_ is the video, the last number is the frame in that video, so
frames of one video can be played back in order."""
import argparse
import io
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/6412647/files/zerowaste-f-final.zip/content"
PREFIX = "splits_final_deblurred"
OUT = Path("data/video/zerowaste")
NAMES = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]     # coco category id 1..4 -> yolo class 0..3
WIDTH = 640
local = threading.local()


def zip_handle():
    if not hasattr(local, "z"):
        local.z = RemoteZip(URL)
    return local.z


def to_yolo(ann, w, h):
    """one polygon (the largest part) as a yolo segmentation line, or None."""
    parts = [p for p in ann["segmentation"] if len(p) >= 6]
    if not parts:
        return None
    poly = max(parts, key=len)
    pts = [f"{min(max(v / (w if i % 2 == 0 else h), 0), 1):.5f}" for i, v in enumerate(poly)]
    return f"{ann['category_id'] - 1} " + " ".join(pts)


def one(split, img, anns):
    stem = Path(img["file_name"]).stem
    jpg = OUT / split / "images" / f"{stem}.jpg"
    txt = OUT / split / "labels" / f"{stem}.txt"
    if jpg.exists() and txt.exists():
        return 0
    data = zip_handle().read(f"{PREFIX}/{split}/data/{img['file_name']}")
    im = Image.open(io.BytesIO(data)).convert("RGB")
    h = round(im.height * WIDTH / im.width)
    im.resize((WIDTH, h), Image.LANCZOS).save(jpg, quality=92)
    lines = [l for l in (to_yolo(a, img["width"], img["height"]) for a in anns) if l]
    txt.write_text("\n".join(lines))
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train-stride", type=int, default=3, help="keep every n-th training frame (neighbouring frames are almost the same picture)")
    ap.add_argument("--val-stride", type=int, default=2, help="keep every n-th validation frame")
    a = ap.parse_args()
    for split in a.splits:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        with RemoteZip(URL) as z:
            coco = json.loads(z.read(f"{PREFIX}/{split}/labels.json"))
        by_img = {}
        for an in coco["annotations"]:
            by_img.setdefault(an["image_id"], []).append(an)
        imgs = coco["images"]
        if split == "train" and a.train_stride > 1:
            imgs = imgs[:: a.train_stride]
        if split == "val" and a.val_stride > 1:
            imgs = imgs[:: a.val_stride]
        if split == "train":
            random.Random(0).shuffle(imgs)         # so that a half-finished download still covers every video
        done = 0
        with ThreadPoolExecutor(a.threads) as ex:
            futs = [ex.submit(one, split, im, by_img.get(im["id"], [])) for im in imgs]
            for k, f in enumerate(futs, 1):
                done += f.result()
                if k % 100 == 0 or k == len(futs):
                    print(f"{split}: {k}/{len(futs)} frames ready", flush=True)
    (OUT / "data.yaml").write_text("path: " + str(OUT.resolve()).replace("\\", "/") + "\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n" +
                                   "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("done; labels follow yolo segmentation format; classes:", NAMES)


if __name__ == "__main__":
    main()
