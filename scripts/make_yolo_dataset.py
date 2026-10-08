import json, os, shutil, sys
from pathlib import Path

# Converts the COCO label folders in labels/ to the YOLO format Ultralytics trains on, so a YOLO model can learn from exactly the
# same clips and boxes as an RF-DETR model. Class index = COCO category id minus 1, in the same order RF-DETR uses.
# As in make_train_dataset.py, the valid split is a copy of the training pictures (a progress check, not a test).
# usage: python make_yolo_dataset.py <out_dir> <clip,clip,...>      e.g. clip1,clip2,clip3,night,intersection,quiet_street
# A clip can name an older label file as clip@file.json, to train on the labels as they were when another model was trained.
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
clips = sys.argv[2].split(",")
shutil.rmtree(OUT, ignore_errors=True)
names, counts = {}, {}
for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)
    n_img = n_box = 0
    for spec in clips:
        clip, _, label_file = spec.partition("@")
        src = ROOT / "labels" / clip
        coco = json.load(open(src / (label_file or "annotations.coco.json")))
        for c in coco["categories"]:
            names[c["id"] - 1] = c["name"]
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            stem = f"{clip}_{Path(im['file_name']).stem}"
            os.link(src / "images" / im["file_name"], OUT / "images" / split / f"{stem}.jpg")
            lines = []
            for a in by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                lines.append(f"{a['category_id'] - 1} {(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            n_img += 1; n_box += len(lines)
    counts[split] = (n_img, n_box)
(OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {names[i]}\n" for i in sorted(names)))
print({k: f"{v[0]} images, {v[1]} boxes" for k, v in counts.items()}, "| classes:", [names[i] for i in sorted(names)])
