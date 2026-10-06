import json, os, sys
from pathlib import Path
import cv2
import supervision as sv
from rfdetr import RFDETRSmall

# First pass on a NEW clip using the fine-tuned model (it already knows rickshaws, so the guesses are better
# than the original pipeline's). Writes stills plus a COCO file in the same format the rest of the repo uses.
# usage: python prelabel_finetuned.py <video.mp4> <out_dir> <seconds_between_stills> [ranges]
# ranges: "5-12,35-42" pulls only those stretches (still numbers jump by 10 between stretches so the review tool never links across them)
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = os.environ.get("WEIGHTS", str(ROOT / "runs/small/checkpoint_best_total.pth"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.4"))   # lower (0.2) on crowded clips, so the review page is offered more candidates
CLASSES = ["person", "bicycle", "motorcycle", "rickshaw", "cart", "car"]  # COCO category id = index + 1

src, out, gap = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
info = sv.VideoInfo.from_video_path(src)
total = info.total_frames / info.fps
ranges = [tuple(float(v) for v in r.split("-")) for r in sys.argv[4].split(",")] if len(sys.argv) > 4 else [(0.0, total)]
(out / "images").mkdir(parents=True, exist_ok=True)
model = RFDETRSmall(pretrain_weights=WEIGHTS)
cap = cv2.VideoCapture(src)
images, annos = [], []
n = 0
for start, end in ranges:
    t = start
    while t < min(end, total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info.fps))
        ok, frame = cap.read()
        if not ok:
            break
        name = f"frame_{n:04d}.jpg"
        cv2.imwrite(str(out / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"id": n, "file_name": name, "width": frame.shape[1], "height": frame.shape[0], "second": round(t, 2)})
        d = model.predict(frame, threshold=THRESHOLD)
        for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence):
            if int(c) >= len(CLASSES): continue          # an earlier checkpoint once returned a class id past the last real class (not reproduced since); skip it
            annos.append({"id": len(annos) + 1, "image_id": n, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0, "score": round(float(s), 3)})
        n += 1
        t += gap
    n += 10                                             # gap in the numbering between stretches
json.dump({"images": images, "annotations": annos, "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]},
          open(out / "annotations.coco.json", "w"))
print("stills:", len(images), "boxes:", len(annos))
