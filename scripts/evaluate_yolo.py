import json, os, sys, time, types
from pathlib import Path
import numpy as np, cv2

# Scores an Ultralytics YOLO model on a held-out clip with the same scoring code as evaluate.py (class-aware precision, recall and F1 at
# IoU 0.5, vehicles only, people left out), so a YOLO model and an RF-DETR model are judged by one function.
# usage: python evaluate_yolo.py <weights.pt> <holdout clip> <out.json>        (CONF=0.3 python ... to change the cutoff, default 0.5)
weights, holdout, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
CONF = float(os.environ.get("CONF", "0.5"))
argv = sys.argv
sys.argv = [argv[0]]                                  # evaluate.py reads sys.argv when imported
sys.modules.setdefault("rfdetr", types.SimpleNamespace(RFDETRSmall=None))   # it imports RF-DETR, which this script does not need
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev
sys.argv = argv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "labels" / holdout
coco, gt = ev.load_gt(str(D))
model = YOLO(weights)
preds, times = {}, []
for im in coco["images"]:
    frame = cv2.imread(str(D / "images" / im["file_name"]))
    t = time.time()
    r = model.predict(frame, conf=CONF, imgsz=640, device="mps", verbose=False)[0]
    times.append(time.time() - t)
    preds[im["id"]] = [{"category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
                       for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
                       if int(c) + 1 in ev.NAMES]
result = {"images": len(coco["images"]), "weights": os.path.basename(os.path.dirname(os.path.dirname(weights))) or weights, "conf": CONF,
          "finetuned": ev.score(gt, preds), "finetuned_confusion": ev.confusion(gt, preds),
          "ms_per_image_median_end_to_end": round(1000 * float(np.median(times[3:])), 1)}
baseline = D / "original_pipeline.coco.json"
if baseline.exists():
    base = ev.load_preds(str(baseline))
    result.update({"baseline": ev.score(gt, base), "baseline_confusion": ev.confusion(gt, base)})
json.dump({holdout: result}, open(out_json, "w"), indent=2)
for k in ("rickshaw", "motorcycle", "truck", "ALL VEHICLES"):
    v = result["finetuned"][k]
    print(f"{k:13s} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}  gt {v['gt']} tp {v['tp']} fp {v['fp']} fn {v['fn']}")
