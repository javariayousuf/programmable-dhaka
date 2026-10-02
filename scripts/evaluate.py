from pathlib import Path
import json, time, sys
import numpy as np, cv2
from rfdetr import RFDETRSmall

# Scores vehicle classes only (bicycle, motorcycle, rickshaw, cart) against the hand-corrected labels.
# Person is left out on purpose: the person labels were never hand-corrected, they are the pretrained
# detector's own output, so scoring them would only measure agreement with itself.
ROOT = str(Path(__file__).resolve().parent.parent)  # repo root
# usage: python evaluate.py [holdout clip] [weights] [out.json]   (the holdout clip must NOT be in the training set)
holdout = sys.argv[1] if len(sys.argv) > 1 else "clip1"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/runs/small/checkpoint_best_total.pth"
OUT_JSON = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/eval/results_{holdout}_holdout.json"
CLIPS = {holdout: {"gt": f"{ROOT}/labels/{holdout}", "baseline": f"{ROOT}/labels/{holdout}/original_pipeline.coco.json"}}
NAMES = {2: "bicycle", 3: "motorcycle", 4: "rickshaw", 5: "cart"}  # COCO category id -> class
CONF = 0.5  # same threshold the video pipeline uses
IOU = 0.5


def iou(a, b):  # boxes as [x, y, w, h]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def score(gt_by_img, pred_by_img):
    """Class-aware precision / recall at IoU 0.5, greedy matching by confidence."""
    out = {}
    for cid, name in NAMES.items():
        tp = fp = fn = 0
        for img, gts in gt_by_img.items():
            g = [x["bbox"] for x in gts if x["category_id"] == cid]
            p = sorted([x for x in pred_by_img.get(img, []) if x["category_id"] == cid], key=lambda x: -x["score"])
            used = set()
            for pr in p:
                best, bj = 0.0, None
                for j, gb in enumerate(g):
                    if j in used:
                        continue
                    s = iou(pr["bbox"], gb)
                    if s > best:
                        best, bj = s, j
                if bj is not None and best >= IOU:
                    used.add(bj); tp += 1
                else:
                    fp += 1
            fn += len(g) - len(used)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(P, 3), "recall": round(R, 3),
                     "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0, "gt": tp + fn}
    tp = sum(v["tp"] for v in out.values()); fp = sum(v["fp"] for v in out.values()); fn = sum(v["fn"] for v in out.values())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    out["ALL VEHICLES"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(P, 3), "recall": round(R, 3),
                           "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0, "gt": tp + fn}
    return out


def confusion(gt_by_img, pred_by_img):
    """For each true vehicle box: what did the model call it (best overlap at IoU 0.5, any vehicle class)?"""
    cols = list(NAMES.values()) + ["missed"]
    m = {n: {c: 0 for c in cols} for n in NAMES.values()}
    for img, gts in gt_by_img.items():
        for g in gts:
            best, label = 0.0, "missed"
            for p in pred_by_img.get(img, []):
                s = iou(g["bbox"], p["bbox"])
                if s >= IOU and s > best:
                    best, label = s, NAMES[p["category_id"]]
            m[NAMES[g["category_id"]]][label] += 1
    return m


def load_gt(folder):
    coco = json.load(open(f"{folder}/annotations.coco.json"))
    by = {i["id"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        if a["category_id"] in NAMES:
            by[a["image_id"]].append(a)
    return coco, by


def load_preds(path):
    coco = json.load(open(path))
    by = {}
    for a in coco["annotations"]:
        if a["category_id"] in NAMES:
            by.setdefault(a["image_id"], []).append(a)
    return by


def main():
    model = RFDETRSmall(pretrain_weights=WEIGHTS)
    results = {}
    for clip, cfg in CLIPS.items():
        coco, gt = load_gt(cfg["gt"])
        preds, times = {}, []
        for im in coco["images"]:
            frame = cv2.imread(f"{cfg['gt']}/images/{im['file_name']}")
            t = time.time()
            d = model.predict(frame, threshold=CONF)
            times.append(time.time() - t)
            preds[im["id"]] = [{"category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)}
                               for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence) if int(c) + 1 in NAMES]
        base = load_preds(cfg["baseline"])
        results[clip] = {
            "images": len(coco["images"]),
            "finetuned": score(gt, preds), "finetuned_confusion": confusion(gt, preds),
            "baseline": score(gt, base), "baseline_confusion": confusion(gt, base),
            "ms_per_image_median": round(1000 * float(np.median(times[3:])), 1),  # skip warm-up calls
        }
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    for clip, r in results.items():
        print(f"\n=== {clip} ({r['images']} images), median {r['ms_per_image_median']} ms/image")
        for tag in ("baseline", "finetuned"):
            print(f"  {tag}:")
            for name, v in r[tag].items():
                print(f"    {name:13s} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}   (true boxes {v['gt']}, tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
        for tag in ("baseline_confusion", "finetuned_confusion"):
            print(f"  {tag} (rows = truth, columns = what it was called):")
            for row, cols in r[tag].items():
                print(f"    {row:11s} " + "  ".join(f"{k}={v}" for k, v in cols.items()))


if __name__ == "__main__":
    main()
