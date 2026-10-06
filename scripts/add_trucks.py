from pathlib import Path
import json, shutil, sys
import cv2
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

# Adds a 7th class, "truck" (buses go in it too), so the model has a name for large vehicles instead of
# calling them a rickshaw. First-pass boxes come from the pretrained detector (truck, bus). Where a box lands on
# a vehicle that already has a label, that label is switched to "truck" and the old name is kept in "was", so
# review_cards.py can show it and you can put it back. Nothing here is final until you review it.
# usage: python add_trucks.py clip3 night rain      (folders under labels/; safe to re-run)
ROOT = str(Path(__file__).resolve().parent.parent)  # repo root
TRUCK_ID = 7
_GAPS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}
ID2NAME = dict(zip([i for i in range(1, 91) if i not in _GAPS], COCO_CLASS_NAMES))


def iou(a, b):  # a, b = [x, y, w, h]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u else 0


model = RFDETRBase()
for clip in sys.argv[1:]:
    ds = f"{ROOT}/labels/{clip}"
    path = f"{ds}/annotations.coco.json"
    shutil.copy(path, f"{ds}/annotations.coco.before_trucks.json")
    coco = json.load(open(path))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    ids = {v: k for k, v in names.items()}
    # re-run safety: undo an earlier run first
    coco["annotations"] = [a for a in coco["annotations"] if not a.get("added_truck")]
    for a in coco["annotations"]:
        if "was" in a:
            a["category_id"] = ids[a.pop("was")]
    if TRUCK_ID not in names:
        coco["categories"].append({"id": TRUCK_ID, "name": "truck"})
    nid = max(a["id"] for a in coco["annotations"]) + 1
    added = switched = 0
    for im in coco["images"]:
        d = model.predict(cv2.imread(f"{ds}/images/{im['file_name']}"), threshold=0.5)
        existing = [a for a in coco["annotations"] if a["image_id"] == im["id"] and names[a["category_id"]] != "person" and a["category_id"] != TRUCK_ID]
        for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence):
            if ID2NAME.get(int(c)) not in ("truck", "bus"):
                continue
            box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            hit = max(existing, key=lambda a: iou(box, a["bbox"]), default=None)
            if hit is not None and iou(box, hit["bbox"]) > 0.5:
                hit["was"] = names[hit["category_id"]]; hit["category_id"] = TRUCK_ID; hit["score"] = round(float(s), 3)
                existing.remove(hit); switched += 1
            else:
                coco["annotations"].append({"id": nid, "image_id": im["id"], "category_id": TRUCK_ID, "bbox": box, "area": box[2] * box[3],
                                            "iscrowd": 0, "score": round(float(s), 3), "added_truck": True})
                nid += 1; added += 1
    json.dump(coco, open(path, "w"))
    print(clip, "truck boxes added:", added, "| existing boxes switched to truck:", switched)
