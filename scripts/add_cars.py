from pathlib import Path
import json, shutil
import numpy as np, cv2
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

# Adds a 6th class, "car", to both label files. Without it the model has to call every four-wheeler
# a rickshaw. Cars come from the pretrained detector (car, truck, bus). A box is skipped when it
# overlaps a hand-checked rickshaw, cart, motorcycle or bicycle box, so the pretrained model's
# mistakes (it sometimes calls e-rickshaws cars) never contradict the corrected labels.
ROOT = str(Path(__file__).resolve().parent.parent)  # repo root
DATASETS = [f"{ROOT}/labels/clip1", f"{ROOT}/labels/clip2"]
CAR_ID = 6
_GAPS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}
ID2NAME = dict(zip([i for i in range(1, 91) if i not in _GAPS], COCO_CLASS_NAMES))


def overlap(a, b):  # a, b = [x, y, w, h]; share of the SMALLER box that is covered
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


model = RFDETRBase()
for ds in DATASETS:
    path = f"{ds}/annotations.coco.json"
    shutil.copy(path, path.replace(".json", ".before_cars.json"))
    coco = json.load(open(path))
    coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] != CAR_ID]  # safe to re-run
    if not any(c["id"] == CAR_ID for c in coco["categories"]):
        coco["categories"].append({"id": CAR_ID, "name": "car"})
    nid = max(a["id"] for a in coco["annotations"]) + 1
    kept = skipped = 0
    for im in coco["images"]:
        frame = cv2.imread(f"{ds}/images/{im['file_name']}")
        d = model.predict(frame, threshold=0.5)
        vehicles = [a["bbox"] for a in coco["annotations"] if a["image_id"] == im["id"] and a["category_id"] in (2, 3, 4, 5)]
        for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence):
            if ID2NAME.get(int(c)) not in ("car", "truck", "bus"):
                continue
            box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            if any(overlap(box, v) > 0.3 for v in vehicles):
                skipped += 1
                continue
            coco["annotations"].append({"id": nid, "image_id": im["id"], "category_id": CAR_ID, "bbox": box,
                                        "area": box[2] * box[3], "iscrowd": 0, "score": round(float(s), 3)})
            nid += 1; kept += 1
    json.dump(coco, open(path, "w"))
    print(ds.split("/")[-1], "car boxes added:", kept, "skipped (overlap a checked box):", skipped)
