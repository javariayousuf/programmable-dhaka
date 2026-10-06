import json, shutil, sys
from pathlib import Path

# Writes the boxes you added (and removed) in add_boxes.py back into the label file. Makes a backup first.
# usage: python apply_added.py <dataset_dir>
D = Path(sys.argv[1])
src = D / "annotations.coco.json"
saved = json.load(open(D / "review_added" / "added.json"))
shutil.copy(src, D / "annotations.coco.before_added_boxes.json")
coco = json.load(open(src))
ids = {c["name"]: c["id"] for c in coco["categories"]}
gone = set(saved["removed"])
coco["annotations"] = [a for a in coco["annotations"] if a["id"] not in gone]
nid = max([a["id"] for a in coco["annotations"]] + [0]) + 1
for a in saved["added"]:
    x, y, w, h = a["bbox"]
    coco["annotations"].append({"id": nid, "image_id": a["image_id"], "category_id": ids[a["category"]], "bbox": [float(x), float(y), float(w), float(h)],
                                "area": float(w * h), "iscrowd": 0, "score": 1.0, "added_by_hand": True})
    nid += 1
json.dump(coco, open(src, "w"))
print(f"added {len(saved['added'])} boxes, removed {len(gone)}. Backup: annotations.coco.before_added_boxes.json")
