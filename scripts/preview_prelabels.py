import json, os
import numpy as np, cv2
import supervision as sv

import sys

# usage: python preview_prelabels.py <dataset_dir> <preview_dir>
ROOT = sys.argv[1]  # dataset folder holding annotations.coco.json and images/
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(f"{ROOT}/annotations.coco.json"))
names = [c["name"] for c in coco["categories"]]  # category id = index + 1
palette = sv.ColorPalette.from_hex(["#3777ff", "#EC9F05", "#3B0086", "#db3069", "#e0ff4f"])  # person, bicycle, motorcycle, rickshaw, cart
LOOKUP = sv.ColorLookup.CLASS
boxes = sv.RoundBoxAnnotator(color=palette, color_lookup=LOOKUP, thickness=3, roundness=0.25)
label = sv.LabelAnnotator(color=palette, color_lookup=LOOKUP, text_color=sv.Color.from_hex("#111111"),
                          text_scale=0.8, text_thickness=2)

by_img = {}
for a in coco["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

for im in coco["images"]:
    frame = cv2.imread(f"{ROOT}/images/{im['file_name']}")
    anns = by_img.get(im["id"], [])
    if anns:
        xyxy = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns])
        d = sv.Detections(xyxy=xyxy, class_id=np.array([a["category_id"] - 1 for a in anns]),
                          confidence=np.array([a["score"] for a in anns]))
        labels = [f"{names[a['category_id'] - 1]} (fixed)" if a.get('corrected') else f"{names[a['category_id'] - 1]} {a['score']:.2f}" for a in anns]
        frame = label.annotate(boxes.annotate(frame, d), d, labels)
    s = 1280 / max(frame.shape[:2])  # long side 1280, keeps portrait videos upright
    cv2.imwrite(f"{OUT}/{im['file_name']}", cv2.resize(frame, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("saved", len(coco["images"]), "previews")
