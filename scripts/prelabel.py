import json, os
import numpy as np, cv2, torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import sys

# usage: python prelabel.py <video.mp4> <out_dir> [stride]
SRC = sys.argv[1]
OUT = sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 12  # 12 frames = one still every 0.2 s at 60fps
os.makedirs(f"{OUT}/images", exist_ok=True)

# COCO ids have gaps, the name list is 0-indexed
_GAPS = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}
ID2NAME = dict(zip([i for i in range(1, 91) if i not in _GAPS], COCO_CLASS_NAMES))
CLASSES = ["person", "bicycle", "motorcycle", "rickshaw", "cart"]  # category id = index + 1
PERSON, BICYCLE, MOTO, RICKSHAW, CART = range(5)

det = RFDETRBase()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
gd_id = "IDEA-Research/grounding-dino-tiny"
gd_proc = AutoProcessor.from_pretrained(gd_id)
gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(dev).eval()


def gd_boxes(img, prompt, class_idx, thr=0.4):
    inp = gd_proc(images=img, text=prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[img.size[::-1]]
    )[0]
    n = len(r["scores"])
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=r["boxes"].cpu().numpy().astype(float),
                         confidence=r["scores"].cpu().numpy(), class_id=np.full(n, class_idx))


images, annos = [], []
counts = {c: 0 for c in CLASSES}
for n, frame in enumerate(sv.get_video_frames_generator(SRC, stride=STRIDE)):
    h, w = frame.shape[:2]
    fname = f"frame_{n:04d}.jpg"
    cv2.imwrite(f"{OUT}/images/{fname}", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    images.append({"id": n, "file_name": fname, "width": w, "height": h})

    # 1) people, bicycles, motorcycles from RF-DETR
    d = det.predict(frame, threshold=0.5)
    names = [ID2NAME.get(int(c)) for c in d.class_id]
    keep = np.array([nm in ("person", "bicycle", "motorcycle") for nm in names], dtype=bool)
    d = d[keep]
    d.class_id = np.array([CLASSES.index(ID2NAME[int(c)]) for c in d.class_id], dtype=int)
    d.data, d.metadata = {}, {}

    # 2) rickshaw and cart candidates from text prompts, one prompt each
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cand = [x for x in (gd_boxes(img, "a decorated three-wheeled cycle rickshaw.", RICKSHAW),
                        gd_boxes(img, "a push cart.", CART)) if len(x)]
    if cand:
        c = sv.Detections.merge(cand).with_nms(threshold=0.5, class_agnostic=True)
        two = d[np.isin(d.class_id, [BICYCLE, MOTO])]
        if len(two):  # bicycle / motorcycle detector wins over the text prompts
            c = c[sv.box_iou_batch(c.xyxy, two.xyxy).max(axis=1) < 0.3]
        if len(c):
            d = sv.Detections.merge([d, c]) if len(d) else c

    for xyxy, cid, conf in zip(d.xyxy, d.class_id, d.confidence):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        annos.append({"id": len(annos) + 1, "image_id": n, "category_id": int(cid) + 1,
                      "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                      "iscrowd": 0, "score": round(float(conf), 3)})
        counts[CLASSES[int(cid)]] += 1
    if n % 10 == 0:
        print("frame", n, flush=True)

coco = {"images": images, "annotations": annos,
        "categories": [{"id": i + 1, "name": nm} for i, nm in enumerate(CLASSES)]}
json.dump(coco, open(f"{OUT}/annotations.coco.json", "w"))
print("images:", len(images), "boxes:", counts)
