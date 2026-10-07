import json, sys
from pathlib import Path
import numpy as np, cv2, torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Runs the original pipeline (pretrained RF-DETR for people, bicycles and motorcycles, plus Grounding DINO text prompts for
# rickshaws and carts) on the stills already in labels/<clip>/images, and writes labels/<clip>/original_pipeline.coco.json,
# the "before" that evaluate.py compares a fine-tuned model against. Same logic as prelabel.py, which starts from a video.
# usage: python original_pipeline_on_stills.py <clip>
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "labels" / sys.argv[1]
coco = json.load(open(D / "annotations.coco.json"))

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
    r = gd_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
    n = len(r["scores"])
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=r["boxes"].cpu().numpy().astype(float), confidence=r["scores"].cpu().numpy(), class_id=np.full(n, class_idx))


annos = []
for im in coco["images"]:
    frame = cv2.imread(str(D / "images" / im["file_name"]))
    d = det.predict(frame, threshold=0.5)
    keep = np.array([ID2NAME.get(int(c)) in ("person", "bicycle", "motorcycle") for c in d.class_id], dtype=bool)
    d = d[keep]
    d.class_id = np.array([CLASSES.index(ID2NAME[int(c)]) for c in d.class_id], dtype=int)
    d.data, d.metadata = {}, {}
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cand = [x for x in (gd_boxes(img, "a decorated three-wheeled cycle rickshaw.", RICKSHAW), gd_boxes(img, "a push cart.", CART)) if len(x)]
    if cand:
        c = sv.Detections.merge(cand).with_nms(threshold=0.5, class_agnostic=True)
        two = d[np.isin(d.class_id, [BICYCLE, MOTO])]
        if len(two):  # the bicycle and motorcycle detector wins over the text prompts
            c = c[sv.box_iou_batch(c.xyxy, two.xyxy).max(axis=1) < 0.3]
        if len(c):
            d = sv.Detections.merge([d, c]) if len(d) else c
    for xyxy, cid, conf in zip(d.xyxy, d.class_id, d.confidence):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        annos.append({"id": len(annos) + 1, "image_id": im["id"], "category_id": int(cid) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                      "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "score": round(float(conf), 3)})
json.dump({"images": coco["images"], "annotations": annos, "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]},
          open(D / "original_pipeline.coco.json", "w"))
print(sys.argv[1], "stills:", len(coco["images"]), "boxes:", len(annos))
