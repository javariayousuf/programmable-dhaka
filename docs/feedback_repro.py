import sys, warnings
from pathlib import Path

# Reproduces product feedback items 1 to 3 from the README in a few seconds, so anyone can check them.
# usage: python docs/feedback_repro.py [any photo with people or vehicles in it]
# needs: pip install rfdetr supervision opencv-python   (tested on rfdetr 1.11.2 and supervision 0.30.8, October 2026)
import cv2

ROOT = Path(__file__).resolve().parent.parent
image_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "media/where_it_wins_rickshaw_street.jpg")

print("== item 1: model.class_names[class_id] against detections.data['class_name'] ==")
from rfdetr import RFDETRBase
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = RFDETRBase()                      # also triggers the RFDETRBase deprecation warning (item 2)
d = model.predict(cv2.imread(image_path), threshold=0.4)
ids = [int(i) for i in d.class_id][:8]
by_index = [model.class_names[i] for i in ids]
by_data = list(d.data["class_name"])[:8]
print("class_id                    :", ids)
print("model.class_names[class_id] :", by_index)
print("detections.data['class_name']:", by_data)
print("MISMATCH" if by_index != by_data else "same names (nothing to see on this photo, try one with people)")

print("\n== item 2: deprecation warnings that name no replacement ==")
for w in caught:
    if "deprecat" in str(w.message).lower():
        print("RFDETRBase:", w.message)
import supervision as sv
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sv.ByteTrack()
for w in caught:
    if "deprecat" in str(w.message).lower():
        print("sv.ByteTrack:", w.message)

print("\n== item 3: the removed-import message ==")
try:
    import rfdetr.util
except ImportError as e:
    print("ImportError:", e)
import rfdetr.utilities as utilities
print("COCO_CLASS_NAMES in rfdetr.utilities:", hasattr(utilities, "COCO_CLASS_NAMES"))
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
print("COCO_CLASS_NAMES in rfdetr.assets.coco_classes:", len(COCO_CLASS_NAMES), "names")
