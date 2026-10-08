import os, sys
from ultralytics import YOLO

# Exports an Ultralytics YOLO checkpoint to CoreML at 32 and 16 bit: <out_dir>/fp32.mlpackage and <out_dir>/fp16.mlpackage.
# On this Mac the export failed with PyTorch 2.14 and coremltools 9.0 (a cast the converter cannot handle) and worked with PyTorch 2.7.1,
# the newest the converter has been tested with, so run this in an environment with torch==2.7.1. Then time the packages with
# COREML_DIR=<out_dir> python benchmark_yolo_coreml.py ...
# usage: python export_yolo_coreml.py <weights.pt> <out_dir>
weights, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for half, tag in ((False, "fp32"), (True, "fp16")):
    p = YOLO(weights).export(format="coreml", imgsz=640, half=half, nms=False, device="cpu")
    os.rename(p, f"{out}/{tag}.mlpackage"); print("exported", tag, f"{out}/{tag}.mlpackage", flush=True)
