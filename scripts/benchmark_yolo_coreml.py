import glob, json, os, platform, subprocess, sys, tempfile, time, warnings
import numpy as np, torch, cv2
from PIL import Image
warnings.filterwarnings("ignore")
import coremltools as ct
from ultralytics import YOLO

# The YOLO twin of benchmark_coreml.py: single-image speed of a fine-tuned Ultralytics YOLO model on a Mac, PyTorch against CoreML
# (each set of compute units), and whether CoreML gives the same detections as PyTorch. Model forward only at 640 x 640, the size
# YOLO is trained at. Close other apps and plug the Mac in first.
# usage: python benchmark_yolo_coreml.py <weights.pt> <folder of .jpg stills> [out.json]
# needs macOS and: pip install ultralytics coremltools opencv-python
WEIGHTS, IMG_DIR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "yolo_speed_results.json"
IMGS = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
R = 640
PKG = {}
if os.environ.get("COREML_DIR"):        # packages made earlier by export_yolo_coreml.py (needed when the export fails on a new PyTorch)
    PKG = {t: f"{os.environ['COREML_DIR']}/{t}.mlpackage" for t in ("fp32", "fp16")}
else:
    work = tempfile.mkdtemp()
    for half, tag in ((False, "fp32"), (True, "fp16")):
        p = YOLO(WEIGHTS).export(format="coreml", imgsz=R, half=half, nms=False, device="cpu")
        dest = f"{work}/{tag}.mlpackage"; os.rename(p, dest); PKG[tag] = dest
CU = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
      "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}
module = YOLO(WEIGHTS).model.float().eval()


def timeit(fn, warm=10, passes=3, iters=100):
    for _ in range(warm):
        fn()
    p50s, allt = [], []
    for _ in range(passes):
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        p50s.append(float(np.median(ts))); allt += ts; time.sleep(0.2)
    return {"p50_ms": round(float(np.median(p50s)), 1), "p50_per_pass": [round(v, 1) for v in p50s],
            "p95_ms": round(float(np.percentile(allt, 95)), 1)}


def torch_fn(device, x):
    m = module.to(device); xt = x.to(device)
    def fn():
        with torch.no_grad():
            m(xt)
        if device == "mps":
            torch.mps.synchronize()
    return fn


sh = lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
info = {"date": time.strftime("%Y-%m-%d %H:%M"), "chip": sh("sysctl -n machdep.cpu.brand_string"),
        "power": sh("pmset -g batt | head -2 | tail -1 | cut -f2 | cut -d';' -f1-2"), "load_start": round(os.getloadavg()[0], 1),
        "torch": torch.__version__, "coremltools": ct.__version__, "python": platform.python_version(), "weights": WEIGHTS, "resolution": R,
        "iterations": "3 passes x 100, 10 warm-up, p50 = median of the three pass medians"}
rgb = cv2.cvtColor(cv2.resize(cv2.imread(IMGS[10]), (R, R)), cv2.COLOR_BGR2RGB)
x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255
results = {}
for dev in ("cpu", "mps"):
    results[f"pytorch_{dev}"] = timeit(torch_fn(dev, x)); print(dev, results[f"pytorch_{dev}"], flush=True)
module.to("cpu")
pil = Image.fromarray(rgb)
for tag, path in PKG.items():
    for name, cu in CU.items():
        t0 = time.perf_counter(); ml = ct.models.MLModel(path, compute_units=cu); load_s = time.perf_counter() - t0
        in_name = ml.get_spec().description.input[0].name
        r = timeit(lambda: ml.predict({in_name: pil})); r["load_s"] = round(load_s, 1)
        results[f"coreml_{tag}_{name}"] = r; print(tag, name, r, flush=True)

# parity: Ultralytics' own predict on the PyTorch weights against the CoreML packages, same stills, cutoff 0.5
def boxes(model, frame):
    r = model.predict(frame, conf=0.5, imgsz=R, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


ref = YOLO(WEIGHTS)
cml = {t: YOLO(p, task="detect") for t, p in PKG.items()}
parity = {t: {"reference_detections": 0, "matched": 0, "extra": 0} for t in PKG}
for path in IMGS[::3][:30]:
    frame = cv2.imread(path)
    rb, rc = boxes(ref, frame)
    for t, m in cml.items():
        cb, cc = boxes(m, frame)
        used, matched = set(), 0
        for i in range(len(rb)):
            if len(cb) == 0:
                break
            ious = iou(rb[i], cb); ious[list(used)] = 0
            j = int(np.argmax(ious))
            if ious[j] > 0.5 and cc[j] == rc[i]:
                matched += 1; used.add(j)
        parity[t]["reference_detections"] += len(rb); parity[t]["matched"] += matched; parity[t]["extra"] += len(cb) - len(used)
for t, d in parity.items():
    d["share_matched"] = round(d["matched"] / max(1, d["reference_detections"]), 3)
info["load_end"] = round(os.getloadavg()[0], 1)
json.dump({"info": info, "latency": results, "parity_vs_pytorch": parity}, open(OUT, "w"), indent=2)
print("parity", json.dumps(parity)); print("done", OUT)
