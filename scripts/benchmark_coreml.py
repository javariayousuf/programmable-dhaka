import glob, json, os, platform, subprocess, sys, tempfile, time, warnings
import numpy as np, torch, cv2
warnings.filterwarnings("ignore")
import coremltools as ct
import torchvision.transforms.functional as F
from rfdetr import RFDETRSmall
from rfdetr.models.postprocess import PostProcess

# Single-image speed of a fine-tuned RF-DETR Small on a Mac, and whether CoreML gives the same detections as PyTorch.
# It exports the checkpoint to CoreML at 32 and 16 bit, times PyTorch (CPU and the graphics chip) and CoreML (each set of
# compute units), then compares detections. Model forward only: every runtime gets the same preprocessed square tensor, so
# the numbers compare the model, not the steps before and after it. Close other apps and plug the Mac in first.
# usage: python benchmark_coreml.py <checkpoint.pth> <folder of .jpg stills> [out.json]
# needs macOS and: pip install "rfdetr[coreml]" opencv-python
CKPT, IMG_DIR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "speed_results.json"
IMGS = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
work = tempfile.mkdtemp()
PKG = {}
for precision, tag in (("float32", "fp32"), ("float16", "fp16")):
    PKG[tag] = str(RFDETRSmall(pretrain_weights=CKPT).export(format="coreml", output_dir=f"{work}/{precision}", coreml_precision=precision))
CU = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
      "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}

model = RFDETRSmall(pretrain_weights=CKPT)
module = model.model.model.eval()
R = model.model_config.resolution


def prep(path):
    img = cv2.imread(path)[:, :, ::-1].copy()
    t = F.resize(F.to_tensor(img), [R, R], antialias=False)
    return F.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)


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
        "torch": torch.__version__, "coremltools": ct.__version__, "python": platform.python_version(), "checkpoint": os.path.basename(CKPT),
        "resolution": R, "iterations": "3 passes x 100, 10 warm-up, p50 = median of the three pass medians"}
x = prep(IMGS[10])
results = {}
for dev in ("cpu", "mps"):
    results[f"pytorch_{dev}"] = timeit(torch_fn(dev, x)); print(dev, results[f"pytorch_{dev}"], flush=True)
module.to("cpu")
for tag, path in PKG.items():
    for name, cu in CU.items():
        t0 = time.perf_counter(); ml = ct.models.MLModel(path, compute_units=cu); load_s = time.perf_counter() - t0
        in_name = ml.get_spec().description.input[0].name
        arr = x.numpy().astype(np.float32)
        r = timeit(lambda: ml.predict({in_name: arr})); r["load_s"] = round(load_s, 1)
        results[f"coreml_{tag}_{name}"] = r; print(tag, name, r, flush=True)

# parity: same tensor, same RF-DETR post-processing, PyTorch on the CPU in float32 as the reference
pp = PostProcess(num_select=300)


def decode(boxes, logits):
    out = pp({"pred_logits": torch.as_tensor(logits), "pred_boxes": torch.as_tensor(boxes)}, torch.tensor([[R, R]]))[0]
    keep = out["scores"] > 0.5
    return out["boxes"][keep].numpy(), out["labels"][keep].numpy(), out["scores"][keep].numpy()


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


mls = {t: ct.models.MLModel(p, compute_units=ct.ComputeUnit.ALL) for t, p in PKG.items()}
names = {t: m.get_spec().description.input[0].name for t, m in mls.items()}
outs = {t: [o.name for o in m.get_spec().description.output] for t, m in mls.items()}   # outputs are positional: boxes, then logits
parity = {t: {"reference_detections": 0, "matched": 0, "extra": 0, "score_diff": []} for t in PKG}
for path in IMGS[::3][:30]:
    xi = prep(path)
    with torch.no_grad():
        o = module(xi)
    rb, rl, rs = decode(o["pred_boxes"].numpy(), o["pred_logits"].numpy())
    for t, m in mls.items():
        pr = m.predict({names[t]: xi.numpy().astype(np.float32)})
        cb, cl, cs = decode(pr[outs[t][0]], pr[outs[t][1]])
        used = set(); matched = 0
        for i in range(len(rb)):
            if len(cb) == 0:
                break
            ious = iou(rb[i], cb); ious[list(used)] = 0
            j = int(np.argmax(ious))
            if ious[j] > 0.5 and cl[j] == rl[i]:
                matched += 1; used.add(j); parity[t]["score_diff"].append(abs(float(cs[j]) - float(rs[i])))
        parity[t]["reference_detections"] += len(rb); parity[t]["matched"] += matched; parity[t]["extra"] += len(cb) - len(used)
for t, d in parity.items():
    d["share_matched"] = round(d["matched"] / max(1, d["reference_detections"]), 3)
    d["mean_abs_score_diff"] = round(float(np.mean(d["score_diff"])) if d["score_diff"] else 0, 4); d.pop("score_diff")
info["load_end"] = round(os.getloadavg()[0], 1)
json.dump({"info": info, "latency": results, "parity_vs_pytorch_cpu": parity}, open(OUT, "w"), indent=2)
print("parity", json.dumps(parity)); print("done", OUT)
