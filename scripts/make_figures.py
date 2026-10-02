import argparse, json, os
from pathlib import Path
import numpy as np, cv2
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

# Builds the README pictures for a held-out test clip (default clip1, the Dhaka street): a side-by-side before/after GIF, three before/after stills,
# and a gallery of real mistakes (old pipeline and fine-tuned model). Vehicles only, same style both sides.
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder with the test clip's still frames (frame_0000.jpg ...)")
ap.add_argument("--clip", default="clip1", help="which labels/<clip> folder is the held-out test clip")
ap.add_argument("--tag", default="dhaka", help="suffix for the output files")
ap.add_argument("--weights", default=str(ROOT / "runs/small/checkpoint_best_total.pth"))
ap.add_argument("--font", default=os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Futura.ttc"))
args = ap.parse_args()

NAMES = {2: "bicycle", 3: "motorcycle", 4: "rickshaw", 5: "cart"}
HEX = {1: "#3777ff", 2: "#EC9F05", 3: "#3B0086", 4: "#db3069", 5: "#e0ff4f", 6: "#8a8f98"}
rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
FONT = lambda n: ImageFont.truetype(args.font, n)
OUT = ROOT / "media"

gt = json.load(open(ROOT / f"labels/{args.clip}/annotations.coco.json"))
orig = json.load(open(ROOT / f"labels/{args.clip}/original_pipeline.coco.json"))
files = {i["id"]: i["file_name"] for i in gt["images"]}


def by_image(coco):
    d = {}
    for a in coco["annotations"]:
        if a["category_id"] in NAMES:
            x, y, w, h = a["bbox"]
            d.setdefault(a["image_id"], []).append((a["category_id"], [x, y, x + w, y + h], a.get("score", 1.0)))
    return d


G, O = by_image(gt), by_image(orig)

from rfdetr import RFDETRSmall
model = RFDETRSmall(pretrain_weights=args.weights)
BLUR = sv.BlurAnnotator(kernel_size=81)


def head_boxes(xyxy, w_img, h_img):
    """Top slice of each person box, a bit narrower than the body, clipped to the frame."""
    if len(xyxy) == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = xyxy.T
    w, h = x2 - x1, y2 - y1
    hh = np.minimum(h * 0.25, w * 1.0)
    b = np.stack([x1 + 0.05 * w, y1, x2 - 0.05 * w, y1 + hh], axis=1)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w_img - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h_img - 1)
    return b[(b[:, 2] - b[:, 0] > 4) & (b[:, 3] - b[:, 1] > 4)]


_blur_cache = {}


def read_blurred(path):
    """Every still is read through here, so no figure ever shows an unblurred head."""
    if path not in _blur_cache:
        img = cv2.imread(path)
        d = model.predict(img, threshold=0.25)               # low cutoff so fewer people are missed
        hb = head_boxes(d[d.class_id == 0].xyxy, img.shape[1], img.shape[0])
        _blur_cache[path] = BLUR.annotate(img, sv.Detections(xyxy=hb)) if len(hb) else img
    return _blur_cache[path].copy()


P = {}
for iid, fn in files.items():
    d = model.predict(cv2.imread(f"{args.images}/{fn}"), threshold=0.5)
    P[iid] = [(int(c) + 1, [float(v) for v in b], float(s)) for b, c, s in zip(d.xyxy, d.class_id, d.confidence) if int(c) + 1 in NAMES]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0


def draw(img_bgr, items, size=22):
    """items = [(category, xyxy, text)]. Rounded-ish boxes in the palette, label chip in Futura."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil); f = FONT(size)
    for cat, (x1, y1, x2, y2), text in items:
        col = rgb(HEX[cat])
        dr.rounded_rectangle([x1, y1, x2, y2], radius=14, outline=col, width=4)
        tb = dr.textbbox((0, 0), text, font=f); tw, th = tb[2] - tb[0] + 12, tb[3] - tb[1] + 10
        ty = max(0, y1 - th)
        dr.rectangle([x1, ty, x1 + tw, ty + th], fill=col)
        dr.text((x1 + 6, ty + 3), text, font=f, fill=(255, 255, 255) if cat == 3 else (17, 17, 17))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def banner(img_bgr, text):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil); f = FONT(max(16, img_bgr.shape[1] // 22))
    tb = dr.textbbox((0, 0), text, font=f)
    dr.rectangle([0, 0, tb[2] + 24, tb[3] + 20], fill=(255, 255, 255)); dr.text((12, 8), text, font=f, fill=(17, 17, 17))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render(iid, source, w):
    img = read_blurred(f"{args.images}/{files[iid]}")
    items = [(c, b, NAMES[c]) for c, b, s in source.get(iid, [])]
    big = draw(img, items, size=34)
    return cv2.resize(big, (w, int(big.shape[0] * w / big.shape[1])), interpolation=cv2.INTER_AREA)


def hits(pred, iid, cls=4):
    r = [b for c, b, _ in G.get(iid, []) if c == cls]
    return sum(any(c == cls and iou(b, pb) > 0.5 for c, pb, _ in pred.get(iid, [])) for b in r)


# 1) before / after GIF over the stretch where rickshaws are on screen
frames = sorted(files)[::3]
pairs = []
for iid in frames:
    a, b = banner(render(iid, O, 480), "Original pipeline"), banner(render(iid, P, 480), "After fine-tuning")
    pairs.append(np.hstack([a, b]))
import subprocess, tempfile, imageio_ffmpeg
tmp = tempfile.mkdtemp()
for k, p in enumerate(pairs):
    cv2.imwrite(f"{tmp}/f{k:03d}.png", p)
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "6", "-i", f"{tmp}/f%03d.png",
                "-vf", "split[a][b];[a]palettegen=max_colors=112:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop", "0", str(OUT / f"before_after_{args.tag}.gif")], check=True)

# the same comparison where the fine-tuned model does BETTER. Written only if there are at least 3 frames where it
# truly finds more rickshaws, so the strip can never be filled with frames that are not real wins.
gain = sorted((i for i in files if hits(P, i) > hits(O, i)), key=lambda i: hits(P, i) - hits(O, i), reverse=True)
pick = []
for i in gain:
    if all(abs(i - j) >= 10 for j in pick): pick.append(i)
    if len(pick) == 3: break
if len(pick) == 3:
    rows = [np.hstack([banner(render(i, O, 800), "Original pipeline"), banner(render(i, P, 800), "After fine-tuning")]) for i in sorted(pick)]
    cv2.imwrite(str(OUT / f"where_it_wins_{args.tag}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])
else:
    print("not enough real wins for a strip:", len(gain), "frames")
# 2) the same comparison where the fine-tuned model does WORSE (carts it no longer finds). Not hidden on purpose.
loss = sorted(files, key=lambda i: hits(O, i, 5) - hits(P, i, 5), reverse=True)
pick2 = []
for i in loss:
    if all(abs(i - j) >= 10 for j in pick2): pick2.append(i)
    if len(pick2) == 3: break
rows = [np.hstack([banner(render(i, O, 800), "Original pipeline"), banner(render(i, P, 800), "After fine-tuning")]) for i in sorted(pick2)]
cv2.imwrite(str(OUT / f"where_it_is_worse_{args.tag}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])

# 3) gallery of real mistakes
tiles = []


def tile(iid, cat, box, label, caption):
    img = read_blurred(f"{args.images}/{files[iid]}")
    img = draw(img, [(cat, box, label)], size=34)
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(x2 - x1, (y2 - y1) * 16 / 9) * 1.5; h = w * 9 / 16
    x0 = int(min(max(cx - w / 2, 0), max(0, img.shape[1] - w))); y0 = int(min(max(cy - h / 2, 0), max(0, img.shape[0] - h)))
    crop = cv2.resize(img[y0:y0 + int(h), x0:x0 + int(w)], (640, 360), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); canvas = Image.new("RGB", (640, 420), (255, 255, 255))
    canvas.paste(pil, (0, 0)); ImageDraw.Draw(canvas).text((12, 372), caption, font=FONT(24), fill=(17, 17, 17))
    tiles.append(np.array(canvas))


def best(cands, used):
    for c in sorted(cands, key=lambda t: -t[0]):
        if all(abs(c[1] - u) >= 8 for u in used): used.append(c[1]); return c
    return None


used = []
for cat_name, cid, caption in (("cart", 5, "Old pipeline: a rickshaw called a cart"), ("bicycle", 2, "Old pipeline: a rickshaw called a bicycle")):
    c = []
    for iid, gl in G.items():
        for gc, gb, _ in gl:
            if gc != 4: continue
            m = [pb for pc, pb, _ in O.get(iid, []) if pc == cid and iou(gb, pb) > 0.5]
            if m and not any(pc == 4 and iou(gb, pb) > 0.5 for pc, pb, _ in O.get(iid, [])):
                c.append(((gb[2] - gb[0]) * (gb[3] - gb[1]), iid, m[0]))
    t = best(c, used)
    if t: tile(t[1], cid, t[2], cat_name, caption)

# model boxes drawn where the answer key has nothing: kept only if I confirmed by eye that they are wrong (see AUDITED_WRONG)
AUDITED_WRONG = set()   # (frame id, class) pairs, filled in after looking at every unmatched box
fp = [((b[2] - b[0]) * (b[3] - b[1]), iid, b) for iid, pl in P.items() for c, b, s in pl
      if (iid, c) in AUDITED_WRONG and not any(iou(b, gb) > 0.3 for _, gb, _ in G.get(iid, []))]
t = best(fp, used)
if t: tile(t[1], 4, t[2], "rickshaw", "Fine-tuned: a wrong guess (audited by eye)")
miss = [((b[2] - b[0]) * (b[3] - b[1]), iid, b) for iid, gl in G.items() for c, b, _ in gl
        if c == 4 and not any(iou(b, pb) > 0.1 for pc, pb, _ in P.get(iid, []))]  # nothing at all drawn on it
t = best(miss, used)
if t: tile(t[1], 4, t[2], "missed", "Fine-tuned: a rickshaw it did not see")
mo = [((b[2] - b[0]) * (b[3] - b[1]), iid, pb) for iid, gl in G.items() for c, b, _ in gl if c == 4
      for pc, pb, _ in P.get(iid, []) if pc == 3 and iou(b, pb) > 0.5]
t = best(mo, used)
if t: tile(t[1], 3, t[2], "motorcycle", "Fine-tuned: a rickshaw called a motorcycle")
while len(tiles) % 3: tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite(str(OUT / f"mistakes_gallery_{args.tag}.jpg"), cv2.cvtColor(np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 86])
print("pick", sorted(pick), "worse", sorted(pick2), "tiles", len(tiles))
