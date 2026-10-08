import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, cv2
import supervision as sv
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSmall

# Renders the README hero GIF with the same drawing style as make_figures.py (thick rounded boxes, plain class names, the official palette),
# heads blurred, and a credit box. Samples the video at about 8 frames per second and writes a 640 px wide GIF.
# usage: python make_hero_gif.py <video> <start seconds> <length seconds> <out.gif> <weights.pth> [credit line 1] [credit line 2]
SRC, START, LENGTH, OUT, WEIGHTS = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
CREDIT = sys.argv[6:]
FONT_PATH = os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Futura.ttc")
NAMES = {1: "person", 2: "bicycle", 3: "motorcycle", 4: "rickshaw", 5: "cart", 6: "car"}
HEX = {1: "#2de1fc", 2: "#EC9F05", 3: "#3B0086", 4: "#db3069", 5: "#e0ff4f", 6: "#8a8f98"}
rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
TOOLS = "RF-DETR | supervision | Grounding DINO | PyTorch | OpenCV"
FPS_OUT = 8

model = RFDETRSmall(pretrain_weights=WEIGHTS)
blur = sv.BlurAnnotator(kernel_size=81)


def head_boxes(xyxy, w_img, h_img):
    if len(xyxy) == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = xyxy.T
    w, h = x2 - x1, y2 - y1
    hh = np.minimum(h * 0.25, w * 1.0)
    b = np.stack([x1 + 0.05 * w, y1, x2 - 0.05 * w, y1 + hh], axis=1)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w_img - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h_img - 1)
    return b[(b[:, 2] - b[:, 0] > 4) & (b[:, 3] - b[:, 1] > 4)]


def draw(img_bgr, items, size=44, width=18):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil); f = ImageFont.truetype(FONT_PATH, size)
    for cat, (x1, y1, x2, y2), text in items:
        col = rgb(HEX[cat])
        dr.rounded_rectangle([x1, y1, x2, y2], radius=14, outline=col, width=width)
        tb = dr.textbbox((0, 0), text, font=f); tw, th = tb[2] - tb[0] + 12, tb[3] - tb[1] + 10
        ty = max(0, y1 - th)
        dr.rectangle([x1, ty, x1 + tw, ty + th], fill=col)
        dr.text((x1 + 6, ty + 3), text, font=f, fill=(17, 17, 17) if cat == 5 else (255, 255, 255))   # white label text, except dark on the light yellow-green cart color
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def credit(img, lines, size=int(os.environ.get("CREDIT_SIZE", "36"))):   # smaller for wide scenes where the box would hide the action
    f = ImageFont.truetype(FONT_PATH, size)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    boxes = [dr.textbbox((0, 0), t, font=f) for t in lines]
    lh = max(b[3] - b[1] for b in boxes) + 6
    bw, bh = max(b[2] for b in boxes) + 36, lh * len(lines) + 30
    x1, y1 = 30, img.shape[0] - 30 - bh
    roi = img[y1:y1 + bh, x1:x1 + bw]
    img[y1:y1 + bh, x1:x1 + bw] = cv2.addWeighted(roi, 0.15, np.full_like(roi, 255), 0.85, 0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    for i, t in enumerate(lines):
        dr.text((x1 + 18, y1 + 14 + i * lh), t, font=f, fill=(17, 17, 17))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS); step = fps / FPS_OUT
tmp = tempfile.mkdtemp(); k = 0
t = START * fps
while t < (START + LENGTH) * fps:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t)))
    ok, frame = cap.read()
    if not ok:
        break
    d = model.predict(frame, threshold=0.5)
    low = model.predict(frame, threshold=0.25)                 # low cutoff so the blur misses fewer people
    hb = head_boxes(low[low.class_id == 0].xyxy, frame.shape[1], frame.shape[0])
    img = blur.annotate(frame.copy(), sv.Detections(xyxy=hb)) if len(hb) else frame.copy()
    items = [(int(c) + 1, [float(v) for v in b], NAMES[int(c) + 1]) for b, c in zip(d.xyxy, d.class_id) if int(c) + 1 in NAMES and int(c) != 0]   # people are blurred at the head but not boxed, as in the before/after GIFs
    img = draw(img, items)
    if CREDIT:
        img = credit(img, ["Created by J. Yousuf", TOOLS] + CREDIT)
    cv2.imwrite(f"{tmp}/f{k:03d}.png", cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)); k += 1; t += step
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS_OUT), "-i", f"{tmp}/f%03d.png",
                "-vf", "scale=640:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                "-loop", "0", OUT], check=True)
print("frames", k, "->", OUT)
