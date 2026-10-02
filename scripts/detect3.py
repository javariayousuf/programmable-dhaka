import os, sys
from pathlib import Path
import numpy as np, cv2
import supervision as sv
import imageio
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSmall

# usage: python detect3.py <video> <out.mp4> [credit line 1] [credit line 2]
SRC, DST = sys.argv[1], sys.argv[2]
CREDIT = sys.argv[3:]
WEIGHTS = os.environ.get("WEIGHTS", str(Path(__file__).resolve().parent.parent / "runs/small/checkpoint_best_total.pth"))  # produced by train.py
FONT_PATH = os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Futura.ttc")  # Futura Medium on macOS; set FONT_PATH to any .ttf on other systems
FONT_SIZE = 18  # every piece of text in the video (labels and credit box) uses this size
SLOW = 0.75  # playback speed, 1.0 = real time

CLASSES = ["person", "bicycle", "motorcycle", "rickshaw", "cart", "car"]  # same order the model was trained with
COLORS = ["#3777ff", "#EC9F05", "#3B0086", "#db3069", "#e0ff4f", "#8a8f98"]  # person blue, bicycle amber, motorcycle deep purple, rickshaw raspberry, cart neon yellow-green, car gray
palette = sv.ColorPalette.from_hex(COLORS)
LOOKUP = sv.ColorLookup.CLASS

info = sv.VideoInfo.from_video_path(SRC)
STRIDE = max(1, round(info.fps / 30))  # process about 30 frames per second of video
rate = info.fps / STRIDE
model = RFDETRSmall(pretrain_weights=WEIGHTS)

tracker = sv.ByteTrack(frame_rate=int(rate), lost_track_buffer=45, minimum_consecutive_frames=3)
smoother = sv.DetectionsSmoother(length=7)
boxes = sv.RoundBoxAnnotator(color=palette, color_lookup=LOOKUP, thickness=2, roundness=0.25)
label = sv.RichLabelAnnotator(color=palette, color_lookup=LOOKUP, text_color=sv.Color.from_hex("#111111"),
                              font_path=FONT_PATH, font_size=FONT_SIZE, text_padding=6)
label_light = sv.RichLabelAnnotator(color=palette, color_lookup=LOOKUP, text_color=sv.Color.WHITE,
                                    font_path=FONT_PATH, font_size=FONT_SIZE, text_padding=6)  # white text on the deep purple motorcycle label
blur = sv.BlurAnnotator(kernel_size=61)


def head_boxes(xyxy, w_img, h_img):
    """Top slice of each person box, a bit narrower than the body, clipped to the frame."""
    if len(xyxy) == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = xyxy.T
    w, h = x2 - x1, y2 - y1
    hh = np.minimum(h * 0.22, w * 0.9)
    b = np.stack([x1 + 0.1 * w, y1, x2 - 0.1 * w, y1 + hh], axis=1)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w_img - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h_img - 1)
    return b[(b[:, 2] - b[:, 0] > 4) & (b[:, 3] - b[:, 1] > 4)]


TOOLS = "RF-DETR | supervision | Grounding DINO | PyTorch | OpenCV"  # the open source libraries behind this video


def draw_credit(img, lines):
    """White box (about 85% opaque), bottom left. Every line uses the same Calibri size as the labels."""
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    pad, margin, gap = 9, 20, 3
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    sizes = [probe.textbbox((0, 0), t, font=font) for t in lines]
    line_h = max(b[3] - b[1] for b in sizes) + 2
    box_w = max(b[2] - b[0] for b in sizes) + 2 * pad
    box_h = line_h * len(lines) + gap * (len(lines) - 1) + 2 * pad
    x1 = margin                                   # bottom left, clear of the labels that sit above each box
    y1 = img.shape[0] - margin - box_h
    x2 = x1 + box_w
    roi = img[y1:y1 + box_h, x1:x2]
    img[y1:y1 + box_h, x1:x2] = cv2.addWeighted(roi, 0.15, np.full_like(roi, 255), 0.85, 0)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
    for i, t in enumerate(lines):
        dr.text((x1 + pad, y1 + pad + i * (line_h + gap)), t, font=font, fill=(17, 17, 17))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


writer = imageio.get_writer(DST, fps=rate * SLOW, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=2, quality=6)
tracks = {c: set() for c in CLASSES}
for i, frame in enumerate(sv.get_video_frames_generator(SRC, stride=STRIDE)):
    d_all = model.predict(frame, threshold=0.3)
    d_all = d_all[d_all.class_id < len(CLASSES)]  # drop the 7th slot the model sometimes emits
    heads_src = d_all[d_all.class_id == 0]  # low threshold so the blur misses fewer people
    d = d_all[d_all.confidence >= 0.5]
    d = tracker.update_with_detections(d)
    d = smoother.update_with_detections(d)
    for t, c in zip(d.tracker_id, d.class_id):
        tracks[CLASSES[int(c)]].add(int(t))
    labels = [f"#{t} {CLASSES[int(c)]}" for t, c in zip(d.tracker_id, d.class_id)]
    hb = np.concatenate([head_boxes(d[d.class_id == 0].xyxy, info.width, info.height),
                         head_boxes(heads_src.xyxy, info.width, info.height)])
    out = blur.annotate(frame.copy(), sv.Detections(xyxy=hb)) if len(hb) else frame.copy()
    out = boxes.annotate(out, d)
    dark = d.class_id != 2  # motorcycle labels (class 2) get white text, everything else dark text
    out = label.annotate(out, d[dark], [t for t, k in zip(labels, dark) if k])
    if (~dark).any():
        out = label_light.annotate(out, d[~dark], [t for t, k in zip(labels, dark) if not k])
    if CREDIT:  # CREDIT = [video credit line, optional creator link]
        lines = ["Created by J. Yousuf", TOOLS, CREDIT[0]]
        lines += [CREDIT[1]] if len(CREDIT) > 1 else []
        out = draw_credit(out, lines)
    writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    if i % 50 == 0:
        print("frame", i, flush=True)
writer.close()
print({k: len(v) for k, v in tracks.items()})
