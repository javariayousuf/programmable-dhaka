import os, sys
import cv2
import imageio

# usage: python render_labels_video.py <preview_dir> <out.mp4> [fps]
PREVIEW = sys.argv[1]  # folder of preview jpgs from preview_prelabels.py
OUT = sys.argv[2]
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0  # stills are 0.2 s apart, so 5 fps is real time

files = sorted(f for f in os.listdir(PREVIEW) if f.endswith(".jpg"))
# H.264 + yuv420p plays everywhere (OpenCV's mp4v shows as a green screen in some players)
writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=2, quality=5)
for f in files:
    img = cv2.imread(f"{PREVIEW}/{f}")
    h, w = img.shape[:2]
    n = int(f.split("_")[1].split(".")[0])
    cv2.rectangle(img, (w - 190, 0), (w, 44), (255, 255, 255), -1)
    cv2.putText(img, f"frame {n}", (w - 178, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (17, 17, 17), 2)
    writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
writer.close()
print("wrote", len(files), "frames at", FPS, "fps")
