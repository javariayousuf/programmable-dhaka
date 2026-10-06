from pathlib import Path
import json, os, shutil, sys

# usage: python make_train_dataset.py [clips] [out_dir] [--no-truck]
#   clips: comma list of folders under labels/ (default: the final split below). Example: clip1,clip2 trains on those two
#   --no-truck: leave the truck boxes out (and the truck class), so a run can be compared with and without the truck class

NO_TRUCK = "--no-truck" in sys.argv
sys.argv = [a for a in sys.argv if a != "--no-truck"]
ROOT = str(Path(__file__).resolve().parent.parent)  # repo root
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/train_dataset"
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip2", "clip3", "night", "rain"]  # default: clip 1 (Dhaka street) is the held-out test
CLIPS = {n: f"{ROOT}/labels/{n}" for n in names}
# train on both clips. The valid split is only a progress check: it overlaps the training data,
# so its score is optimistic. A fair test needs a third clip the model has never seen.
shutil.rmtree(OUT, ignore_errors=True)
for split in ("train", "valid"):
    os.makedirs(f"{OUT}/{split}")
    images, annos, cats = [], [], {}
    iid = aid = 0
    for tag, src in CLIPS.items():
        coco = json.load(open(f"{src}/annotations.coco.json"))
        for c in coco["categories"]:       # clips are labeled at different times, so some lack the newest class; merge by id
            if not (NO_TRUCK and c["name"] == "truck"): cats[c["id"]] = c
        truck_ids = {c["id"] for c in coco["categories"] if c["name"] == "truck"} if NO_TRUCK else set()
        remap = {}
        for im in coco["images"]:
            new = f"{tag}_{im['file_name']}"
            os.link(f"{src}/images/{im['file_name']}", f"{OUT}/{split}/{new}")
            remap[im["id"]] = iid
            images.append({**im, "id": iid, "file_name": new}); iid += 1
        for a in coco["annotations"]:
            if a["category_id"] in truck_ids: continue
            annos.append({**a, "id": aid, "image_id": remap[a["image_id"]]}); aid += 1
    json.dump({"images": images, "annotations": annos, "categories": [cats[k] for k in sorted(cats)]}, open(f"{OUT}/{split}/_annotations.coco.json", "w"))
    per = {}
    for a in annos: per[a["category_id"]] = per.get(a["category_id"], 0) + 1
    names = {k: c["name"] for k, c in cats.items()}
    print(split, len(images), "images,", {names[k]: v for k, v in sorted(per.items())})
