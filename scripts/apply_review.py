import json, shutil, sys
from pathlib import Path

# Writes the answers from review_cards.py back into the label file. Makes a backup first.
# usage: python apply_review.py <dataset_dir> [review_dir]
D = Path(sys.argv[1])
REV = D / (sys.argv[2] if len(sys.argv) > 2 else "review")   # second argument: review_boxes, after review_cards.py --mixed
IDS = {"person": 1, "bicycle": 2, "motorcycle": 3, "rickshaw": 4, "cart": 5, "car": 6, "truck": 7}
src = D / "annotations.coco.json"
shutil.copy(src, D / f"annotations.coco.before_{REV.name}.json")
coco = json.load(open(src))
tracks = {t["id"]: t for t in json.load(open(REV / "tracks.json"))}
answers = {int(k): v for k, v in json.load(open(REV / "answers.json")).items()}
by = {a["id"]: a for a in coco["annotations"]}
drop, changed, kept, mixed = set(), 0, 0, []
for tid, ans in answers.items():
    t = tracks[tid]
    if ans == "mixed: check frames":
        mixed.append(tid); continue
    for bid in t["box_ids"]:
        if ans == "not a vehicle":
            drop.add(bid)
        elif by[bid]["category_id"] != IDS[ans]:
            by[bid]["category_id"] = IDS[ans]; by[bid]["corrected"] = True; changed += 1
        else:
            kept += 1
dup = set(json.load(open(REV / "dropped.json"))) if (REV / "dropped.json").exists() else set()
bad = {a["id"] for a in coco["annotations"] if a["category_id"] > len(IDS)}      # the model sometimes emits a 7th class that does not exist
coco["annotations"] = [a for a in coco["annotations"] if a["id"] not in drop | dup | bad]
print(f"also removed {len(dup)} duplicate boxes and {len(bad)} boxes with an invalid class")
json.dump(coco, open(src, "w"))
left = [t["id"] for t in tracks.values() if t["id"] not in answers]
print(f"{len(answers)} of {len(tracks)} vehicles answered: {changed} boxes relabeled, {kept} already right, {len(drop)} removed.")
if mixed: print("Marked mixed (left untouched, review these frames by hand):", mixed)
if left: print(f"{len(left)} vehicles still unanswered, left as the model guessed.")
