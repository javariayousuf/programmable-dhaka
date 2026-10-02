import json, os, shutil, sys
import cv2
from pathlib import Path
from rfdetr import RFDETRSmall

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # score(), load_gt(), CONF

# How much labeling does it take? Train on every 4th, every 2nd, and every frame of the training clips,
# then score each model on the never-seen clip 1. Frames are taken at a regular stride so each subset
# still covers the whole clip.
ROOT = str(Path(__file__).resolve().parent.parent)
# usage: python experiment_label_budget.py [training clips] [test clip]   (default: clip1,clip2 and clip3, the setup of Test 1)
train = (sys.argv[1] if len(sys.argv) > 1 else "clip1,clip2").split(",")
holdout = sys.argv[2] if len(sys.argv) > 2 else "clip3"
CLIPS = {n: f"{ROOT}/labels/{n}" for n in train}
HOLDOUT = f"{ROOT}/labels/{holdout}"
EPOCHS = 20
BUDGETS = [("25pct", 4), ("50pct", 2), ("100pct", 1)]
OUT_JSON = f"{ROOT}/eval/label_budget_{holdout}.json"


def build(name, stride):
    out = f"{ROOT}/exp_data/{name}"
    shutil.rmtree(out, ignore_errors=True)
    n_boxes = {}
    for split in ("train", "valid"):
        os.makedirs(f"{out}/{split}")
        images, annos, cats, iid, aid = [], [], None, 0, 0
        for tag, src in CLIPS.items():
            coco = json.load(open(f"{src}/annotations.coco.json"))
            cats = coco["categories"]
            keep = [im for k, im in enumerate(coco["images"]) if k % stride == 0]
            remap = {}
            for im in keep:
                new = f"{tag}_{im['file_name']}"
                os.link(f"{src}/images/{im['file_name']}", f"{out}/{split}/{new}")
                remap[im["id"]] = iid
                images.append({**im, "id": iid, "file_name": new}); iid += 1
            for a in coco["annotations"]:
                if a["image_id"] in remap:
                    annos.append({**a, "id": aid, "image_id": remap[a["image_id"]]}); aid += 1
        json.dump({"images": images, "annotations": annos, "categories": cats}, open(f"{out}/{split}/_annotations.coco.json", "w"))
    for a in annos:
        n_boxes[a["category_id"]] = n_boxes.get(a["category_id"], 0) + 1
    return out, len(images), n_boxes


def main():
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    holdout_coco, gt = ev.load_gt(HOLDOUT)
    frames = [(im["id"], cv2.imread(f"{HOLDOUT}/images/{im['file_name']}")) for im in holdout_coco["images"]]
    for name, stride in BUDGETS:
        if name in results:
            continue
        data, n_img, boxes = build(name, stride)
        run = f"{ROOT}/runs/exp_{name}"
        shutil.rmtree(run, ignore_errors=True)
        RFDETRSmall().train(dataset_dir=data, epochs=EPOCHS, batch_size=2, grad_accum_steps=4, output_dir=run)
        model = RFDETRSmall(pretrain_weights=f"{run}/checkpoint_best_total.pth")
        preds = {}
        for iid, frame in frames:
            d = model.predict(frame, threshold=ev.CONF)
            preds[iid] = [{"category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)}
                          for (x1, y1, x2, y2), c, s in zip(d.xyxy, d.class_id, d.confidence) if int(c) + 1 in ev.NAMES]
        results[name] = {"train_images": n_img, "train_boxes": {str(k): v for k, v in boxes.items()},
                         f"{holdout}_score": ev.score(gt, preds)}
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        shutil.rmtree(run, ignore_errors=True)       # keep the numbers, drop the checkpoints
        shutil.rmtree(data, ignore_errors=True)
        print("done", name, results[name][f"{holdout}_score"]["rickshaw"], flush=True)


if __name__ == "__main__":
    main()
