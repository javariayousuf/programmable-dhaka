import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "media"
FONT = os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Futura.ttc")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333", "text.color": "#111",
                     "axes.labelcolor": "#111", "xtick.color": "#111", "ytick.color": "#111", "font.size": 13})
RICK, CART, BIKE, MOTO, GRAY, BLUE = "#db3069", "#e0ff4f", "#EC9F05", "#3B0086", "#c9cdd3", "#3777ff"


def save(fig, name):
    fig.savefig(MEDIA / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1) what happened to every rickshaw, for each of the two tests
order = [("rickshaw", "Called a rickshaw (correct)", RICK), ("cart", "Called a cart", CART), ("bicycle", "Called a bicycle", BIKE),
         ("motorcycle", "Called a motorcycle", MOTO), ("missed", "Not found at all", GRAY)]
for jf, ck, out, what in (("results_clip3_holdout.json", "clip3", "chart_rickshaw_outcomes_street.png", "the rickshaw street clip"),
                          ("results_clip1_holdout.json", "clip1", "chart_rickshaw_outcomes_dhaka.png", "the Dhaka street clip")):
    r = json.load(open(ROOT / "eval" / jf))[ck]
    old, new = r["baseline_confusion"]["rickshaw"], r["finetuned_confusion"]["rickshaw"]
    fig, ax = plt.subplots(figsize=(10, 3.1))
    labeled = set()
    for row, (name, d) in enumerate((("Original pipeline", old), ("After fine-tuning", new))):
        left = 0
        for key, label, col in order:
            v = d.get(key, 0)
            if v:
                ax.barh(row, v, left=left, color=col, edgecolor="#333", linewidth=0.8, label=label if key not in labeled else None); labeled.add(key)
                if v >= 12: ax.text(left + v / 2, row, str(v), ha="center", va="center", color="white" if key == "motorcycle" else "#111", fontsize=13)
                left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Original\npipeline", "After\nfine-tuning"]); ax.invert_yaxis()
    ax.set_xlabel(f"The {sum(old.values())} rickshaws in {what} (never used for training)")
    h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=11)
    ax.set_title("What happened to every rickshaw", loc="left", fontsize=16)
    save(fig, out)
res = json.load(open(ROOT / "eval/results_clip1_holdout.json"))["clip1"]

# 3) how much of the first guess needed fixing, per clip
names = {"clip1": "Dhaka street\n(test)", "clip2": "e-rickshaws", "clip3": "rickshaw\nstreet", "night": "night\ntraffic", "rain": "rainy\nwalk"}
tot, fixed = [], []
for c in names:
    j = json.load(open(ROOT / f"labels/{c}/annotations.coco.json"))
    veh = [a for a in j["annotations"] if a["category_id"] > 1]
    tot.append(len(veh)); fixed.append(sum(1 for a in veh if a.get("corrected")))
fig, ax = plt.subplots(figsize=(9, 3.9))
ax.bar(range(len(names)), tot, color=GRAY, edgecolor="#333"); ax.bar(range(len(names)), fixed, color=RICK, edgecolor="#333")
for i, (t, f) in enumerate(zip(tot, fixed)): ax.text(i, t + 8, f"{round(100 * f / t)}%\nchanged", ha="center", fontsize=12)
ax.set_xticks(range(len(names))); ax.set_xticklabels(list(names.values())); ax.set_ylim(0, max(tot) * 1.25)
ax.set_ylabel("Vehicle boxes"); ax.set_title("How much of the first guess I changed", loc="left", fontsize=16)
save(fig, "chart_corrections.png")

# 4) what the model had to learn from (explains why motorcycles got worse)
cnt = {}
for c in ("clip2", "clip3", "night", "rain"):
    j = json.load(open(ROOT / f"labels/{c}/annotations.coco.json"))
    for a in j["annotations"]:
        if a["category_id"] > 1: cnt[a["category_id"]] = cnt.get(a["category_id"], 0) + 1
cls = [(6, "Car", GRAY), (4, "Rickshaw", RICK), (2, "Bicycle", BIKE), (3, "Motorcycle", MOTO), (5, "Cart", CART)]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (k, n, col) in enumerate(cls):
    ax.barh(i, cnt.get(k, 0), color=col, edgecolor="#333"); ax.text(cnt.get(k, 0) + 2, i, str(cnt.get(k, 0)), va="center", fontsize=13)
ax.set_yticks(range(len(cls))); ax.set_yticklabels([n for _, n, _ in cls]); ax.invert_yaxis()
ax.set_xlabel("Labeled boxes the model learned from"); ax.set_title("What it learned from", loc="left", fontsize=16)
save(fig, "chart_training_boxes.png")

# 5) found per class on the Dhaka street clip
fig, ax = plt.subplots(figsize=(9, 4))
cls5 = [("bicycle", BIKE), ("motorcycle", MOTO), ("rickshaw", RICK), ("cart", CART)]
for i, (n, col) in enumerate(cls5):
    gtn = res["baseline"][n]["gt"]
    o, f = res["baseline"][n]["tp"], res["finetuned"][n]["tp"]
    ax.bar(i - 0.2, o, 0.38, color=GRAY, edgecolor="#333", label="Original pipeline" if i == 0 else None)
    ax.bar(i + 0.2, f, 0.38, color=col, edgecolor="#333", label="After fine-tuning" if i == 0 else None)
    ax.text(i - 0.2, o + 1, str(o), ha="center", fontsize=12); ax.text(i + 0.2, f + 1, str(f), ha="center", fontsize=12)
ax.set_xticks(range(len(cls5))); ax.set_xticklabels([f"{n}\n({res['baseline'][n]['gt']} in the clip)" for n, _ in cls5])
ax.set_ylabel("Found correctly (box and name)"); ax.legend(frameon=False)
ax.set_title("Dhaka street clip: found, by kind of vehicle", loc="left", fontsize=16)
save(fig, "chart_per_class.png")

# 6) two honest tests side by side
a = json.load(open(ROOT / "eval/results_clip3_holdout.json"))["clip3"]
two = json.load(open(ROOT / "eval/results_clip1_holdout_trained_on_2_clips.json"))["clip1"]
fin = res
rows = [("Rickshaw street\ntrained on 2 similar clips", a["baseline"]["ALL VEHICLES"]["f1"], a["finetuned"]["ALL VEHICLES"]["f1"], None),
        ("Dhaka street\ntrained on 2 other clips", two["baseline"]["ALL VEHICLES"]["f1"], two["finetuned"]["ALL VEHICLES"]["f1"], None),
        ("Dhaka street\ntrained on 4 other clips", fin["baseline"]["ALL VEHICLES"]["f1"], fin["finetuned"]["ALL VEHICLES"]["f1"], None)]
fig, ax = plt.subplots(figsize=(9, 4))
for i, (n, o, f, _) in enumerate(rows):
    ax.bar(i - 0.2, o, 0.38, color=GRAY, edgecolor="#333", label="Original pipeline" if i == 0 else None)
    ax.bar(i + 0.2, f, 0.38, color=RICK, edgecolor="#333", label="After fine-tuning" if i == 0 else None)
    ax.text(i - 0.2, o + 0.02, f"{o:.2f}", ha="center", fontsize=13); ax.text(i + 0.2, f + 0.02, f"{f:.2f}", ha="center", fontsize=13)
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1.05)
ax.set_ylabel("Overall score on a clip never used in training (higher is better)", fontsize=10); ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_title("It depends on how much the new scene looks like the training footage", loc="left", fontsize=14)
save(fig, "chart_two_tests.png")
print("charts written")
