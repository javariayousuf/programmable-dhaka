# programmable-dhaka

**Teaching RF-DETR to identify rickshaws in Dhaka street footage, with supervision.**

<img src="media/claude_orange.png" width="12" height="12" alt=""> Built with Claude Code, Sonnet 5.5

![The fine-tuned model on a Dhaka street clip it never trained on](media/hero_dhaka_street.gif)

*The Dhaka street clip, which RF-DETR never trained on.*

<img src="media/dot_person.png" width="12" height="12" alt="blue dot"> person&nbsp;&nbsp;·&nbsp;&nbsp;<img src="media/dot_rickshaw.png" width="12" height="12" alt="pink dot"> rickshaw&nbsp;&nbsp;·&nbsp;&nbsp;<img src="media/dot_motorcycle.png" width="12" height="12" alt="purple dot"> motorcycle&nbsp;&nbsp;·&nbsp;&nbsp;<img src="media/dot_bicycle.png" width="12" height="12" alt="amber dot"> bicycle&nbsp;&nbsp;·&nbsp;&nbsp;<img src="media/dot_cart.png" width="12" height="12" alt="yellow-green dot"> cart

## Contents

- [Reasons to try Roboflow's open source tools](#reasons-to-try-roboflows-open-source-tools)
- [Why Dhaka](#why-dhaka)
- [What I learned](#what-i-learned)
- [Limits](#limits)
- [Product feedback](#product-feedback)
- [Questions I'd like to dig into](#questions-id-like-to-dig-into)
- [Scores, how to rerun it, and tips](#scores-how-to-rerun-it-and-tips)
- [Built with open source tools and Claude Code](#built-with-open-source-tools-and-claude-code)

## Reasons to try Roboflow's open source tools

- **Plug and play.** All I needed to start was my laptop and free videos I downloaded.
- **Intelligent modeling from RF-DETR.**
  - **Pre-labeling.** RF-DETR was already trained on everyday objects, and Grounding DINO (which finds things from a written description) handled rickshaws and carts. Together they drew a first set of labels, each one a box and a name around a vehicle. I call that first-pass setup the original pipeline. I corrected its mistakes instead of drawing every box myself: 39 of 168 boxes on the busy Dhaka street, and about two thirds on the clips that were mostly rickshaws.
  - **An unknown concept, from a handful of examples.** RF-DETR knows 80 everyday kinds of objects, but it had no word for a rickshaw. I trained it on 141 pictures, but they show only about five different rickshaws, many of them near-copies of each other. About 10 to 15 minutes later, on a laptop, it found about 130 of 163 rickshaws in a clip it had never seen. The original pipeline found 45.
- **Easy to customize and tune.** supervision let me change how vehicles are followed, how steady the boxes are, what gets blurred, and how everything is drawn. Each one took a few settings.
- **Everything ran on one laptop,** at about 30 milliseconds per picture.

## Why Dhaka

I'm a Bangladeshi-American who has spent time in Dhaka, Bangladesh. I know from experience that there are really unique movement patterns: people with different modalities, really unexpected pathways of travel, non conformity in shapes and colors, culturally vibrant. I wanted to stress test the capabilities of the open source model with something I knew was complex.

## What I learned

I corrected the first-pass labels by hand, with one rule: **anything that carries a passenger behind a driver is a rickshaw**, pedal or motorized. "Fine-tuned" means RF-DETR trained further on those corrected labels. Each test scores the model on a clip it never trained on, against my corrected labels. Scores run from 0 to 1, and higher is better. The tests are small, so I read the numbers as a direction, not a benchmark.

### A few different examples can teach something new

*In this case:* about five different rickshaws, shown many times, took RF-DETR from finding 45 of 163 rickshaws to about 130, on a clip it had never seen.

### The model learns what it is shown, and not much more

*In this case, with only a few short clips:* each test trained on two to four short clips, so the model only knows the kinds of scenes in them. Test 1 was scored on a street shot like one of those clips: a steady side-on view, mostly rickshaws. Test 2 was scored on a crowded market shot by hand, full of metal trolley carts, which the training clips did not have. I think that difference explains the gap, and it points to one thing to try: more varied training footage. I have not tested that.

![Three tests](media/chart_two_tests.png)

**Test 1: a steady side-on street, like one of the training clips.** I trained on the Dhaka street and e-rickshaw clips and scored on the rickshaw street clip. The original pipeline found 45 of 163 rickshaws and called 118 of them carts or bicycles. The fine-tuned model found about 130. I trained it twice and got 131 and 135.

![Three frames where it gained](media/where_it_wins_rickshaw_street.jpg)

**Test 2: a crowded, handheld market full of trolley carts.** I trained on the rickshaw street and e-rickshaw clips, which are steady side-on streets with mostly rickshaws, and scored on the Dhaka street clip. The original pipeline scored higher here: 0.81 against 0.44, or 0.54 once I added the night and rainy clips. Most of the gap is carts. The original pipeline found 41 of 49, and the fine-tuned model found 1. My guess, which I have not tested: the market's carts are metal trolleys piled with goods, and the training clips only had banana carts and one umbrella cart.

![Frames where the original pipeline found more](media/where_it_is_worse_dhaka.jpg)

One caution about how I scored. My answer keys, the corrected labels each test is scored against, started as the original pipeline's own boxes. That gives the original pipeline an advantage. It also scored lower on the rickshaw street, so the advantage does not explain Test 1, but it probably explains some of the gap in Test 2. More in the [details](docs/DETAILS.md).

**What I take from it.** The two approaches seemed strong in different places. The original pipeline found most of the market's carts but often called rickshaws carts or bicycles. The fine-tuned model named rickshaws well but found almost none of the carts, which I think is because it had not been shown that kind. Using both looks like a natural next step, along with training on footage that includes the market's carts.

### A scene that is mostly one thing needs the most correcting

*In this case:* the first three clips started from the original pipeline. I corrected far more of its guesses on the clips that were mostly rickshaws than on the varied one.

![How much of the first guess I corrected](media/chart_corrections.png)

On the busy Dhaka street I corrected 23% of the boxes. On the two clips that were mostly rickshaws I corrected 62% and 67%. My guess, which I have not tested: the first-pass tools already handle everyday objects, so a scene full of them needs little correcting, and a scene that is mostly the one thing they have no word for needs correcting almost everywhere. The night and rainy clips started from a model already fine-tuned on those two, so they are not a fair comparison. At night I corrected 86% of the boxes, mostly cars the model had called rickshaws.

### Pictures that look almost the same may not teach much extra

*In this case:* training on every 4th picture, 36 in all, found 124 of 163 rickshaws. Training on all 141 found 127. That is one run each, and training the same setup twice gave 131 and 135, so the difference is inside the noise. More in the [details](docs/DETAILS.md).

## Limits

**These are not solid findings about the model.** The samples are small, and I did not check every step as carefully as I would on a real project.

- **Trucks have no label of their own.** I never gave trucks a label, so the model had nothing to call them and often said "rickshaw." I expect a truck label and a retrain would help, but I have not tried it.
- **People are not scored.** I only corrected labels for vehicles, so how well the model finds people is not measured.

More pictures of the mistakes are in the [details](docs/DETAILS.md).

## Product feedback

Tested with rfdetr 1.11.1 and supervision 0.30.6, on Python 3.14 and macOS (Apple silicon). Each item shows what I ran, what came back, and an idea. I may be missing a better way to do some of these, so please read them as questions as much as suggestions.

<img src="media/claude_orange.png" width="12" height="12" alt=""> Summarized by Claude Code.

**1. I found `model.class_names` and `class_id` hard to line up.**

```python
d = model.predict(image, threshold=0.5)          # model = RFDETRBase()
d.class_id                                       # [1, 4, 1, 1, ...]
[model.class_names[i] for i in d.class_id]       # ['bicycle', 'airplane', 'bicycle', 'bicycle', ...]
d.data["class_name"]                             # ['person', 'motorcycle', 'person', 'person', ...]
```

The ids are COCO's: 1 to 90, with gaps. The name list has 80 entries counted from zero. So a name I looked up by id landed on a different object, and my first video labeled people "bicycle." The right names were in `d.data["class_name"]`, which I only found later. *Idea:* make `class_names` a dictionary keyed by class id, or add a line to its docstring pointing to `detections.data["class_name"]`.

**2. A question about the removed-import message.**

```python
import rfdetr.util   # ImportError: rfdetr.util was removed in v1.9.0. Use rfdetr.utilities instead.
```

The message sent me to `rfdetr.utilities`, but the class names are not there (`hasattr(rfdetr.utilities, "COCO_CLASS_NAMES")` is `False`). They are in `rfdetr.assets.coco_classes`, where I eventually found them. *Idea:* the message could point there.

**3. I looked for the replacement in the deprecation warnings.**

```python
RFDETRBase()      # FutureWarning: The `RFDETRBase` was deprecated since v1.7.0. It will be removed in v2.0.0.
sv.ByteTrack()    # FutureWarning: The `ByteTrack` was deprecated since v0.28.0. It will be removed in v0.31.0.
```

Neither message names a replacement, but the docs do: `RFDETRBase` is replaced by `RFDETRSmall` (or Nano, Medium, Large) in the [RF-DETR migration guide](https://rfdetr.roboflow.com/latest/getting-started/migration/), and `sv.ByteTrack` by `ByteTrackTracker` from the `trackers` package, with `update_with_detections()` renamed `update()`, on [supervision's deprecated page](https://supervision.roboflow.com/latest/deprecated/). *Idea:* adding the replacement to the warning text would save a trip to the docs.

**4. Setting up training took me two small steps.** `model.train(...)` asked me to install `rfdetr[train]`, and the error named the command, which was clear. On macOS the data loader then stopped with Python's generic multiprocessing error until I wrapped the call in `if __name__ == "__main__":`. *Idea:* a line about each in the quickstart might help the next person.

## Questions I'd like to dig into

- What makes one scene count as "similar" to another for the model? Is it the camera angle, how crowded it is, the time of day, the weather, or the mix of vehicles? And which kinds of clips would teach it rickshaws best? I would test it by adding one kind of clip at a time (night, rain, a crowded market, a new camera angle) and scoring each on the same clips every time.
- Could Grounding DINO (which finds things from a written description) and the fine-tuned model work together, so the carts get found too?
- How should near-duplicate frames be handled when labeling video? These are stills a fraction of a second apart that look almost the same. In one test, 36 pictures did about as well as 141.
- Would RF-DETR still be fast enough at the edge, on the small computer that sits next to a camera? I measured about 30 milliseconds per picture on a laptop, and I have not measured a small device.

## Scores, how to rerun it, and tips

- [Details and caveats](docs/DETAILS.md): scores, pictures, how the labels were corrected, limits.
- [Reproduce it](docs/REPRODUCE.md): commands and layout.
- [Tips and learnings](docs/LEARNINGS.md): what I'd tell someone trying this.

## Built with open source tools and Claude Code

[RF-DETR](https://github.com/roboflow/rf-detr), [supervision](https://github.com/roboflow/supervision), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) (IDEA Research, via Hugging Face), PyTorch, OpenCV, matplotlib, and [Claude Code](https://claude.com/claude-code).

### Footage credits

All footage is from [Pexels](https://www.pexels.com) under the Pexels License. Everything shown here is altered, and the original videos are not in this repo.

- Dhaka street ("Suhrawardy Udyan TSC") by Faisal Ibne Kalam: [video](https://www.pexels.com/video/suhrawardy-udyan-tsc-26689635/), [profile](https://www.pexels.com/@faisal-ibne-kalam-774996459/)
- e-rickshaws ("Bustling Indian Street with Auto Rickshaws") by md Jahangir alam, filmed in India, not Dhaka: [video](https://www.pexels.com/video/bustling-indian-street-with-auto-rickshaws-38248681/), [profile](https://www.pexels.com/@mdjahangir/)
- Rickshaw street ("Colorful Rickshaws on Bustling Street") by Somogro Bangladesh: [video](https://www.pexels.com/video/colorful-rickshaws-on-bustling-street-36526831/), [profile](https://www.pexels.com/@somogrobangladesh/)
- Night traffic ("Vibrant City Night Traffic Scene") by Jubayer Hossain, tagged Dhaka and Chittagong: [video](https://www.pexels.com/video/vibrant-city-night-traffic-scene-35041521/), [profile](https://www.pexels.com/@jubayer-wh/)
- Rainy walk ("Rainy Day Street Scene in Dhaka, Bangladesh") by Latiful Jawad, labels only: [video](https://www.pexels.com/video/rainy-day-street-scene-in-dhaka-bangladesh-29662763/), [profile](https://www.pexels.com/@latiful-jawad-431220084/)

Created by J. Yousuf.
