# programmable-dhaka

**Teaching RF-DETR to identify rickshaws in Dhaka street footage, with Roboflow's open source tools.**

<img src="media/claude_orange.png" width="12" height="12" alt=""> Built with Claude Code, Sonnet 5.5

![The fine-tuned model on a Dhaka street clip it never trained on](media/hero_dhaka_street.gif)

*The Dhaka street clip, which RF-DETR never trained on. Blue is a person, pink a rickshaw, purple a motorcycle, amber a bicycle, yellow-green a cart.*

## Contents

- [Reasons to try Roboflow's open source tools](#reasons-to-try-roboflows-open-source-tools)
- [Why Dhaka](#why-dhaka)
- [What I found](#what-i-found)
- [Limits](#limits)
- [Product feedback](#product-feedback)
- [Questions I'd like to dig into](#questions-id-like-to-dig-into)
- [Details, reproduce steps and tips](#details-reproduce-steps-and-tips)
- [Built with](#built-with)
- [Footage credits](#footage-credits)

## Reasons to try Roboflow's open source tools

I wanted to understand the product by using it. I found it easy to stand up and onboard, and the speed is good.

- **Plug and play.** All I needed to start was videos I downloaded.
- **Intelligent modeling from RF-DETR.**
  - **Pre-labeling.** RF-DETR's pretrained model (already trained on everyday objects), with Grounding DINO (which finds things from a written description) for rickshaws and carts, drew a first set of labels, a box and a name around each vehicle. I call that first-pass setup the original pipeline. I corrected its mistakes instead of drawing every box myself: 39 of 168 boxes on the busy Dhaka street, and about two thirds on the clips that were mostly rickshaws.
  - **An unknown concept, from a handful of examples.** RF-DETR knows 80 everyday kinds of objects but had no word for a rickshaw. I trained RF-DETR on 141 pictures, but they show only about five different rickshaws, many of them near-copies of each other. About 10 to 15 minutes of training later, on a laptop, RF-DETR found about 130 of 163 rickshaws in a clip it had never seen, up from 45 with the original pipeline.
- **Easy to customize and tune.** supervision let me change how vehicles are followed, how steady the boxes are, what gets blurred, and how everything is drawn, with a few settings each.
- **Everything ran on one laptop,** at about 30 milliseconds per picture.

## Why Dhaka

I'm a Bangladeshi-American who has spent time in Dhaka, Bangladesh. I know from experience that there are really unique movement patterns: people with different modalities, really unexpected pathways of travel, non conformity in shapes and colors, culturally vibrant. I wanted to stress test the capabilities of the open source model with something I knew was complex.

## What I found

I corrected the first-pass labels by hand, with one rule: **anything that carries a passenger behind a driver is a rickshaw**, pedal or motorized. "Fine-tuned" below means RF-DETR trained further on those corrected labels. Each test scores the model on a clip it never trained on, against my corrected labels. The overall score runs from 0 to 1, and higher is better. These are small tests: each model trained on only two to four short clips, so read the numbers as a direction, not a benchmark.

### Fine-tuned RF-DETR learned rickshaws on a similar street, and missed the carts on a different one

![Three tests](media/chart_two_tests.png)

**Test 1: a street like the ones the model trained on.** I trained on the Dhaka street and e-rickshaw clips and tested on the rickshaw street clip. The original pipeline found 45 of 163 rickshaws and called 118 of them carts or bicycles. The fine-tuned model found about 130 (I trained the model twice: 131 and 135).

![Three frames where it gained](media/where_it_wins_rickshaw_street.jpg)

**Test 2: a street unlike its training footage.** I trained on the rickshaw street and e-rickshaw clips and tested on the Dhaka street clip. The fine-tuned model did worse than the original pipeline (0.44 against 0.81). Adding the night and rainy clips to training helped, to 0.54, but the score stayed below. The biggest gap is carts: the original pipeline found 41 of 49 and the fine-tuned model found 1. My guess: the carts in this market are metal trolleys piled with goods, and the training footage had banana carts and one umbrella cart, so the model had never seen this kind of cart. I have not tested that.

![Where it did worse](media/where_it_is_worse_dhaka.jpg)

My answer keys (the corrected labels each test is scored against) started as the original pipeline's own boxes, which flatters the original pipeline. The original pipeline still lost on the rickshaw street, so that does not explain Test 1, but that caveat probably explains some of the gap in Test 2. More in [docs/DETAILS.md](docs/DETAILS.md).

**The takeaway.** The two approaches fail in opposite ways. The original pipeline found most of the market's carts, but called rickshaws carts and bicycles. The fine-tuned model named rickshaws well, but missed the carts because it had never been shown that kind. Each is strong where the other is weak, so the next step is to use both, and to train on footage that includes the market's carts.

### The more a scene was just rickshaws, the more labels I had to correct

For the first three clips the first set of labels came from the original pipeline. I corrected far more of the original pipeline's guesses on the clips with less variety than on the varied one.

![How much of the first guess I corrected](media/chart_corrections.png)

On the busy Dhaka street I corrected 23% of the boxes. On the two clips that were mostly rickshaws I corrected 62% and 67%. My guess, which I have not tested: the first-pass tools already handle everyday objects, so a scene full of them needs little correcting, and a scene that is mostly the one thing they have no word for needs correcting nearly everywhere. (The night and rainy clips started from a model already fine-tuned on those two, so they are not a like-for-like comparison. At night 86% of the boxes needed correcting, mostly cars the model had called rickshaws.)

## Limits

- **Trucks are my choice, and I would fix it.** I never gave trucks their own label, so RF-DETR had no name for them and guessed. A truck often came out as a rickshaw. I would add a truck class and rerun.
- **Carts and rickshaws in a crowd** are the hardest cases.
- **People were not hand-reviewed, also my choice,** so they are left out of every score.

Pictures of the mistakes are in the [details](docs/DETAILS.md).

## Product feedback

Things that cost me time, and what I would suggest (rfdetr 1.11.1, supervision 0.30.6).

<img src="media/claude_orange.png" width="12" height="12" alt=""> Summarized by Claude Code.

- **Class labels are inconsistent between the model's output and the library's name list.** *Evidence:* the pretrained model returns each object as a number from 1 to 90, with gaps. The library's list of 80 names counts from zero and starts person, bicycle, car, motorcycle. So the number 1 (a person) looks up "bicycle," and the number 4 (a motorcycle) looks up "airplane." My first video labeled people "bicycle." *Suggestion:* return the names with the detections.
- **The removed-import error points to a place that does not have the answer.** *Evidence:* importing `rfdetr.util` fails with "rfdetr.util was removed in v1.9.0. Use rfdetr.utilities instead." I checked: `rfdetr.utilities` does not contain the class names, and `rfdetr.assets.coco_classes` does. *Suggestion:* name the real location in the message.
- **The model can return a class it was never trained on.** *Evidence:* after training on six kinds of objects, the model returned a seventh class number on 11 boxes in one night clip. I filter it out and do not know why. *Suggestion:* document it, or stop returning it.
- **The deprecation warnings name no replacement.** *Evidence:* creating `RFDETRBase()` prints "deprecated since v1.7.0. It will be removed in v2.0.0," and creating `sv.ByteTrack()` prints "deprecated since v0.28.0. It will be removed in v0.31.0." Neither message says what to use instead. *Suggestion:* name the replacement in the message.
- **The first training run needs one extra install and one Mac-only rule.** *Evidence:* `model.train` stops with an import error until `rfdetr[train]` is installed, and that error names the command, which is clear. On macOS the data loader then crashes with Python's generic multiprocessing message unless the call sits inside `if __name__ == "__main__":`. *Suggestion:* put both in the quickstart.

## Questions I'd like to dig into

- Could Grounding DINO (which finds things from a written description) and the fine-tuned model work together, so the carts get found too?
- How should near-duplicate frames (stills a fraction of a second apart that look almost identical) be handled when labeling video footage? In one test, 36 pictures did about as well as 141.
- Would RF-DETR still be fast enough at the edge, on the small computer that sits next to a camera? I measured about 30 milliseconds per picture on a laptop and have not measured a small device.

## Details, reproduce steps and tips

- [Details and caveats](docs/DETAILS.md): scores, pictures, how the labels were corrected, limits.
- [Reproduce it](docs/REPRODUCE.md): commands and layout.
- [Tips and learnings](docs/LEARNINGS.md): what I'd tell someone trying this.

## Built with

[RF-DETR](https://github.com/roboflow/rf-detr), [supervision](https://github.com/roboflow/supervision), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) (IDEA Research, via Hugging Face), PyTorch, OpenCV, matplotlib, and [Claude Code](https://claude.com/claude-code).

## Footage credits

All footage is from [Pexels](https://www.pexels.com) under the Pexels License. Everything shown here is altered, and the original videos are not in this repo.

- Dhaka street ("Suhrawardy Udyan TSC") by Faisal Ibne Kalam: [video](https://www.pexels.com/video/suhrawardy-udyan-tsc-26689635/), [profile](https://www.pexels.com/@faisal-ibne-kalam-774996459/)
- e-rickshaws ("Bustling Indian Street with Auto Rickshaws") by md Jahangir alam, filmed in India, not Dhaka: [video](https://www.pexels.com/video/bustling-indian-street-with-auto-rickshaws-38248681/), [profile](https://www.pexels.com/@mdjahangir/)
- Rickshaw street ("Colorful Rickshaws on Bustling Street") by Somogro Bangladesh: [video](https://www.pexels.com/video/colorful-rickshaws-on-bustling-street-36526831/), [profile](https://www.pexels.com/@somogrobangladesh/)
- Night traffic ("Vibrant City Night Traffic Scene") by Jubayer Hossain, tagged Dhaka and Chittagong: [video](https://www.pexels.com/video/vibrant-city-night-traffic-scene-35041521/), [profile](https://www.pexels.com/@jubayer-wh/)
- Rainy walk ("Rainy Day Street Scene in Dhaka, Bangladesh") by Latiful Jawad, labels only: [video](https://www.pexels.com/video/rainy-day-street-scene-in-dhaka-bangladesh-29662763/), [profile](https://www.pexels.com/@latiful-jawad-431220084/)

Created by J. Yousuf.
