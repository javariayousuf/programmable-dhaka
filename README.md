# programmable-dhaka

**Teaching RF-DETR to identify rickshaws in Dhaka street footage, with Roboflow's open source tools.**

![The fine-tuned model on a Dhaka street clip it never trained on](media/hero_dhaka_street.gif)

*The Dhaka street clip, which RF-DETR never trained on. Blue is a person, pink a rickshaw, purple a motorcycle, amber a bicycle, yellow-green a cart. Heads are blurred.*

> **In 30 seconds**
> - **What:** I taught Roboflow's RF-DETR to identify rickshaws, using street video from Dhaka.
> - **Result:** on a street like its training footage it found about 130 of 163 rickshaws, up from 45. On a very different street it did worse than the original setup, mostly because it missed the carts.
> - **Lesson:** fine-tuning works on what it is shown, so show it the messy cases.

## Contents

- [Reasons to try Roboflow's open source tools](#reasons-to-try-roboflows-open-source-tools)
- [Why I picked Dhaka: a scene I knew would stress the model](#why-i-picked-dhaka-a-scene-i-knew-would-stress-the-model)
- [I corrected far more guesses when the footage had less variety](#i-corrected-far-more-guesses-when-the-footage-had-less-variety)
- [Fine-tuning helped on familiar footage and fell short on unfamiliar footage](#fine-tuning-helped-on-familiar-footage-and-fell-short-on-unfamiliar-footage)
- [Where it still goes wrong: trucks, crowded carts, unreviewed people](#where-it-still-goes-wrong-trucks-crowded-carts-unreviewed-people)
- [Product feedback: five things that cost me time](#product-feedback-five-things-that-cost-me-time)
- [Three questions I'd like to dig into: carts, near-duplicate frames, the edge](#three-questions-id-like-to-dig-into-carts-near-duplicate-frames-the-edge)
- [More: scores, pictures, reproduce steps and tips](#more-scores-pictures-reproduce-steps-and-tips)
- [Built with RF-DETR, supervision and Grounding DINO](#built-with-rf-detr-supervision-and-grounding-dino)
- [Footage credits: five Pexels clips and their creators](#footage-credits-five-pexels-clips-and-their-creators)

## Reasons to try Roboflow's open source tools

I wanted to understand the product by using it. I found it easy to stand up and onboard, and the speed is good.

- **Plug and play, with intelligent modeling from RF-DETR.** All I needed was videos I downloaded. RF-DETR's pretrained model, with Grounding DINO (which finds things from a written description) for rickshaws and carts, drew a first set of labels, so I corrected mistakes instead of drawing every box myself. I corrected 39 of 168 boxes on the busy Dhaka street, and about two thirds on the clips that were mostly rickshaws.
- **RF-DETR learned an unknown concept from a handful of examples.** It had no word for a rickshaw. I trained it on 141 pictures, but they show only about five different rickshaws, many of them near-copies of each other. About 10 to 15 minutes of training later, on a laptop, it found about 130 of 163 rickshaws in a clip it had never seen, up from 45 with the original setup.
- **Easy to customize and tune.** supervision let me change how vehicles are followed, how steady the boxes are, what gets blurred, and how everything is drawn, with a few settings each.
- **Everything ran on one laptop,** at about 30 milliseconds per picture.

## Why I picked Dhaka: a scene I knew would stress the model

I'm a Bangladeshi-American who has spent time in Dhaka, Bangladesh. I know from experience that there are really unique movement patterns: people with different modalities, really unexpected pathways of travel, non conformity in shapes and colors, culturally vibrant. I wanted to stress test the capabilities of the open source model with something I knew was complex.

## I corrected far more guesses when the footage had less variety

To get started I corrected a first set of labels by hand, with one rule: **anything that carries a passenger behind a driver is a rickshaw**, pedal or motorized. For the first three clips the first set came from what I call the **original pipeline**: RF-DETR plus Grounding DINO. I corrected far more of its guesses on the clips with less variety than on the varied one.

![How much of the first guess I corrected](media/chart_corrections.png)

On the busy Dhaka street I corrected 23% of the boxes. On the two clips that were mostly rickshaws I corrected 62% and 67%. My guess, which I have not tested: the first-pass tools already handle everyday objects, so a scene full of them needs little correcting, and a scene that is mostly the one thing they have no word for needs correcting nearly everywhere. (The night and rainy clips started from a model already fine-tuned on those two, so they are not a like-for-like comparison. At night 86% of the boxes needed correcting, mostly cars the model had called rickshaws.)

## Fine-tuning helped on familiar footage and fell short on unfamiliar footage

"Fine-tuned" means RF-DETR trained further on my own corrected labels. Each test scores the model on a clip it never trained on, against my corrected labels. The overall score runs from 0 to 1, and higher is better.

![Three tests](media/chart_two_tests.png)

**Test 1: a street like the ones it trained on.** I trained on the Dhaka street and e-rickshaw clips and tested on the rickshaw street clip. The original pipeline found 45 of 163 rickshaws and called 118 of them carts or bicycles. The fine-tuned model found about 130 (I trained it twice: 131 and 135).

![Three frames where it gained](media/where_it_wins_rickshaw_street.jpg)

**Test 2: a street unlike its training footage.** I trained on the rickshaw street and e-rickshaw clips and tested on the Dhaka street clip. It did worse than the original pipeline (0.44 against 0.81). Adding the night and rainy clips helped, to 0.54, but it stayed below. The biggest gap is carts: the original pipeline found 41 of 49 and the fine-tuned model found 1. My guess: the carts in this market are metal trolleys piled with goods, and the training footage had banana carts and one umbrella cart, so the model had never seen this kind of cart. I have not tested that.

![Where it did worse](media/where_it_is_worse_dhaka.jpg)

My answer keys started as the original pipeline's own boxes, which flatters it. It still lost on the rickshaw street, so that does not explain Test 1, but it probably explains some of the gap in Test 2. More in [docs/DETAILS.md](docs/DETAILS.md).

**The takeaway.** Fine-tuning worked on what it was shown, on scenes like the ones it was shown. It did not carry over to a scene with kinds of carts it had not seen, and the text-prompt pipeline was better there. A hybrid is the obvious next step, and so is footage that includes the market's carts.

## Where it still goes wrong: trucks, crowded carts, unreviewed people

- Trucks are not a class, so the model guesses, and a truck often comes out as a rickshaw.
- Carts and rickshaws in a crowd are the hardest cases.
- People were not hand-reviewed, so they are left out of every score.

Pictures of the mistakes are in the [details](docs/DETAILS.md).

## Product feedback: five things that cost me time

Things that cost me time, and what I would suggest (rfdetr 1.11.1, supervision 0.30.6). I checked each one against the library.

- **The names and the numbers don't line up.** The model returns each object as a number up to 90, with gaps, but the library's name list has 80 entries starting at zero, so a name lands on the wrong object. My people came out as "bicycle." *Suggestion:* return the names with the detections.
- **A removed-import error points to the wrong place.** It says to use `rfdetr.utilities`, but the class names are in `rfdetr.assets.coco_classes`. *Suggestion:* name the real location.
- **A seventh kind of object that does not exist.** After training on six, the model occasionally returned a seventh number (11 boxes in one night clip). I filter it out and do not know why. *Suggestion:* document it, or stop returning it.
- **The deprecation warnings name no replacement.** `RFDETRBase` and `ByteTrack` say when they will be removed, not what to use instead.
- **Training took one extra install and one Mac-only rule.** `rfdetr[train]` (the error is clear), and wrapping the script in `if __name__ == "__main__":` (the error is Python's generic one). *Suggestion:* put both in the quickstart.

## Three questions I'd like to dig into: carts, near-duplicate frames, the edge

- Could the text-prompt model and the fine-tuned model work together, so the carts get found too?
- How should near-duplicate frames from video be handled when labeling footage? In one test, 36 pictures did about as well as 141.
- Would it still be fast enough at the edge, on the small computer that sits next to a camera? I measured about 30 milliseconds per picture on a laptop and have not measured a small device.

## More: scores, pictures, reproduce steps and tips

- [Details and caveats](docs/DETAILS.md): scores, pictures, how the labels were corrected, limits.
- [Reproduce it](docs/REPRODUCE.md): commands and layout.
- [Tips and learnings](docs/LEARNINGS.md): what I'd tell someone trying this.

## Built with RF-DETR, supervision and Grounding DINO

[RF-DETR](https://github.com/roboflow/rf-detr), [supervision](https://github.com/roboflow/supervision), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) (IDEA Research, via Hugging Face), PyTorch, OpenCV, matplotlib.

## Footage credits: five Pexels clips and their creators

All footage is from [Pexels](https://www.pexels.com) under the Pexels License. Everything shown here is altered, and the original videos are not in this repo.

- Dhaka street ("Suhrawardy Udyan TSC") by Faisal Ibne Kalam: [video](https://www.pexels.com/video/suhrawardy-udyan-tsc-26689635/), [profile](https://www.pexels.com/@faisal-ibne-kalam-774996459/)
- e-rickshaws ("Bustling Indian Street with Auto Rickshaws") by md Jahangir alam, filmed in India, not Dhaka: [video](https://www.pexels.com/video/bustling-indian-street-with-auto-rickshaws-38248681/), [profile](https://www.pexels.com/@mdjahangir/)
- Rickshaw street ("Colorful Rickshaws on Bustling Street") by Somogro Bangladesh: [video](https://www.pexels.com/video/colorful-rickshaws-on-bustling-street-36526831/), [profile](https://www.pexels.com/@somogrobangladesh/)
- Night traffic ("Vibrant City Night Traffic Scene") by Jubayer Hossain, tagged Dhaka and Chittagong: [video](https://www.pexels.com/video/vibrant-city-night-traffic-scene-35041521/), [profile](https://www.pexels.com/@jubayer-wh/)
- Rainy walk ("Rainy Day Street Scene in Dhaka, Bangladesh") by Latiful Jawad, labels only: [video](https://www.pexels.com/video/rainy-day-street-scene-in-dhaka-bangladesh-29662763/), [profile](https://www.pexels.com/@latiful-jawad-431220084/)

Created by J. Yousuf.
