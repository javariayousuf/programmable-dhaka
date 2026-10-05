# Experiment log: round 2 (October 3 to 5, 2026)

What I tried after the first round, what happened, and why I changed direction. All numbers come from the files in `eval/`. Every score is one training run, and training the same setup twice has moved the rickshaw count by about 4 before, so differences of a few points are noise.

## 1. Giving trucks and buses a name (a truck class)

The first round never gave trucks a label, and the model sometimes called them rickshaws. I added a `truck` class (buses included), corrected the first-pass boxes by hand, and trained with and without it. I held out one clip each time: rain, then night.

| Held-out clip | Rickshaw F1 without truck class | with truck class | Trucks found |
|---|---|---|---|
| rain | 0.61 | 0.57 | 12 of 34 |
| night | 0.13 | 0.23 | 5 of 131 |

The truck class learned daytime trucks a little and night trucks almost not at all. It did not clearly help rickshaws.

**A correction to the first round.** Reviewing the truck boxes showed that 12 boxes in the rickshaw-street answer key (clip 3) were really trucks, so that clip has 152 rickshaws, not 163. The night and rain clips had 4 and 1. The Test 1 numbers in the README were scored against the old key and need rescoring.

## 2. A fresh test on a new scene

I added a clip the model never trained on, a roundabout full of pedal rickshaws (165 rickshaws after review), as a new held-out test.

| Model | Rickshaws found | False alarms | Rickshaw F1 |
|---|---|---|---|
| Trained on clips 1, 2, 3, night, no truck class | 109 | 147 | 0.52 |
| Same, with truck class | 72 | 53 | 0.50 |
| Round 1 final model | 66 | 237 | 0.28 |

The truck class cut false alarms but lost real rickshaws, so the score barely moved. Its truck score of 0.90 is mostly one red bus that appears in every still, so it is closer to one vehicle than a hundred.

## 3. Adding a clip of an intersection

The intersection clip adds 319 truck and bus boxes and 271 cars, but only 22 rickshaws. Rickshaw F1 went from 0.50 to 0.54 on the roundabout and from 0.57 to 0.54 on rain. That is noise. Truck and motorcycle scores got a little worse.

## 4. What this changed

Clips with few rickshaws do not teach the model about rickshaws, and cleaning up cars and trucks only trims false alarms. A crowded rickshaw clip then showed a second problem: my review page only shows vehicles the model already found, so rickshaws it missed never get a box. In a packed scene that left dozens of unlabeled rickshaws in the background, and training on those would teach the model to ignore them.

The new plan is to add clearer footage, with rickshaws large in the frame and fewer of them, and still photos, which are quick to label completely. I removed the crowded clip from the project rather than train or test on labels I knew were incomplete.

## Still open

- Rescore Test 1 on the corrected answer key.
- Check how much of the score noise comes from training randomness, by repeating a run.
- Try Roboflow's own tools for finding where a model fails: model evaluation with a confusion matrix, vector analysis, and dataset health checks.
