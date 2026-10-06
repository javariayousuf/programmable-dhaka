# Experiment log: round 2 (October 3 to 6, 2026)

What I tried after the first round, what happened, and why I changed direction. Every number comes from the files in `eval/`. Each score is one training run, and training the same setup twice has moved the rickshaw count by about 4 before, so differences of a few points are noise. Clips are numbered as in the README table.

## The short version

- Giving trucks and buses their own class did not clearly help rickshaws.
- My review page hid missing boxes. On one clip it hid half the rickshaws, so some earlier scores rested on incomplete answer keys.
- After fixing that, the one change that clearly helped was adding a single clear, fully labeled rickshaw clip (clip 7). Rickshaw F1 on a new scene (clip 8) went from 0.48 to 0.64.

## 1. A truck class

The first round never gave trucks a label, and the model sometimes called them rickshaws. I added a `truck` class (buses included), corrected the first-pass boxes by hand, and trained with and without it, holding out one clip each time.

| Held-out clip | Rickshaw F1 without truck class | with truck class | Trucks found |
|---|---|---|---|
| 5 (rain) | 0.61 | 0.57 | 12 of 34 |
| 4 (night) | 0.13 | 0.23 | 4 of 131 |

The truck class learned daytime trucks a little and night trucks almost not at all. It did not clearly help rickshaws.

**A correction to round 1.** Reviewing the truck boxes showed that 12 boxes in the clip 3 answer key were really trucks, so that clip has 152 rickshaws, not 163. Clips 4 and 5 had 4 and 1.

## 2. A fresh test clip, and a problem with my answer keys

Clip 8 is a roundabout full of decorated pedal rickshaws. The model never trains on it. I labeled it with my review page, and then drew in the boxes the model never found. That added 164 rickshaws and took the count from 165 to 328. My review page can only judge boxes that already exist, so a rickshaw the model missed never got a box. The keys for clips 1 and 3, which Test 1 and Test 2 in the README use, were made the same way and I have not rechecked them.

I also tried a crowded rickshaw clip. Dozens of background rickshaws never got boxes, and training on that would teach the model to ignore them, so I removed it.

## 3. Scores on clip 8, with the corrected key (328 rickshaws, cutoff 0.5)

| Model | Rickshaws found | False alarms | Rickshaw F1 |
|---|---|---|---|
| Clips 1, 2, 3, 4, no truck class | 168 | 88 | 0.57 |
| Same, with truck class | 106 | 19 | 0.47 |
| + clip 6 (intersection, 22 rickshaws) | 108 | 17 | 0.48 |
| + clip 7 (quiet street, 143 rickshaws) | 160 | 14 | 0.64 |
| Round 1 final model | 66 | 237 | 0.21 |

On clip 5 (rain), rickshaw F1 went from 0.57 (truck class) to 0.54 (+ clip 6) to 0.65 (+ clip 7).

Adding the intersection clip, which has lots of trucks, buses and cars but few rickshaws, did nothing for rickshaws. Adding the quiet street, with every rickshaw labeled, moved both held-out clips by more than the noise.

The main weakness on a new scene is missed rickshaws, not false alarms. Even the best model finds about half of them.

## 4. The confidence cutoff

All scores above use a 0.5 cutoff. Lowering it finds more rickshaws:

| Model | Cutoff | Found | False alarms | F1 |
|---|---|---|---|---|
| + clip 6 | 0.5 | 108 | 17 | 0.48 |
| + clip 6 | 0.3 | 183 | 106 | 0.59 |
| + clip 7 | 0.5 | 160 | 14 | 0.64 |
| + clip 7 | 0.3 | 211 | 76 | 0.69 |

I looked at this test clip to see the effect, so I would not call 0.69 the result. A fair version picks the cutoff on one clip and applies it to another.

## 5. What changed

Clips with few rickshaws do not teach the model about rickshaws, and a truck class mostly trims false alarms. What helped was a clear clip, a steady camera, big rickshaws and every one labeled. I also now draw the missing boxes before I trust a clip, either for training or as an answer key.

## Still open

- Redraw the missing boxes in clips 1 and 3, then rescore Test 1 and Test 2.
- Repeat a run to see how much of the difference is training randomness.
- Add clear rickshaw clips and photos, one kind at a time, and keep a test clip the model never sees.
- Try Roboflow's own tools for finding where a model fails: model evaluation with a confusion matrix, vector analysis, and dataset health checks.
- A highway clip with many trucks is labeled by the model only and not reviewed, so it is not used and not included.
