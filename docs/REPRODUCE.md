# Reproduce it

The original videos and still frames are not in this repo. The Pexels License restricts selling or re-uploading unaltered copies, and the stills show real people's faces. The label files are here, and the pictures and videos in `media/` have heads blurred by the model.

```bash
pip install -r requirements.txt "rfdetr[train]" matplotlib
```

1. Download the clips from Pexels (links in the README), HD or Full HD.
2. First-pass labels with the original pipeline: `python scripts/prelabel.py clip.mp4 /tmp/out 12`. The number is how many video frames between pictures. Use about 0.2 seconds, so 12 for a 60 fps clip and 6 for 30 fps. Copy `/tmp/out/images` to `labels/<clip>/images`.
3. `python scripts/make_train_dataset.py clip2,clip3,night,rain` builds a training set from those label folders. Then `python scripts/train.py small 25`.
4. `python scripts/evaluate.py clip1` scores the model on a clip that is not in the training set.
5. New clips: `python scripts/prelabel_finetuned.py clip.mp4 out_dir 0.2` makes first-pass labels with the fine-tuned model, `python scripts/review_cards.py out_dir` opens the review page, and `python scripts/apply_review.py out_dir` writes your answers back.
6. `python scripts/detect3.py clip.mp4 out.mp4 "Video by <creator> on Pexels" "pexels.com/@<handle>"` renders a video. `make_figures.py` and `make_charts.py` rebuild the pictures.

`detect3.py`, `make_figures.py` and `make_charts.py` use Futura from macOS by default. Set `FONT_PATH` to any `.ttf` on another system.

```
labels/<clip>/annotations.coco.json         corrected labels (training data, and the answer key for a test clip)
labels/<clip>/original_pipeline.coco.json   the first-pass guesses, for before and after
scripts/                                    label, review, train, score, render, charts
eval/                                       scores as JSON, including the first run of Test 1
media/                                      pictures, charts, videos
```
