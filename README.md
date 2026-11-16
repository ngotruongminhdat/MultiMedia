# Vietnamese infographic question answering with small vision-language models

Final project for the Multimedia Systems course (group Pi-Cap).

Reference paper: [ViInfographicVQA](https://arxiv.org/abs/2512.12424) (AI4Research @ AAAI 2026) —
[code](https://github.com/duongtruongbinh/ViInfographicVQA),
[dataset](https://huggingface.co/datasets/duytranus/ViInfographicVQA).

## Numbers to beat (paper, ANLS %)

| Model | Single-image | Multi-image |
|---|---|---|
| Qwen2.5-VL-7B, zero-shot | 67.42 | 54.92 |
| Qwen2.5-VL-7B, fine-tuned (combined) | 68.53 | 55.62 |
| InternVL3.5-8B | 67.02 | 35.50 |
| Ovis2.5-9B | 71.02 | – |

## Running

Models run on Google Colab (T4 GPU); scoring and analysis run locally.

1. Open `notebooks/01_baseline_zero_shot.ipynb` in Colab and run all cells.
   It downloads the test images itself and saves predictions to
   `MyDrive/viinfographicvqa/results/`.
2. Open `notebooks/02_ocr_augmented_prompt.ipynb` with the same `MODEL_ID`.
   It runs Vietnamese OCR (EasyOCR) on each image, adds the text to the prompt,
   and its last cell compares the result with the baseline from step 1.
   Set `MODE = "ocr_only"` for the ablation without the image.
3. Open `notebooks/03_answer_format_prompt.ipynb` with the same `MODEL_ID`.
   It replaces the paper's "single term" prompt with explicit answer-format rules
   derived from the training answers. `N_FEWSHOT = 8` adds training examples,
   `USE_OCR = True` adds the OCR text cached by step 2. Its last cell prints one
   table comparing every run saved for the model.
4. Open `notebooks/04_ocr_locator_zoom.ipynb` with the same `MODEL_ID`.
   It uses OCR only to find where the question points, crops those regions from
   the full-resolution image, and gives the model the whole image plus the crops.
   `CROP_MODE = "random"` and `"grid"` are the control runs.
5. Copy the prediction files into `results/` and score them:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/score.py results/single_test__Qwen2.5-VL-3B-Instruct.json
```

## Code

- `viinfo/metrics.py` — scorer; `anls` matches the paper's official scorer exactly
  (checked on all 1,374 test rows), with per answer-type / element / topic breakdowns
- `viinfo/locate.py` — OCR-as-locator: match a question to OCR boxes and pick crop windows
- `scripts/score.py` — score prediction files
- `notebooks/` — Colab notebooks

The dataset has no licence and its images belong to the Vietnam News Agency:
do not commit or redistribute them (`data/` is git-ignored).
