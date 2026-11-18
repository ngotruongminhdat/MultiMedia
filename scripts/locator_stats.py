"""GPU-free statistics for the paper: image resolution, locator quality, token cost.

    .venv/bin/python scripts/locator_stats.py data/external/ocr_boxes_test_canvas2560.json
    .venv/bin/python scripts/locator_stats.py <boxes.json> --tokens     # also count visual tokens (slow)

The boxes file is {"sizes": {image: [w, h]}, "boxes": {image: [[x0, y0, x1, y1, text, conf], ...]}},
as written by the OCR step of notebook 04 (there `sizes` is separate; see --help).

Locator quality uses the gold answers only as a diagnostic: a question counts as
"OCR-readable" when OCR found at least 80% of the answer's words somewhere in the
image, and as "in crops" when it found them inside the chosen windows.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viinfo.locate import _window, answer_coverage, boxes_in, build_idf, locate  # noqa: E402

MIN_CONF = 0.3
N_CROPS, CROP_FRAC = 2, 0.35
FULL_PIXELS, CROP_PIXELS = 1280 * 28 * 28, 768 * 28 * 28
COVERED = 0.8
RANDOM_DRAWS = 20


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def locator_quality(rows, boxes, sizes, idf, n_crops=N_CROPS, crop_frac=CROP_FRAC, seed=0):
    """Per answer type (and "all"): n, readable, located, random (expected), crops, none."""
    rng = random.Random(seed)
    stat = defaultdict(lambda: defaultdict(float))
    for r in rows:
        b, size = boxes[r["image_path"]], sizes[r["image_path"]]
        wins = locate(r["question"], b, size, idf, n_crops=n_crops, crop_frac=crop_frac)
        readable = answer_coverage(r["answer"], b) >= COVERED
        located = readable and answer_coverage(r["answer"], [x for w in wins for x in boxes_in(w, b)]) >= COVERED
        rand = 0.0
        if readable:
            for _ in range(RANDOM_DRAWS):
                rw = [_window(rng.uniform(0, size[0]), rng.uniform(0, size[1]), size[0], size[1], crop_frac)
                      for _ in range(n_crops)]
                rand += (answer_coverage(r["answer"], [x for w in rw for x in boxes_in(w, b)]) >= COVERED) / RANDOM_DRAWS
        for key in ("all", r["answer_source"]):
            s = stat[key]
            s["n"] += 1
            s["readable"] += readable
            s["located"] += located
            s["random"] += rand
            s["crops"] += len(wins)
            s["none"] += not wins
    return stat


def load(boxes_path: Path, split_path: Path):
    """Return (rows with OCR, all rows, boxes above MIN_CONF, sizes, idf)."""
    data = json.loads(boxes_path.read_text(encoding="utf-8"))
    sizes = {k: tuple(v) for k, v in data["sizes"].items()}
    boxes = {k: [b for b in v if b[5] >= MIN_CONF] for k, v in data["boxes"].items()}
    all_rows = json.loads(split_path.read_text(encoding="utf-8"))
    rows = [r for r in all_rows if r["image_path"] in boxes]
    return rows, all_rows, boxes, sizes, build_idf([r["question"] for r in all_rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", type=Path)
    ap.add_argument("--split", type=Path, default=ROOT / "data/external/ViInfographicVQA/data/single_test.json")
    ap.add_argument("--images", type=Path, default=ROOT / "data/external/ViInfographicVQA")
    ap.add_argument("--tokens", action="store_true", help="count visual tokens with the Qwen2.5-VL preprocessor")
    args = ap.parse_args()

    rows, all_rows, boxes, sizes, idf = load(args.boxes, args.split)
    images = sorted({r["image_path"] for r in rows})
    print(f"{len(rows)} questions on {len(images)} images (of {len(all_rows)} questions in the split)\n")

    # --- resolution -------------------------------------------------------
    sized = sorted({r["image_path"] for r in all_rows} & set(sizes))  # resolution needs no OCR
    widths = [sizes[p][0] for p in sized]
    mps = [sizes[p][0] * sizes[p][1] / 1e6 for p in sized]
    down = [max(1.0, (sizes[p][0] * sizes[p][1] / FULL_PIXELS) ** 0.5) for p in sized]
    zoom = []
    for p in sized:
        w, h = sizes[p]
        full_scale = min(1.0, (FULL_PIXELS / (w * h)) ** 0.5)
        crop_scale = min(1.0, (CROP_PIXELS / (w * h * CROP_FRAC ** 2)) ** 0.5)
        zoom.append(crop_scale / full_scale)
    print(f"== resolution ({len(sized)} images)")
    print(f"width px: median {statistics.median(widths):.0f}, min {min(widths)}, max {max(widths)}")
    print(f"megapixels: median {statistics.median(mps):.1f}, min {min(mps):.1f}, max {max(mps):.1f}")
    print(f"linear downscale to fit 1280 tokens: median {statistics.median(down):.2f}x, max {max(down):.2f}x; "
          f"images downscaled at all: {pct(sum(d > 1 for d in down) / len(down))}")
    print(f"crop magnification relative to the whole-image view: median {statistics.median(zoom):.2f}x")
    counts = [len(boxes[p]) for p in images]
    print(f"OCR boxes per image: median {statistics.median(counts):.0f}, min {min(counts)}, max {max(counts)}\n")

    # --- locator quality --------------------------------------------------
    stat = locator_quality(rows, boxes, sizes, idf)
    print(f"== locator quality ({N_CROPS} crops, {CROP_FRAC} of width and height each)")
    print(f"{'':16s}{'n':>6s}{'OCR-readable':>14s}{'in located':>12s}{'in random':>11s}{'crops/q':>9s}{'no crop':>9s}")
    for key in ("all", "image-span", "question-span", "multi-span", "non-extractive"):
        s = stat[key]
        r = max(s["readable"], 1)
        print(f"{key:16s}{int(s['n']):6d}{pct(s['readable'] / s['n']):>14s}{pct(s['located'] / r):>12s}"
              f"{pct(s['random'] / r):>11s}{s['crops'] / s['n']:9.2f}{pct(s['none'] / s['n']):>9s}")
    print("(in located / in random are shares of the OCR-readable questions)\n")

    # --- token cost -------------------------------------------------------
    if args.tokens:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=256 * 28 * 28, max_pixels=FULL_PIXELS)
        pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        cache: dict = {}

        def count(row, wins) -> int:
            p = row["image_path"]
            if p not in cache:
                cache.clear()
                cache[p] = Image.open(args.images / p).convert("RGB")
            full = cache[p]
            content = [{"type": "image", "image": full, "min_pixels": 256 * 28 * 28, "max_pixels": FULL_PIXELS}]
            content += [{"type": "image", "image": full.crop(w), "min_pixels": 256 * 28 * 28, "max_pixels": CROP_PIXELS}
                        for w in wins]
            content.append({"type": "text", "text": "q"})
            messages = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(messages)
            ids = proc(text=[text], images=imgs, return_tensors="pt").input_ids
            return int((ids == pad).sum())

        def grid(size):
            w, h = size
            return [(x, y, x + w // 2, y + h // 2) for y in (0, h - h // 2) for x in (0, w - w // 2)]

        modes = {
            "baseline (whole image)": lambda r: [],
            "located crops": lambda r: locate(r["question"], boxes[r["image_path"]], sizes[r["image_path"]], idf,
                                              n_crops=N_CROPS, crop_frac=CROP_FRAC),
            "grid 2x2": lambda r: grid(sizes[r["image_path"]]),
        }
        print("== visual tokens per question")
        for name, pick in modes.items():
            n = sorted(count(r, pick(r)) for r in rows)
            print(f"{name:24s} mean {statistics.mean(n):7.0f}  median {statistics.median(n):6.0f}  max {n[-1]:6d}")


if __name__ == "__main__":
    main()
