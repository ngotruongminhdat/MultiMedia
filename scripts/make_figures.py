"""Draw the paper's figures into paper/figs/ (PDF, sized for one IEEE column).

    .venv/bin/python scripts/make_figures.py data/external/ocr_boxes_test_canvas2560.json

Figures: fig_dataset (what the test questions ask about), fig_resolution (image
size vs. the model's pixel budget), fig_locator (answer inside located vs. random
crops), fig_example (one question with its located crops).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from locator_stats import CROP_FRAC, FULL_PIXELS, N_CROPS, load, locator_quality  # noqa: E402
from viinfo.locate import locate  # noqa: E402

# Categorical slots 1-2 of the validated palette (blue, orange); text wears ink, not series colour.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.45  # IEEE column width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.pad_inches": 0.02,
})

TOPIC_EN = {
    "Kinh tế - Hội nhập": "Economy", "Y tế - Cộng đồng": "Health", "Văn hóa - Xã hội": "Culture & society",
    "Dự báo thời tiết": "Weather forecast", "Thể thao - Nghệ thuật": "Sports & arts",
    "Thiên tai - Tai nạn": "Disasters & accidents", "Giáo dục - Khoa Học": "Education & science",
    "Nhân vật - Sự kiện": "People & events", "Quốc phòng - An ninh": "Defence & security",
}
TYPE_LABEL = {"all": "All questions", "image-span": "Image-span", "question-span": "Question-span",
              "multi-span": "Multi-span", "non-extractive": "Non-extractive"}


def hbars(ax, labels, values, color=BLUE, fmt="{:d}", height=0.62):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=height, color=color, linewidth=0)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", ha="left", color=INK, fontsize=7)
    ax.set_xlim(0, max(values) * 1.16)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)


def fig_dataset(all_rows, out):
    elements = Counter(e for r in all_rows for e in (r["element"] if isinstance(r["element"], list) else [r["element"]]))
    topics = Counter(TOPIC_EN[r["image_type"]] for r in all_rows)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.05), gridspec_kw={"height_ratios": [6, 9], "hspace": 0.32})
    for ax, counter, title in ((a, elements, "Element holding the evidence"), (b, topics, "Topic of the infographic")):
        items = counter.most_common()
        hbars(ax, [k for k, _ in items], [v for _, v in items])
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_resolution(all_rows, sizes, out):
    images = sorted({r["image_path"] for r in all_rows} & set(sizes))
    mp = np.array([sizes[p][0] * sizes[p][1] / 1e6 for p in images])
    budget = FULL_PIXELS / 1e6
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    bins = np.logspace(np.log10(0.4), np.log10(32), 20)
    ax.hist(mp, bins=bins, color=BLUE, linewidth=0.6, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel("Image size (megapixels, log scale)")
    ax.set_ylabel("Test images")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=MUTED)
    top = ax.get_ylim()[1] * 1.42  # headroom above the bars for the two annotations
    ax.set_ylim(0, top)
    ax.set_yticks([0, 10, 20, 30])
    ax.axvline(budget, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(budget * 1.08, top * 0.98, "model budget:\n1,280 tokens, about 1 MP", va="top", ha="left", fontsize=7, color=INK)
    share = (mp > budget).mean()
    ax.text(0.99, 0.98, f"{share:.0%} of images are downscaled\nmedian {statistics.median(mp):.1f} MP",
            transform=ax.transAxes, va="top", ha="right", fontsize=7, color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_locator(stat, n_images, out):
    keys = ["all", "image-span", "question-span", "multi-span", "non-extractive"]
    located = [100 * stat[k]["located"] / max(stat[k]["readable"], 1) for k in keys]
    rand = [100 * stat[k]["random"] / max(stat[k]["readable"], 1) for k in keys]
    labels = [f"{TYPE_LABEL[k]}\n(n = {int(stat[k]['readable'])})" for k in keys]
    y = np.arange(len(keys))[::-1] * 1.0
    h = 0.30
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    ax.barh(y + h / 2 + 0.02, located, height=h, color=BLUE, linewidth=0, label="OCR-located crops")
    ax.barh(y - h / 2 - 0.02, rand, height=h, color=ORANGE, linewidth=0, label="Random crops (same size)")
    for yi, a, b in zip(y, located, rand):
        ax.text(a + 1.2, yi + h / 2 + 0.02, f"{a:.0f}%", va="center", ha="left", fontsize=7, color=INK)
        ax.text(b + 1.2, yi - h / 2 - 0.02, f"{b:.0f}%", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Questions whose answer lies inside the crops (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=2, frameon=False, fontsize=7,
              handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_example(rows, boxes, sizes, idf, image_dir, out, image="images/8513.jpg", needle="bồ đào nha"):
    row = next((r for r in rows if r["image_path"] == image and needle in r["question"]), None)
    if row is None or not (image_dir / image).exists():
        print("  fig_example skipped: example image or question not available")
        return
    wins = locate(row["question"], boxes[image], sizes[image], idf, n_crops=N_CROPS, crop_frac=CROP_FRAC)
    full = Image.open(image_dir / image).convert("RGB")
    scale = 900 / max(full.size)
    small = full.resize((round(full.width * scale), round(full.height * scale)), Image.LANCZOS)
    fig = plt.figure(figsize=(COL_W, 2.3))
    gs = fig.add_gridspec(len(wins) or 1, 2, width_ratios=[1.25, 1], wspace=0.06, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(small)
    for i, w in enumerate(wins):
        color = (BLUE, ORANGE)[i % 2]
        ax.add_patch(Rectangle((w[0] * scale, w[1] * scale), (w[2] - w[0]) * scale, (w[3] - w[1]) * scale,
                               fill=False, edgecolor=color, linewidth=1.4))
        ax.text(w[0] * scale + 8, w[1] * scale + 8, str(i + 1), va="top", ha="left", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none"))
    ax.set_axis_off()
    for i, w in enumerate(wins):
        cax = fig.add_subplot(gs[i, 1])
        crop = full.crop(w)
        crop.thumbnail((700, 700), Image.LANCZOS)
        cax.imshow(crop)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(True); sp.set_edgecolor((BLUE, ORANGE)[i % 2]); sp.set_linewidth(1.2)
        cax.set_ylabel(f"crop {i + 1}", fontsize=7, color=INK, labelpad=2)
    fig.savefig(out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"  example question: {row['question']}  ->  {row['answer']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", type=Path)
    ap.add_argument("--split", type=Path, default=ROOT / "data/external/ViInfographicVQA/data/single_test.json")
    ap.add_argument("--images", type=Path, default=ROOT / "data/external/ViInfographicVQA")
    ap.add_argument("--out", type=Path, default=ROOT / "paper/figs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, all_rows, boxes, sizes, idf = load(args.boxes, args.split)
    n_images = len({r["image_path"] for r in rows})
    print(f"{len(rows)} of {len(all_rows)} questions have OCR ({n_images} images)")
    fig_dataset(all_rows, args.out / "fig_dataset.pdf")
    fig_resolution(all_rows, sizes, args.out / "fig_resolution.pdf")
    fig_locator(locator_quality(rows, boxes, sizes, idf), n_images, args.out / "fig_locator.pdf")
    fig_example(rows, boxes, sizes, idf, args.images, args.out / "fig_example.pdf")
    print("wrote", sorted(p.name for p in args.out.glob("*.pdf")))


if __name__ == "__main__":
    main()
