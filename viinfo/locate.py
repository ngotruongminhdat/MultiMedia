"""Use OCR as a locator: find where on an infographic a question points.

OCR text on Vietnamese infographics is noisy (wrong diacritics, wrong digits),
so instead of handing the text to the model we only use it to decide which
regions to crop from the full-resolution image. Matching ignores diacritics and
tolerates small spelling errors, which is all that locating needs.

An OCR box is `[x0, y0, x1, y1, text, confidence]` in original-image pixels.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import Levenshtein

TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")
FUZZY_MIN = 0.8      # minimum similarity for an inexact word match
MIN_WINDOW_SCORE = 1.0  # windows scoring below this are not worth cropping


def fold(text: str) -> str:
    """Lowercase and strip Vietnamese diacritics: 'Bồ Đào Nha' -> 'bo dao nha'."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(fold(text))


def build_idf(questions: list[str]) -> dict[str, float]:
    """Inverse document frequency over the questions: 'bao nhiêu', 'theo', ... get ~0 weight."""
    df = Counter(t for q in questions for t in set(tokens(q)))
    n = len(questions)
    return {t: math.log((n + 1) / (c + 1)) for t, c in df.items()}


def _match(word: str, box_words: set[str]) -> float:
    if word in box_words:
        return 1.0
    if len(word) < 4 or word[0].isdigit():  # short words and numbers must match exactly
        return 0.0
    best = max((Levenshtein.ratio(word, w) for w in box_words), default=0.0)
    return best if best >= FUZZY_MIN else 0.0


def score_boxes(question: str, boxes: list, idf: dict[str, float]) -> list[float]:
    """Relevance of each OCR box to the question.

    A matched word counts more when it is rare among questions (not a stop word)
    and rare among this image's boxes (it pins down one place, unlike a word
    repeated all over the image). The sum is then scaled by how much of the box
    the question covers, so a short label fully named in the question
    ("HÀ NỘI") beats a long sentence that happens to share a few words.
    """
    default = max(idf.values(), default=1.0)  # unseen words are rare, hence informative
    box_words = [set(tokens(b[4])) for b in boxes]
    q_words = set(tokens(question))
    weight = {}
    for t in q_words:
        spread = sum(_match(t, words) > 0 for words in box_words)
        if spread:
            weight[t] = idf.get(t, default) / default * math.log(1 + len(boxes) / spread)
    scores = []
    for words in box_words:
        matched = sum(w * _match(t, words) for t, w in weight.items())
        covered = sum(_match(t, q_words) > 0 for t in words) / len(words) if words else 0.0
        scores.append(matched * (0.5 + covered))
    return scores


def _window(cx: float, cy: float, w: int, h: int, crop_frac: float) -> tuple[int, int, int, int]:
    cw, ch = w * crop_frac, h * crop_frac
    x0 = min(max(cx - cw / 2, 0), w - cw)
    y0 = min(max(cy - ch / 2, 0), h - ch)
    return round(x0), round(y0), round(x0 + cw), round(y0 + ch)


def _iou(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def boxes_in(window, boxes: list) -> list:
    """OCR boxes whose centre lies inside the window."""
    x0, y0, x1, y1 = window
    return [b for b in boxes if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]


def locate(question: str, boxes: list, size: tuple[int, int], idf: dict[str, float],
           n_crops: int = 2, crop_frac: float = 0.35, max_iou: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Up to `n_crops` windows (x0, y0, x1, y1) covering the text most related to the question.

    Each window spans `crop_frac` of the image width and height. A candidate
    window is centred on every matching box and scored by the matches it
    contains; windows are then picked greedily, skipping heavy overlaps.
    Returns [] when nothing in the image matches the question.
    """
    w, h = size
    scores = score_boxes(question, boxes, idf)
    hits = [(s, b) for s, b in zip(scores, boxes) if s > 0]
    candidates = []
    for _, anchor in hits:
        win = _window((anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2, w, h, crop_frac)
        inside = sum(s for s, b in hits
                     if win[0] <= (b[0] + b[2]) / 2 <= win[2] and win[1] <= (b[1] + b[3]) / 2 <= win[3])
        candidates.append((inside, win))
    candidates.sort(key=lambda c: -c[0])
    chosen: list[tuple[int, int, int, int]] = []
    for total, win in candidates:
        if total < MIN_WINDOW_SCORE or len(chosen) == n_crops:
            break
        if all(_iou(win, c) <= max_iou for c in chosen):
            chosen.append(win)
    return chosen


def answer_coverage(answer: str, boxes: list) -> float:
    """Share of the gold answer's words that OCR found in `boxes` (diagnostic only, never used to answer)."""
    want = set(tokens(answer))
    if not want:
        return 0.0
    have = {t for b in boxes for t in tokens(b[4])}
    return sum(_match(t, have) > 0 for t in want) / len(want)
