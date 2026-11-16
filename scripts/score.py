"""Score one or more prediction files produced by the Colab notebooks.

    .venv/bin/python scripts/score.py results/single_test__Qwen2.5-VL-3B-Instruct.json

Each file is the test split with a "predict" field added to every row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viinfo.metrics import format_report, score  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = sum("predict" not in r for r in rows)
        print(f"\n== {path} ({len(rows)} rows, {missing} without a prediction)")
        print(format_report(score(rows)))


if __name__ == "__main__":
    main()
