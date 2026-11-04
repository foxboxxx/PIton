"""Compare two Pi profiles (perf/results/pi-*.json from profile_pi.py) case by case.

    .venv/bin/python perf/compare.py pi-2026-11-03-baseline.json pi-2026-11-04.json
"""
from __future__ import annotations

import json
import math
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(path: str) -> dict:
    if not os.path.exists(path):
        path = os.path.join(RESULTS, path)
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    before, after = load(argv[0]), load(argv[1])
    print(f"{'case':16} {'before_us':>10} {'after_us':>10} {'speedup':>8}  {'busy':>13}  check")
    logs = []
    for name, b in before.items():
        a = after.get(name)
        if a is None or "best_us" not in a or "best_us" not in b:
            print(f"{name:16} {'missing or failed':>30}")
            continue
        s = b["best_us"] / a["best_us"]
        logs.append(math.log(s))
        print(f"{name:16} {b['best_us']:>10} {a['best_us']:>10} {s:>7.3f}x  "
              f"{100 * b['busy']:5.1f}->{100 * a['busy']:5.1f}%  {'ok' if not a.get('errors') else 'WRONG'}")
    if logs:
        print(f"geometric mean speedup over {len(logs)} cases: {math.exp(sum(logs) / len(logs)):.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
