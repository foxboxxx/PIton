"""Search compile-time constants of a suite case on the Pi (backlog G1).

Each value of the swept constant is one boot, best of --repeats launches,
outputs checked. For constants that set how many elements a program handles
(BLOCK, BLOCK_SIZE) the 1-D grid is rescaled so the problem stays the same.

    .venv/bin/python perf/tune.py copy BLOCK=256,512,1024,2048
    .venv/bin/python perf/tune.py mandel_32x16 UNROLL=1,2,4
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import date
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf.profile_pi import RESULTS, run_case  # noqa: E402
from perf.suite import CASES_BY_NAME, constants_for  # noqa: E402
from piton.runtime.pi import PiRunner  # noqa: E402

PER_PROGRAM = ("BLOCK", "BLOCK_SIZE")


def variant(case, name: str, value: int):
    old = constants_for(case, "pi")[name]
    make = case.make

    def remade(size):
        run = make(size)
        if name in PER_PROGRAM and len(run.grid) == 1:
            total = run.grid[0] * old
            if total % value:
                raise ValueError(f"{name}={value} does not divide the problem ({total})")
            run = dataclasses.replace(run, grid=(total // value,))
        return run
    return dataclasses.replace(case, constants={**case.constants, name: value}, make=remade)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("sweep", help="NAME=v1,v2,...")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    case = CASES_BY_NAME[args.case]
    name, values = args.sweep.split("=")
    runner = PiRunner(timeout=240)
    rows = []
    for v in (int(x) for x in values.split(",")):
        try:
            r = run_case(variant(case, name, v), args.repeats, runner)
            rows.append({name: v, **r})
            print(f"{name}={v}: {r['best_us']} us  instr {r['instructions']}  progs {r['programs']}  "
                  f"busy {100 * r['busy']:.1f}%  {'ok' if not r['errors'] else 'WRONG'}", flush=True)
        except Exception as ex:
            rows.append({name: v, "failed": f"{type(ex).__name__}: {str(ex).splitlines()[0][:200]}"})
            print(f"{name}={v}: FAILED {rows[-1]['failed']}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"tune-{date.today().isoformat()}-{args.case}-{name}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
