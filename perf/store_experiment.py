"""Where does an elementwise program's time go? Variants of `copy`, on the Pi.

The V3D counters account for only 20-30% of the QPU cycles of copy/add/mul
(busy + TMU stall + idle), so the rest is spent stalled on something they
do not classify. Each variant removes one part of the program:

  empty        no loads or stores: launch cost per program
  store_only   values computed in registers, stored: mutex + VDW + vw_wait
  load_only    loads, reduced to 16 values per program: the TMU path
  copy         both (the suite's copy kernel)
  copy_nomutex copy compiled with store_mutex=False (experiment only; the VDW
               stride register is shared, so outputs are checked as usual)
  copy_512     512 elements per program, half as many programs

    .venv/bin/python perf/store_experiment.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402
from perf.suite import PTR, VEC, Case, Inputs, Run  # noqa: E402
from primitive_kernels import copy_kernel  # noqa: E402


@triton.jit
def empty_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < 0:                         # never true: keeps out_ptr an argument
        tl.store(out_ptr + tl.arange(0, 16), tl.zeros([16], tl.float32))


@triton.jit
def store_only_kernel(out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, offs.to(tl.float32), mask=m)


@triton.jit
def load_only_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    part = tl.sum(tl.reshape(x, (BLOCK // 16, 16)), axis=0)
    tl.store(out_ptr + tl.program_id(0) * 16 + tl.arange(0, 16), part)


def _copy(name, block, options=None):
    def make(size):
        n = VEC[size]
        g = Inputs()
        x = g.floats("x_ptr", n).astype(np.float64)
        g.output("out_ptr", n)
        g.scalar("n", n)
        return Run(g, (n // block,), {"out_ptr": x}, 0, 0)
    return Case(name, copy_kernel, {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"BLOCK": block}, make,
                options=dict(options or {}))


def _empty():
    def make(size):
        g = Inputs()
        g.output("out_ptr", 16)
        g.scalar("n", VEC[size])
        return Run(g, (VEC[size] // 256,), {})
    return Case("empty", empty_kernel, {"out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, make)


def _store_only():
    def make(size):
        n = VEC[size]
        g = Inputs()
        g.output("out_ptr", n)
        g.scalar("n", n)
        return Run(g, (n // 256,), {"out_ptr": np.arange(n, dtype=np.float64)}, 0, 0)
    return Case("store_only", store_only_kernel, {"out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, make)


def _load_only():
    def make(size):
        n = VEC[size]
        g = Inputs()
        x = g.floats("x_ptr", n).astype(np.float64)
        g.output("out_ptr", (n // 256) * 16)
        g.scalar("n", n)
        expected = x.reshape(-1, 16, 16).sum(axis=1).ravel()
        return Run(g, (n // 256,), {"out_ptr": expected}, 1e-5, 1e-5)
    return Case("load_only", load_only_kernel, {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, make)


CASES = [_empty(), _store_only(), _load_only(), _copy("copy", 256),
         _copy("copy_nomutex", 256, {"store_mutex": False}), _copy("copy_512", 512)]


def main():
    from datetime import date
    import json
    from perf.profile_pi import RESULTS, run_case
    from piton.runtime.pi import PiRunner

    runner = PiRunner(timeout=300)
    rows = []
    for case in CASES:
        print(f"{case.name} ...", file=sys.stderr, flush=True)
        try:
            rows.append(run_case(case, 5, runner))
        except Exception as ex:
            rows.append(dict(name=case.name, failed=f"{type(ex).__name__}: {ex}"))
    print(f"{'variant':13} {'programs':>8} {'best_us':>8} {'us/program':>11} {'issue_us':>9} "
          f"{'busy':>5} {'tmu':>5} {'idle':>5}  check")
    for r in rows:
        if "failed" in r:
            print(f"{r['name']:13} FAILED {r['failed'].splitlines()[0][:70]}")
            continue
        c = r["counters"]
        per_program = 12 * r["best_us"] / r["programs"]       # one QPU's view: programs run 12 at a time
        issue = c["valid"] / 250 / r["programs"]              # cycles executing instructions, per program
        print(f"{r['name']:13} {r['programs']:8} {r['best_us']:8} {per_program:11.2f} {issue:9.2f} "
              f"{100 * r['busy']:4.0f}% {100 * r['tmu']:4.0f}% {100 * r['idle']:4.0f}%  "
              f"{'ok' if not r['errors'] else 'WRONG'}")
    path = os.path.join(RESULTS, f"store-experiment-{date.today().isoformat()}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
