"""Is long straight-line code bound by instruction fetch? exp at several code sizes.

Kernels of 800-4300 straight-line instructions (exp, log, sqrt, softmax,
idiv) miss the QPU instruction cache on ~3.06% of fetches, about one miss per
32 instructions, while loop kernels miss <0.3%. The stall cycles are not
counted as busy, so they may explain why exp is only ~40% busy. Each variant
computes exp over 262144 floats:

  exp_256      BLOCK=256 unrolled (the suite case)
  exp_128      BLOCK=128 unrolled, twice the programs
  exp_loop64   256 elements per program as a runtime loop over 64-element pieces
  exp_loop32   the same with 32-element pieces

    .venv/bin/python perf/icache_experiment.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402
from perf.suite import PTR, VEC, Case, Inputs, Run  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402


@triton.jit
def exp_loop_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, SUB: tl.constexpr):
    base = tl.program_id(0) * BLOCK
    for i in range(BLOCK // SUB):
        offs = base + i * SUB + tl.arange(0, SUB)
        m = offs < n
        tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=m, other=1.0)), mask=m)


def _case(name, kernel, constants, block):
    def make(size):
        n = VEC[size]
        g = Inputs()
        x = g.floats("x_ptr", n, -8.0, 8.0).astype(np.float64)
        g.output("out_ptr", n)
        g.scalar("n", n)
        return Run(g, (n // block,), {"out_ptr": np.exp(x)}, 4e-6, 1e-30)
    return Case(name, kernel, {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, constants, make)


CASES = [_case("exp_256", sfu_kernel, {"OP": 4, "BLOCK": 256}, 256),
         _case("exp_128", sfu_kernel, {"OP": 4, "BLOCK": 128}, 128),
         _case("exp_loop64", exp_loop_kernel, {"BLOCK": 256, "SUB": 64}, 256),
         _case("exp_loop32", exp_loop_kernel, {"BLOCK": 256, "SUB": 32}, 256)]


def main():
    from perf.profile_pi import run_case
    from piton.runtime.pi import PiRunner
    runner = PiRunner(timeout=300)
    print(f"{'variant':11} {'instr':>6} {'progs':>6} {'best_us':>8} {'busy':>6} {'icache_miss':>11}  check")
    for case in CASES:
        r = run_case(case, 5, runner)
        print(f"{case.name:11} {r['instructions']:6} {r['programs']:6} {r['best_us']:8} {100 * r['busy']:5.1f}% "
              f"{100 * r['icache_miss']:10.2f}%  {'ok' if not r['errors'] else r['errors']}", flush=True)


if __name__ == "__main__":
    main()
