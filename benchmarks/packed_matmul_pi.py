"""Measure the packed-B experiment, with input preparation timed separately.

    PITON_HW=1 .venv/bin/python benchmarks/packed_matmul_pi.py 512 1024 --bk 8

GPU times exclude packed input generation. PITON_PREP measures B generation
and placement directly into packed memory; it is not a standalone transpose
measurement or an end-to-end speedup. Reusing a packed B can amortize its cost.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
import piton.compiler as pc  # noqa: E402
from piton.runtime.launcher import program_ids  # noqa: E402
from piton.runtime.pi import Buffer, PiRunner, lcg_floats  # noqa: E402
from packed_matmul import matmul_packed_b_kernel  # noqa: E402


def bench(n, bm=8, bn=64, bk=8, group=4, stages=2, repeats=3, sample_stride=97,
          seeds=(12345, 67890), specialize=False, verbose=False):
    if any(v <= 0 or v & (v - 1) for v in (bm, bn, bk)) or group <= 0:
        raise ValueError("BM, BN and BK must be positive powers of two; group must be positive")
    if n <= 0 or n % bn:
        raise ValueError("N must be positive and divisible by BN for the packed layout")
    if repeats <= 0 or sample_stride <= 0:
        raise ValueError("repeats and sample_stride must be positive")
    sig = {name: "*fp32" for name in ("a_ptr", "b_ptr", "c_ptr")}
    scalars = {name: n for name in ("M", "N", "stride_am", "stride_cm")}
    cons = dict(K=n, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=group)
    if specialize:
        cons.update(scalars)
    else:
        sig.update({name: "i32" for name in scalars})
    grid = (piton.cdiv(n, bm) * piton.cdiv(n, bn),)
    ck = pc.compile(matmul_packed_b_kernel, sig, cons, num_stages=stages, grid=grid)
    programs = []
    for pid, g in program_ids(grid):
        uniforms = []
        for kind, name in ck.uniforms:
            if kind == "arg":
                uniforms.append(("buf", name[0], 0) if name in ("a_ptr", "b_ptr", "c_ptr")
                                else scalars[name])
            elif kind == "pid":
                uniforms.append(pid[int(name[3:])])
            elif kind == "nprog":
                uniforms.append(g[int(name[5:])])
            elif kind == "zero":
                uniforms.append(("buf", "__zero", 0))
            elif kind == "scratch":
                uniforms.append(("buf", "__scratch", 0))
            elif kind == "self":
                uniforms.append(("self",))
            else:
                raise ValueError(f"unknown uniform {kind}: {name}")
        programs.append(uniforms)
    buffers = [Buffer("a", 4 * n * n, lcg_seed=seeds[0]),
               Buffer("b", 4 * n * n, lcg_seed=seeds[1], column_block=(n, bn)),
               Buffer("c", 4 * n * n), Buffer("__zero", 64), Buffer("__scratch", 64)]
    result = PiRunner(verbose=verbose).run(
        ck.words, buffers, programs, ["c"], repeats=repeats, label=ck.name,
        sample={"c": sample_stride})
    a = lcg_floats(n * n, seeds[0]).reshape(n, n).astype(np.float64)
    b = lcg_floats(n * n, seeds[1]).reshape(n, n).astype(np.float64)
    idx = np.arange(0, n * n, sample_stride)
    rows, cols = idx // n, idx % n
    # Only compute sampled references, keeping this practical for large matrices.
    reference = np.einsum("ij,ji->i", a[rows], b[:, cols])
    magnitude = np.einsum("ij,ji->i", np.abs(a[rows]), np.abs(b[:, cols]))
    got = result.outputs["c"].view(np.float32)
    if len(got) != len(idx):
        raise AssertionError(f"expected {len(idx)} samples, received {len(got)}")
    errors = np.abs(got.astype(np.float64) - reference)
    # Each rounded product and sum has relative error at most 2^-23 under RTZ.
    # A gamma bound on 2*K operations covers cancellation without relative-to-C checks.
    eps_ops = 2 * n * (2.0 ** -23)
    if eps_ops >= 1:
        raise ValueError("N is too large for this FP32 forward error bound")
    bounds = (eps_ops / (1 - eps_ops)) * magnitude + 1e-6
    passed = bool(np.all(np.isfinite(got)) and np.all(errors <= bounds))
    if not passed:
        raise AssertionError(f"packed matmul failed sampled correctness: max error {errors.max():.6g}, "
                             f"max error/bound {np.max(errors / bounds):.6g}")
    prep = [int(x) for x in re.findall(r"PITON_PREP b (\d+)", result.log)]
    if len(prep) != 1:
        raise AssertionError("packed input preparation timing was not reported exactly once")
    if len(result.times_us) != repeats or any(t <= 0 for t in result.times_us):
        raise AssertionError("missing or invalid GPU timing")
    best = min(result.times_us)
    return dict(n=n, cfg=(bm, bn, bk, group, stages), specialized=specialize,
                times_us=result.times_us, best_us=best, gflops=2.0 * n ** 3 / (best * 1e3),
                b_generation_packed_us=prep[0], samples=len(idx), max_err=float(errors.max()),
                max_error_bound_ratio=float(np.max(errors / bounds)), passed=passed,
                programs=len(programs), instructions=len(ck.program.instrs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=[512, 1024])
    parser.add_argument("--bm", type=int, default=8)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=8)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--specialize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for size in args.sizes:
        r = bench(size, args.bm, args.bn, args.bk, args.group, args.stages, args.repeats,
                  specialize=args.specialize, verbose=args.verbose)
        print(f"N={size} cfg(BM,BN,BK,G,stages)={r['cfg']} specialized={r['specialized']} "
              f"GPU repeats us={r['times_us']} best={r['best_us']} us {r['gflops']:.3f} GFLOPS; "
              f"B generation+packed-placement={r['b_generation_packed_us']} us (outside GPU timer); "
              f"sampled correctness PASS {r['samples']} values, max err={r['max_err']:.3e}, "
              f"max error/bound={r['max_error_bound_ratio']:.3g}; "
              f"{r['programs']} programs, {r['instructions']} instructions")
