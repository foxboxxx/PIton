"""Benchmark the Triton tutorial matmul on a Raspberry Pi (all 12 QPUs).

Inputs are generated on the board (bit-exact LCG shared with the host), the
result is checked on a sample of C entries against a float64 reference.

    python3 benchmarks/matmul_pi.py 256 512 --bm 8 --bn 64 --bk 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
import piton.compiler as pc  # noqa: E402
from piton.runtime.pi import bench_matmul_on_pi  # noqa: E402
from kernels import matmul_kernel  # noqa: E402


def bench(n, bm, bn, bk, group=4, stages=2, repeats=3):
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
           "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}
    cons = dict(K=n, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk, GROUP_SIZE_M=group)
    ck = pc.compile(matmul_kernel, sig, cons, num_stages=stages)
    grid = (piton.cdiv(n, bm) * piton.cdiv(n, bn),)
    r = bench_matmul_on_pi(ck, n, n, n, grid,
                           {"M": n, "N": n, "stride_am": n, "stride_bk": n, "stride_cm": n},
                           repeats=repeats)
    r.update(n=n, cfg=(bm, bn, bk, group, stages), instructions=len(ck.program.instrs))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int, default=[256])
    ap.add_argument("--bm", type=int, default=8)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--stages", type=int, default=2)
    a = ap.parse_args()
    for n in a.sizes:
        r = bench(n, a.bm, a.bn, a.bk, a.group, a.stages)
        print(f"N={n:5d} cfg(BM,BN,BK,G,stages)={r['cfg']}  best {r['best_us']:8d} us  "
              f"{r['gflops']:.3f} GFLOPS  max sampled err {r['max_err']:.2e}  ({r['programs']} programs)")
