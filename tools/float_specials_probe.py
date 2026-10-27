"""Probe how the QPU's float ALU ops treat NaN, infinities, zeros and
denormals, against the emulator (run with a Pi attached).

    PITON_HW=1 python tools/float_specials_probe.py [results.npz]

Expected: 1 mismatch (ftoi of -1.7e38, not modelled).
"""
import sys

import numpy as np

import piton
import piton.language as tl


@piton.jit
def special_ops(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=1.0)
    b = tl.load(b_ptr + offs, mask=m, other=1.0)
    tl.store(o_ptr + 0 * BLOCK + offs, a + b, mask=m)
    tl.store(o_ptr + 1 * BLOCK + offs, a - b, mask=m)
    tl.store(o_ptr + 2 * BLOCK + offs, a * b, mask=m)
    tl.store(o_ptr + 3 * BLOCK + offs, tl.minimum(a, b), mask=m)
    tl.store(o_ptr + 4 * BLOCK + offs, tl.maximum(a, b), mask=m)
    tl.store(o_ptr + 5 * BLOCK + offs, a.to(tl.int32).to(tl.float32, bitcast=True), mask=m)
    tl.store(o_ptr + 6 * BLOCK + offs, a.to(tl.int32, bitcast=True).to(tl.float32), mask=m)


SPECIALS = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.5, 3.4e38, -3.4e38,
                     1e-39, -1e-39], np.float32)
# extra (a, b) pairs: underflowing results, ftoi / itof ranges, rounding of large sums
EXTRA = [(1e-30, -1e-30), (-1e-20, 1e-20), (1.5e-38, -1.4e-38), (-1.5e-38, 1.4e-38), (1.2e-38, 0.5),
         (-1.2e-38, 0.5), (2147483520.0, 1.0), (2147483648.0, 1.0), (-2147483648.0, 1.0),
         (-2147483904.0, 1.0), (1e10, 1.0), (16777217.0, 1.0), (1.0, 2.0 ** -30), (-1.0, 2.0 ** -30),
         (1.0, -(2.0 ** -30)), (1.5, 1.5), (-1.5, -1.5), (2.5, -2.5)]
NAMES = ["add", "sub", "mul", "min", "max", "ftoi", "itof"]


def main():
    a = np.repeat(SPECIALS, len(SPECIALS))
    b = np.tile(SPECIALS, len(SPECIALS))
    a = np.concatenate([a, np.array([p[0] for p in EXTRA], np.float32)])
    b = np.concatenate([b, np.array([p[1] for p in EXTRA], np.float32)])
    for v in (a, b):
        v[np.isnan(v) & np.signbit(v)] = np.uint32(0xFFC00000).view(np.float32)
    ints = np.array([16777217, 16777219, 2147483647, -2147483647, -16777217, 33554435], np.int32)
    a[len(a) - len(ints):] = ints.view(np.float32)       # itof inputs (bit patterns)
    n, B = len(a), 256
    res = {}
    for dev in ("emu", "pi"):
        o = np.zeros(7 * B, np.float32)
        special_ops[(1,)](a, b, o, n, BLOCK=B, device=dev)
        res[dev] = o.view(np.uint32).reshape(7, B)[:, :n]
    bad = 0
    for k, name in enumerate(NAMES):
        for i in np.nonzero(res["emu"][k] != res["pi"][k])[0]:
            bad += 1
            print(f"{name:5s} a={a[i]!r:>22} b={b[i]!r:>22}  emu={res['emu'][k][i]:08x}  pi={res['pi'][k][i]:08x}")
    print("mismatches:", bad)
    if len(sys.argv) > 1:                      # optional: save the raw results
        np.savez(sys.argv[1], a=a, b=b, emu=res["emu"], pi=res["pi"])
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
