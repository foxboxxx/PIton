"""The performance suite: every kernel PIton ships, with inputs and references.

Each `Case` builds its inputs through `Inputs`, which records how every buffer
was generated (the LCG of `piton_rt.h`, optionally scaled or turned into
integers). The emulator runs use the arrays directly; `perf/profile_pi.py`
regenerates the same bits on the board, so a large Pi run uploads no data and
its sampled outputs are still checked against the same numpy reference.

Sizes: "emu" keeps every case to a few seconds in the strict emulator; "pi"
matches the problem sizes of `pi/examples/primitives` and the benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from typing import Callable, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "examples"), os.path.join(ROOT, "benchmarks")):
    if path not in sys.path:
        sys.path.insert(0, path)

import piton  # noqa: E402
from kernels import add_kernel, matmul_kernel, row_sum_kernel, scaled_relu_kernel, softmax_kernel  # noqa: E402
from elementwise import mul_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402
from primitive_kernels import copy_kernel, fma_kernel, idiv_kernel, log_kernel, row_max_kernel, transpose_kernel  # noqa: E402
from nn_kernels import dense_layer_kernel  # noqa: E402
from mandelbrot import SIGNATURE as MANDEL_SIGNATURE, TARGET, mandelbrot_kernel, reference_counts  # noqa: E402
from compute_peak_pi import compute_kernel  # noqa: E402

LCG_A, LCG_C = 1664525, 1013904223
PTR, IPTR, UPTR = "*fp32", "*i32", "*u32"


def lcg_u32(n: int, seed: int) -> np.ndarray:
    """The first n outputs of piton_lcg() from `seed`, vectorised.

    s_k = a^k s_0 + c (1 + a + ... + a^(k-1)) mod 2^32; uint64 products wrap
    mod 2^64, which 2^32 divides, so the low 32 bits stay exact.
    """
    with np.errstate(over="ignore"):
        powers = np.cumprod(np.full(n, LCG_A, dtype=np.uint64))              # a^1 .. a^n
        geometric = np.cumsum(np.concatenate((np.ones(1, np.uint64), powers[:-1])))   # sum a^0..a^(k-1)
        s = powers * np.uint64(seed & 0xFFFFFFFF) + geometric * np.uint64(LCG_C)
    return (s & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def lcg_floats(n: int, seed: int) -> np.ndarray:
    """piton_lcg_float(): [-1, 1) with a 24-bit mantissa."""
    return (lcg_u32(n, seed) >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 8388608.0) - np.float32(1.0)


@dataclass
class BufferSpec:
    """How a buffer's initial contents are produced (on the host and the Pi)."""
    kind: str                       # "float" | "int" | "fill"
    count: int
    dtype: np.dtype
    seed: int = 0
    scale: float = 1.0              # float: lcg * scale + offset
    offset: float = 0.0
    shift: int = 0                  # int: ((int32)lcg >> shift) | or_mask
    or_mask: int = 0
    fill: int = 0                   # fill: every word (uint32 bits)


class Inputs:
    """Kernel arguments in signature order, plus how each buffer was made."""

    def __init__(self):
        self.values: dict = {}
        self.specs: dict = {}
        self._seed = 1

    def _next_seed(self) -> int:
        self._seed += 1
        return 1000003 * self._seed

    def floats(self, name: str, count: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
        seed = self._next_seed()
        scale, offset = np.float32((hi - lo) / 2), np.float32((hi + lo) / 2)
        arr = lcg_floats(count, seed) * scale + offset
        self.specs[name] = BufferSpec("float", count, np.float32, seed, float(scale), float(offset))
        self.values[name] = arr
        return arr.copy()

    def ints(self, name: str, count: int, shift: int = 0, or_mask: int = 0) -> np.ndarray:
        seed = self._next_seed()
        arr = (lcg_u32(count, seed).view(np.int32) >> shift) | np.int32(or_mask)
        self.specs[name] = BufferSpec("int", count, np.int32, seed, shift=shift, or_mask=or_mask)
        self.values[name] = arr
        return arr.copy()

    def output(self, name: str, count: int, dtype=np.float32) -> None:
        # NaN (or all ones) shows any element the kernel failed to write
        fill = 0x7FC00000 if np.dtype(dtype) == np.float32 else 0xFFFFFFFF
        self.specs[name] = BufferSpec("fill", count, np.dtype(dtype), fill=fill)
        self.values[name] = np.full(count, fill, dtype=np.uint32).view(dtype)

    def scalar(self, name: str, value) -> None:
        self.values[name] = value


@dataclass
class Run:
    inputs: Inputs
    grid: tuple
    expected: dict                  # output name -> array (same shape as the buffer)
    rtol: float = 1e-6
    atol: float = 1e-6
    # optional custom check: (name, got, expected) -> error message or None
    check: Optional[Callable] = None


@dataclass
class Case:
    name: str
    kernel: object
    signature: dict
    constants: dict
    make: Callable[[str], Run]      # size: "emu" | "pi"
    note: str = ""
    tags: tuple = field(default_factory=tuple)
    options: dict = field(default_factory=dict)     # extra piton.compiler.compile arguments


def default_check(run: Run, name: str, got: np.ndarray, expected: np.ndarray) -> Optional[str]:
    if run.check is not None:
        return run.check(name, got, expected)
    if got.dtype.kind in "iu":
        bad = np.flatnonzero(got != expected)
        return None if bad.size == 0 else f"{bad.size} wrong, first at {bad[0]}: {got[bad[0]]} != {expected[bad[0]]}"
    if not np.isfinite(got).all():
        return f"{int((~np.isfinite(got)).sum())} non-finite outputs"
    err = np.abs(got.astype(np.float64) - expected)
    bad = np.flatnonzero(err > run.atol + run.rtol * np.abs(expected))
    return None if bad.size == 0 else (f"{bad.size} outside tolerance, worst {err.max():.3g} at "
                                       f"{int(err.argmax())} (expected {expected[int(err.argmax())]:.6g})")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

VEC = {"emu": 4096, "pi": 262144}
MAT = {"emu": 64, "pi": 256}


def elementwise(name, kernel, constants, block, op, lo=-1.0, hi=1.0, inputs=2, scalar=None,
                rtol=1e-6, atol=1e-6, names=("x_ptr", "y_ptr", "out_ptr"), note="", n_name="n"):
    sig = {n: PTR for n in names[:inputs]}
    sig[names[-1]] = PTR
    sig[n_name] = "i32"
    if scalar is not None:
        sig[scalar[0]] = "fp32"

    def make(size):
        n = VEC[size]
        g = Inputs()
        xs = [g.floats(nm, n, lo, hi).astype(np.float64) for nm in names[:inputs]]
        g.output(names[-1], n)
        g.scalar(n_name, n)
        if scalar is not None:
            g.scalar(scalar[0], np.float32(scalar[1]))
        return Run(g, (n // block,), {names[-1]: op(*xs)}, rtol, atol)
    return Case(name, kernel, sig, dict(constants), make, note, ("elementwise",))


def sfu(name, op_code, op, lo, hi, rtol=4e-6, atol=1e-30):
    return elementwise(name, sfu_kernel, {"OP": op_code, "BLOCK": 256}, 256, op, lo, hi, inputs=1,
                       rtol=rtol, atol=atol, names=("x_ptr", "out_ptr"), note=f"sfu_kernel OP={op_code}")


def idiv_case():
    sig = {"x_ptr": IPTR, "y_ptr": IPTR, "q_ptr": IPTR, "r_ptr": IPTR, "n": "i32"}

    def make(size):
        n = VEC[size]
        g = Inputs()
        x = g.ints("x_ptr", n).astype(np.int64)
        y = g.ints("y_ptr", n, shift=8, or_mask=1).astype(np.int64)    # nonzero, varied size
        g.output("q_ptr", n, np.int32)
        g.output("r_ptr", n, np.int32)
        g.scalar("n", n)
        q = np.sign(x) * np.sign(y) * (np.abs(x) // np.abs(y))           # C: truncate toward zero
        return Run(g, (n // 256,), {"q_ptr": q.astype(np.int32), "r_ptr": (x - q * y).astype(np.int32)})
    return Case("idiv", idiv_kernel, sig, {"BLOCK": 256}, make, "exact 32-bit // and %", ("int",))


def row_reduce(name, kernel, op, rtol):
    sig = {"x_ptr": PTR, "out_ptr": PTR, "n_cols": "i32", "stride": "i32"}

    def make(size):
        m = MAT[size]
        g = Inputs()
        x = g.floats("x_ptr", m * m).reshape(m, m).astype(np.float64)
        g.output("out_ptr", m)
        g.scalar("n_cols", m)
        g.scalar("stride", m)
        return Run(g, (m // 4,), {"out_ptr": op(x, axis=1)}, rtol, 1e-5)
    return Case(name, kernel, sig, {"BLOCK_M": 4, "BLOCK_N": 64}, make, "", ("reduction",))


def transpose_case():
    sig = {"x_ptr": PTR, "out_ptr": PTR, "n_rows": "i32", "n_cols": "i32", "stride_x": "i32", "stride_out": "i32"}

    def make(size):
        m = MAT[size]
        g = Inputs()
        x = g.floats("x_ptr", m * m).reshape(m, m)
        g.output("out_ptr", m * m)
        for k in ("n_rows", "n_cols", "stride_x", "stride_out"):
            g.scalar(k, m)
        return Run(g, ((m // 16) ** 2,), {"out_ptr": x.T.ravel().astype(np.float64)}, 0, 0)
    return Case("transpose", transpose_kernel, sig, {"BLOCK_M": 16, "BLOCK_N": 16}, make, "", ("movement",))


def softmax_case():
    sig = {"output_ptr": PTR, "input_ptr": PTR, "input_row_stride": "i32", "output_row_stride": "i32", "n_cols": "i32"}

    def make(size):
        m = MAT[size]
        g = Inputs()
        x = g.floats("input_ptr", m * m, -4.0, 4.0).reshape(m, m).astype(np.float64)
        g.output("output_ptr", m * m)
        g.scalar("input_row_stride", m)
        g.scalar("output_row_stride", m)
        g.scalar("n_cols", m)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return Run(g, (m,), {"output_ptr": (e / e.sum(axis=1, keepdims=True)).ravel()}, 2e-5, 1e-8)
    return Case("softmax", softmax_kernel, sig, {"BLOCK_SIZE": MAT["pi"]}, make, "tutorial 02", ("reduction",))


def matmul_case(name, bm, bn, bk, group=4):
    sig = {"a_ptr": PTR, "b_ptr": PTR, "c_ptr": PTR, "M": "i32", "N": "i32",
           "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}
    k_pi = MAT["pi"]

    def make(size):
        n = MAT[size]
        g = Inputs()
        a = g.floats("a_ptr", n * k_pi).reshape(n, k_pi).astype(np.float64)
        b = g.floats("b_ptr", k_pi * n).reshape(k_pi, n).astype(np.float64)
        g.output("c_ptr", n * n)
        for k, v in (("M", n), ("N", n), ("stride_am", k_pi), ("stride_bk", n), ("stride_cm", n)):
            g.scalar(k, v)
        grid = (((n + bm - 1) // bm) * ((n + bn - 1) // bn),)
        return Run(g, grid, {"c_ptr": (a @ b).ravel()}, 1e-3, 2e-4 * k_pi)
    cons = {"K": k_pi, "stride_ak": 1, "stride_bn": 1, "stride_cn": 1, "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": group}
    return Case(name, matmul_kernel, sig, cons, make, f"tutorial 03, {bm}x{bn}x{bk}", ("dot",))


def dense_case():
    sig = {"x_ptr": PTR, "w_ptr": PTR, "bias_ptr": PTR, "y_ptr": PTR, "M": "i32"}
    n_in, n_out = 64, 128

    def make(size):
        m = {"emu": 32, "pi": 1800}[size]
        g = Inputs()
        x = g.floats("x_ptr", m * n_in, 0.0, 1.0).reshape(m, n_in).astype(np.float64)
        w = g.floats("w_ptr", n_in * n_out, -0.3, 0.3).reshape(n_in, n_out).astype(np.float64)
        b = g.floats("bias_ptr", n_out, -0.1, 0.1).astype(np.float64)
        g.output("y_ptr", m * n_out)
        g.scalar("M", m)
        return Run(g, ((m // 4) * (n_out // 64),), {"y_ptr": np.maximum(x @ w + b, 0).ravel()}, 1e-4, 1e-4)
    cons = dict(N=n_out, K=n_in, BLOCK_SIZE_M=4, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8, GROUP_SIZE_M=1,
                ADD_BIAS=True, RELU=True)
    return Case("dense_relu", dense_layer_kernel, sig, cons, make, "digits layer 1", ("dot",))


def mandelbrot_case(name, block, unroll, tile_h):
    def make(size):
        width, height, zoom, iters = {"emu": (64, 16, 1000, 128), "pi": (640, 480, 1000, 352)}[size]
        step = np.float32(3.5 / zoom / 640)
        x0 = np.float32(TARGET[0] - width / 2 * float(step))
        y0 = np.float32(TARGET[1] - height / 2 * float(step))
        g = Inputs()
        palette = np.arange(iters + 1, dtype=np.int32)          # identity: counts come back
        palette[0] = iters
        g.values["palette_ptr"] = palette
        g.specs["palette_ptr"] = BufferSpec("ramp", iters + 1, np.int32, fill=iters)
        g.output("out_ptr", width * height, np.int32)
        for k, v in (("x0", x0), ("y0", y0), ("step", step), ("width", width), ("pitch", width),
                     ("max_iter", iters)):
            g.scalar(k, v)
        ref = reference_counts(x0, y0, step, width, height, iters).astype(np.int32).ravel()

        def check(_, got, expected):
            # fp32 round-toward-zero moves a few boundary pixels (see tests/test_mandelbrot.py)
            frac = float((got != expected).mean())
            return None if frac < 0.15 else f"{frac:.1%} of pixels differ from float64"
        return Run(g, (width // block, height // 8), {"out_ptr": ref}, check=check)
    cons = dict(BLOCK=block, UNROLL=unroll, ROWS=8, TILE_H=tile_h)
    return Case(name, mandelbrot_kernel, dict(MANDEL_SIGNATURE, out_ptr=IPTR, palette_ptr=IPTR), cons, make,
                f"{block}x{unroll}x{tile_h}, zoom 1000", ("compute",))


def compute_peak_case():
    def make(size):
        programs, iters = {"emu": (12, 4), "pi": (12, 1024)}[size]
        g = Inputs()
        x = g.floats("x_ptr", 64 * programs, -1.0, 1.0)
        g.output("out_ptr", 64 * programs)
        # closed form of a <- 0.9999 a + 0.9999 (fixed point 9999) after ITERS * UNROLL steps
        ref = 9999.0 + (x.astype(np.float64) - 9999.0) * 0.9999 ** (iters * 16)
        return Run(g, (programs,), {"out_ptr": ref}, 5e-3, 0)
    return Case("compute_peak", compute_kernel, {"x_ptr": PTR, "out_ptr": PTR},
                {"ITERS": 1024, "UNROLL": 16}, make, "register-resident FMA recurrences", ("compute",))


def _peak_constants(size):
    return {"ITERS": {"emu": 4, "pi": 1024}[size], "UNROLL": 16}


CASES = [
    elementwise("copy", copy_kernel, {"BLOCK": 256}, 256, lambda x: x, inputs=1, names=("x_ptr", "out_ptr")),
    elementwise("add", add_kernel, {"BLOCK_SIZE": 256}, 256, lambda x, y: x + y,
                names=("x_ptr", "y_ptr", "output_ptr"), note="tutorial 01", n_name="n_elements"),
    elementwise("mul", mul_kernel, {"BLOCK": 256}, 256, lambda x, y: x * y),
    elementwise("fma", fma_kernel, {"BLOCK": 256}, 256, lambda x, y: x * 1.5 + y, scalar=("alpha", 1.5)),
    elementwise("scaled_relu", scaled_relu_kernel, {"BLOCK": 256}, 256, lambda x: np.where(x > 0, x * 0.5, 0.0),
                inputs=1, scalar=("alpha", 0.5), names=("x_ptr", "out_ptr"), note="tl.where"),
    sfu("exp", 4, np.exp, -8.0, 8.0),
    sfu("exp2", 0, np.exp2, -8.0, 8.0),
    elementwise("log", log_kernel, {"BLOCK": 256}, 256, np.log, 0.01, 100.0, inputs=1, rtol=4e-6, atol=4e-7,
                names=("x_ptr", "out_ptr")),
    sfu("log2", 1, np.log2, 0.01, 100.0, atol=4e-7),
    sfu("recip", 2, lambda x: 1.0 / x, 0.01, 100.0),
    sfu("rsqrt", 3, lambda x: 1.0 / np.sqrt(x), 0.01, 100.0),
    sfu("sqrt", 5, np.sqrt, 0.01, 100.0),
    idiv_case(),
    row_reduce("row_sum", row_sum_kernel, np.sum, 1e-5),
    row_reduce("row_max", row_max_kernel, np.max, 0),
    transpose_case(),
    softmax_case(),
    matmul_case("matmul_8x64x4", 8, 64, 4),
    matmul_case("matmul_8x64x8", 8, 64, 8),
    dense_case(),
    mandelbrot_case("mandel_32x16", 32, 16, 1),
    mandelbrot_case("mandel_16x16x2", 16, 16, 2),
    compute_peak_case(),
]
CASES_BY_NAME = {c.name: c for c in CASES}


def constants_for(case: Case, size: str) -> dict:
    """Compile-time constants; a few cases shrink a constant for the emulator."""
    cons = dict(case.constants)
    if case.name == "compute_peak":
        cons.update(_peak_constants(size))
    if case.name == "softmax":
        cons["BLOCK_SIZE"] = MAT[size]
    return cons


def compile_case(case: Case, size: str, run: Optional[Run] = None):
    run = run or case.make(size)
    bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in run.grid)
    return piton.compiler.compile(case.kernel, case.signature, constants_for(case, size), grid=bound,
                                  **case.options), run
