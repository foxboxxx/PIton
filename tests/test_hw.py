"""Hardware tests: run compiled kernels on an attached Raspberry Pi.

Skipped unless a Pi, the bootloader and the course libpi are available.
Results must match the emulator bit for bit (the emulator models the
QPU's round-toward-zero float arithmetic).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
from piton.runtime.pi import pi_available  # noqa: E402

pytestmark = pytest.mark.skipif(not pi_available() or os.environ.get("PITON_HW") != "1",
                                reason="set PITON_HW=1 with a Pi attached")


def _both(kernel, grid, make_args, **const):
    emu = make_args()
    hw = make_args()
    kernel[grid](*emu, **const)
    kernel[grid](*hw, device="pi", **const)
    return emu, hw


def test_vector_add_hw():
    from kernels import add_kernel
    n = 1000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    def args():
        return [x.copy(), y.copy(), np.full(n + 16, 7.0, np.float32), n]
    emu, hw = _both(add_kernel, (piton.cdiv(n, 64),), args, BLOCK_SIZE=64)
    np.testing.assert_array_equal(hw[2], emu[2])
    np.testing.assert_allclose(hw[2][:n], x + y, rtol=1e-6)
    assert np.all(hw[2][n:] == 7.0)


def test_softmax_hw():
    from kernels import softmax_kernel
    rows, cols = 6, 100
    x = np.random.default_rng(3).standard_normal((rows, cols)).astype(np.float32)

    def args():
        return [np.zeros_like(x), x.copy(), cols, cols, cols]
    emu, hw = _both(softmax_kernel, (rows,), args, BLOCK_SIZE=128)
    np.testing.assert_array_equal(hw[0], emu[0])
    e = np.exp(x - x.max(1, keepdims=True))
    np.testing.assert_allclose(hw[0], e / e.sum(1, keepdims=True), rtol=2e-5, atol=1e-6)


def test_where_and_row_sum_hw():
    from kernels import scaled_relu_kernel, row_sum_kernel
    rng = np.random.default_rng(4)
    n = 300
    x = rng.standard_normal(n).astype(np.float32)
    emu, hw = _both(scaled_relu_kernel, (3,), lambda: [x.copy(), np.zeros(n, np.float32), n, 0.5], BLOCK=128)
    np.testing.assert_array_equal(hw[1], emu[1])
    m, k = 16, 96
    y = rng.standard_normal((m, k)).astype(np.float32)
    emu, hw = _both(row_sum_kernel, (2,), lambda: [y.copy(), np.zeros(m, np.float32), k, k], BLOCK_M=8, BLOCK_N=32)
    np.testing.assert_array_equal(hw[1], emu[1])
    np.testing.assert_allclose(hw[1], y.sum(1), rtol=1e-5, atol=1e-5)


def test_matmul_ragged_hw():
    from piton.runtime.launcher import Launcher
    from kernels import matmul_kernel
    rng = np.random.default_rng(5)
    M, N, K = 72, 100, 64
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    L = Launcher(matmul_kernel, (piton.cdiv(M, 8) * piton.cdiv(N, 64),), {"K": K})
    kw = dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=4, GROUP_SIZE_M=4)
    out = []
    for dev in ("emu", "pi"):
        c = np.full((M, N), -1.0, np.float32)
        L(a, b, c, M, N, K, K, 1, N, 1, N, 1, device=dev, **kw)
        out.append(c)
    np.testing.assert_array_equal(out[1], out[0])
    np.testing.assert_allclose(out[1], a @ b, rtol=1e-4, atol=1e-4)


def test_exp_log_hw():
    """exp/log run on the ALUs, so the Pi must match the emulator bit for bit."""
    from test_tiles import explog_k, EXPLOG_SPECIALS
    rng = np.random.default_rng(6)
    for F in range(4):
        if F < 2:
            x = rng.uniform(-87, 88, 500).astype(np.float32)
        else:
            x = np.exp(rng.uniform(-85, 85, 500)).astype(np.float32)
        x = np.concatenate([x, EXPLOG_SPECIALS])
        emu, hw = _both(explog_k, (piton.cdiv(len(x), 128),),
                        lambda: [x.copy(), np.zeros_like(x), len(x)], F=F, BLOCK=128)
        np.testing.assert_array_equal(hw[1].view(np.uint32), emu[1].view(np.uint32))


def test_div_sqrt_hw():
    from test_tiles import divsqrt_k, DIV_SPECIALS
    rng = np.random.default_rng(9)
    for F in range(3):
        y = np.concatenate([np.exp(rng.uniform(-80, 80, 500)).astype(np.float32), DIV_SPECIALS])
        x = rng.standard_normal(len(y)).astype(np.float32)
        emu, hw = _both(divsqrt_k, (piton.cdiv(len(y), 128),),
                        lambda: [x.copy(), y.copy(), np.zeros_like(y), len(y)], F=F, BLOCK=128)
        np.testing.assert_array_equal(hw[2].view(np.uint32), emu[2].view(np.uint32))


def test_float_compare_hw():
    from test_tiles import fcmp_k, SPECIALS
    rng = np.random.default_rng(7)
    for C in (None, 0.0, 1.5, float("-inf")):
        x = rng.permutation(SPECIALS)
        y = rng.permutation(SPECIALS) if C is None else np.full(16, C, np.float32)
        emu, hw = _both(fcmp_k, (1,), lambda: [x.copy(), y.copy(), np.zeros(96, np.int32)],
                        C=0.0 if C is None else C, MODE=0 if C is None else 1, BLOCK=16)
        np.testing.assert_array_equal(hw[2], emu[2])
        with np.errstate(invalid="ignore"):
            ref = np.concatenate([x < y, x <= y, x > y, x >= y, x == y, x != y]).astype(np.int32)
        np.testing.assert_array_equal(hw[2], ref)


def test_small_tiles_hw():
    """Lane permutations (masked rotated moves): reductions, transpose, small dot."""
    from test_tiles import reduce2d, trans_k, dot_k
    rng = np.random.default_rng(8)
    for R, L, axis in ((16, 4, 1), (8, 8, 0), (64, 4, 1)):
        x = rng.integers(-50, 50, (R, L)).astype(np.float32)
        emu, hw = _both(reduce2d, (1,), lambda: [x.copy(), np.zeros(L if axis == 0 else R, np.float32)],
                        R=R, L=L, AXIS=axis, KIND=0)
        np.testing.assert_array_equal(hw[1], emu[1])
        np.testing.assert_array_equal(hw[1], x.sum(axis=axis))
    x = rng.integers(-100, 100, (8, 8)).astype(np.float32)
    emu, hw = _both(trans_k, (1,), lambda: [x.copy(), np.zeros(64, np.float32)], M=8, N=8)
    np.testing.assert_array_equal(hw[1], (x + 1.0).T.ravel())
    a = rng.integers(-4, 4, (4, 8)).astype(np.float32)
    b = rng.integers(-4, 4, (8, 8)).astype(np.float32)
    emu, hw = _both(dot_k, (1,), lambda: [a.copy(), b.copy(), np.zeros(32, np.float32), np.zeros(32, np.float32)],
                    M=4, N=8, K=8, ACC=False)
    np.testing.assert_array_equal(hw[3], (a @ b).ravel())


@piton.jit
def _strided_store(x_ptr, out_ptr, base, stride, R: piton.language.constexpr, L: piton.language.constexpr):
    tl = piton.language
    r = tl.arange(0, R)
    c = tl.arange(0, L)
    x = tl.load(x_ptr + r[:, None] * L + c[None, :])
    tl.store(out_ptr + base + r[:, None] * stride + c[None, :], x)


def test_store_row_strides_hw():
    """Row strides that do not fit the VDW stride field (here negative ones; huge
    ones would need large buffers over the serial link) use one DMA per row."""
    R, L = 4, 16
    x = np.arange(R * L, dtype=np.float32) + 1
    for stride in (16, -16, -48):
        n = R * abs(stride) + L
        base = 0 if stride > 0 else (R - 1) * -stride    # negative stride: start at the last row
        emu, hw = _both(_strided_store, (1,), lambda: [x.copy(), np.zeros(n, np.float32), base, stride],
                        R=R, L=L)
        ref = np.zeros(n, np.float32)
        for r in range(R):
            ref[base + r * stride: base + r * stride + L] = x[r * L:(r + 1) * L]
        np.testing.assert_array_equal(emu[1], ref)
        np.testing.assert_array_equal(hw[1], ref)
