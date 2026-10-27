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
