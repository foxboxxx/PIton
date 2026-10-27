"""Emulator tests for a range of Triton kernels against numpy."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
from piton.runtime.launcher import Launcher  # noqa: E402
import kernels as K  # noqa: E402

rng = np.random.default_rng(42)


@pytest.mark.parametrize("rows,cols,block", [(4, 64, 64), (3, 100, 128), (5, 16, 16)])
def test_softmax(rows, cols, block):
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    y = np.zeros_like(x)
    K.softmax_kernel[(rows,)](y, x, cols, cols, cols, BLOCK_SIZE=block)
    e = np.exp(x - x.max(1, keepdims=True))
    np.testing.assert_allclose(y, e / e.sum(1, keepdims=True), rtol=2e-5, atol=1e-6)


def test_scaled_relu():
    n = 300
    x = rng.standard_normal(n).astype(np.float32)
    y = np.full(n, 9.0, np.float32)
    K.scaled_relu_kernel[(piton.cdiv(n, 128),)](x, y, n, 0.5, BLOCK=128)
    np.testing.assert_allclose(y, np.where(x > 0, x * 0.5, 0.0), rtol=1e-6)


def test_row_sum():
    m, n = 16, 96
    x = rng.standard_normal((m, n)).astype(np.float32)
    out = np.zeros(m, np.float32)
    K.row_sum_kernel[(m // 8,)](x, out, n, n, BLOCK_M=8, BLOCK_N=32)
    np.testing.assert_allclose(out, x.sum(1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("M,N,Kd,const_k", [(64, 64, 64, True), (72, 100, 64, True), (40, 70, 36, False),
                                            (16, 128, 32, True)])
def test_tutorial_matmul(M, N, Kd, const_k):
    a = rng.standard_normal((M, Kd)).astype(np.float32)
    b = rng.standard_normal((Kd, N)).astype(np.float32)
    c = np.full((M, N), -1.0, np.float32)
    grid = (piton.cdiv(M, 8) * piton.cdiv(N, 64),)
    L = Launcher(K.matmul_kernel, grid, {"K": Kd} if const_k else {})
    L(a, b, c, M, N, Kd, Kd, 1, N, 1, N, 1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=4, GROUP_SIZE_M=4)
    np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-4)
