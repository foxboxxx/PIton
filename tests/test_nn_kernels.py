"""Dense-layer fusion, incomplete tiles and a complete MLP on the strict emulator."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton.compiler as pc
from nn_kernels import dense_layer_kernel, dense_tiles
from piton.runtime.launcher import run_emulator


def run_dense(x, w, bias, *, relu, add_bias=True, stages=2, tiles=None, compiled=None):
    m, k = x.shape
    n = w.shape[1]
    bm, bn, bk = tiles or dense_tiles(n)
    grid = ((m + bm - 1) // bm * ((n + bn - 1) // bn),)
    constants = dict(N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                     BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, ADD_BIAS=add_bias, RELU=relu)
    signature = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    signature["M"] = "i32"
    if compiled is None:
        compiled = pc.compile(dense_layer_kernel, signature, constants,
                              grid=grid, num_stages=stages)
    # The suffix catches stores from masked rows in the final tile.
    storage = np.full(m * n + 32, np.float32(-1234.5), dtype=np.float32)
    args = dict(x_ptr=x, w_ptr=w, bias_ptr=bias, y_ptr=storage, M=m)
    originals = [a.copy() for a in (x, w, bias)]
    run_emulator(compiled, grid, args, strict=True)
    for a, original in zip((x, w, bias), originals):
        np.testing.assert_array_equal(a.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(storage[m * n:], np.float32(-1234.5))
    return storage[:m * n].reshape(m, n)


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("m,n,k,add_bias,relu", [
    (8, 64, 64, True, True),
    (9, 17, 11, True, False),
    (3, 10, 7, False, True),
    (1, 16, 8, False, False),
])
def test_dense_layer_fused_and_ragged(m, n, k, add_bias, relu, stages):
    rng = np.random.default_rng(340 + m)
    # Dyadic values have exact products and sums in this range, enabling a bit
    # comparison of fused bias broadcasting, masking and ReLU with NumPy.
    x = rng.integers(-4, 5, (m, k)).astype(np.float32) / np.float32(4)
    w = rng.integers(-4, 5, (k, n)).astype(np.float32) / np.float32(4)
    bias = rng.integers(-8, 9, n).astype(np.float32) / np.float32(4)
    # The single-stage schedule needs a smaller output tile to leave room for
    # its live load masks; the measured two-stage schedule uses dense_tiles().
    tiles = (4, 32, 8) if stages == 1 and n >= 64 else None
    actual = run_dense(x, w, bias, relu=relu, add_bias=add_bias,
                       stages=stages, tiles=tiles)
    expected = x @ w
    if add_bias:
        expected += bias
    if relu:
        expected = np.maximum(expected, np.float32(0))
    np.testing.assert_array_equal(actual, expected)


def test_dense_network_64_128_64_16():
    """Real model dimensions, three launches and a ragged inference batch."""
    rng = np.random.default_rng(340)
    x = rng.uniform(-1, 1, (3, 64)).astype(np.float32)
    reference = x.astype(np.float64)
    for index, (k, n) in enumerate(((64, 128), (128, 64), (64, 16))):
        w = rng.normal(0, 0.1, (k, n)).astype(np.float32)
        bias = rng.normal(0, 0.1, n).astype(np.float32)
        relu = index < 2
        x = run_dense(x, w, bias, relu=relu)
        reference = reference @ w.astype(np.float64) + bias.astype(np.float64)
        if relu:
            reference = np.maximum(reference, 0)
        # VC4 rounds arithmetic toward zero, so use an absolute tolerance near
        # cancellation and check every activation after each layer.
        np.testing.assert_allclose(x, reference, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(np.argmax(x[:, :10], axis=1),
                                  np.argmax(reference[:, :10], axis=1))


def test_runtime_batch_reuses_compiled_layer():
    """A firmware layer compiled for 1,800 rows can run tiny batch prefixes."""
    bm, bn, bk = dense_tiles(64)
    constants = dict(N=64, K=128, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                     BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, ADD_BIAS=True, RELU=True)
    signature = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    signature["M"] = "i32"
    compiled = pc.compile(dense_layer_kernel, signature, constants,
                          grid=((1800 + bm - 1) // bm,))
    rng = np.random.default_rng(12864)
    w = rng.normal(0, 0.05, (128, 64)).astype(np.float32)
    bias = rng.normal(0, 0.05, 64).astype(np.float32)
    for m in (1, 8, 9):
        x = rng.uniform(-1, 1, (m, 128)).astype(np.float32)
        actual = run_dense(x, w, bias, relu=True, compiled=compiled)
        reference = np.maximum(x.astype(np.float64) @ w.astype(np.float64) + bias, 0)
        np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-5)
