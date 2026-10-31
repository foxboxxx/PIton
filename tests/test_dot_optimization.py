"""Matmul coverage for A tiles larger than the rotation-accumulator cache."""
import numpy as np
import pytest

import piton
import piton.language as tl
from examples.kernels import matmul_kernel
from piton.compiler import compile
from piton.runtime.launcher import Launcher, run_emulator


@piton.jit
def _loop_matmul(a_ptr, b_ptr, out_ptr, k,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(k, BK)):
        a = tl.load(a_ptr + m * k + i * BK + ak, mask=i * BK + ak < k, other=0.0)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n, mask=i * BK + bk < k, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * BN + n, acc)


@pytest.mark.parametrize("bm,bn,bk,stages", [(4, 32, 16, 2), (4, 64, 16, 2),
                                           (8, 16, 16, 1), (8, 16, 16, 2),
                                           (8, 32, 8, 1)])
@pytest.mark.parametrize("constant_k", [False, True])
def test_large_a_tile_ragged_matmul(bm, bn, bk, stages, constant_k):
    """Several K iterations must preserve every A lane, including ragged tails."""
    m, n, k = bm, bn, 2 * bk + 5
    rng = np.random.default_rng(130)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    out = np.full((m, n), np.nan, np.float32)
    launch = Launcher(_loop_matmul, (1,), {"k": k} if constant_k else {})
    launch(a, b, out, k, BM=bm, BN=bn, BK=bk, num_stages=stages)
    np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bm,bn,bk", [(8, 64, 8), (4, 32, 16), (16, 32, 8)])
def test_unmasked_large_a_loop(bm, bn, bk):
    """Aligned K enables A caching across several iterations of the dot loop."""
    k = 3 * bk
    rng = np.random.default_rng(132)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    out = np.zeros((bm, bn), np.float32)
    Launcher(_loop_matmul, (1,), {"k": k})(a, b, out, k, BM=bm, BN=bn, BK=bk)
    np.testing.assert_array_equal(out, a @ b)


def test_ragged_k_leaves_room_for_mask_address_fixups():
    bm, bn, bk, k = 8, 64, 8, 21
    rng = np.random.default_rng(133)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    out = np.zeros((bm, bn), np.float32)
    Launcher(_loop_matmul, (1,), {"k": k})(a, b, out, k, BM=bm, BN=bn, BK=bk)
    np.testing.assert_array_equal(out, a @ b)


@piton.jit
def _reuse_dot_inputs(a_ptr, b_ptr, c_ptr, out_ptr,
                     M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    ak = tl.arange(0, K)[None, :]
    bk = tl.arange(0, K)[:, None]
    a = tl.load(a_ptr + m * K + ak)
    b = tl.load(b_ptr + bk * N + n)
    c = tl.load(c_ptr + m * N + n)
    acc = tl.dot(a, b, c)
    acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * N + n, acc)


@pytest.mark.parametrize("m,n,k", [(8, 32, 8), (4, 16, 16)])
def test_cached_a_values_survive_later_dot(m, n, k):
    rng = np.random.default_rng(131)
    a = rng.integers(-3, 4, (m, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, n)).astype(np.float32)
    c = rng.integers(-3, 4, (m, n)).astype(np.float32)
    out = np.zeros_like(c)
    _reuse_dot_inputs[(1,)](a, b, c, out, M=m, N=n, K=k)
    np.testing.assert_array_equal(out, 2 * (a @ b) + c)


def test_bk8_matmul_inner_loop_instruction_budget():
    """Keep the measured 8x64x8 inner loop below its old 396 instructions."""
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
           "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}
    ck = compile(matmul_kernel, sig,
                 dict(K=512, stride_ak=1, stride_bn=1, stride_cn=1,
                      BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8, GROUP_SIZE_M=4),
                 grid=(512,))
    labels = ck.program.labels
    start = next(pos for label, pos in labels.items()
                 if label.startswith("loop") and not label.startswith(("loop_end", "loop_exit")))
    end = next(pos for label, pos in labels.items() if label.startswith("loop_end"))
    assert end - start <= 388
    assert ck.stats["dot_a_cache_fallback"] == 0


@piton.jit
def _dot_with_live_rotation_source(a_ptr, b_ptr, x_ptr, tmp_ptr, out_ptr, last_ptr,
                                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    lanes = tl.arange(0, 4)
    x = tl.load(x_ptr + lanes)
    # Broadcasting receives x into a required rotation accumulator. Reusing x
    # after the dot keeps that accumulator live across the entire dot.
    y = x[:, None] + tl.zeros((4, 16), tl.float32)
    tl.store(tmp_ptr + lanes[:, None] * 16 + tl.arange(0, 16)[None, :], y)
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    ak = tl.arange(0, K)[None, :]
    bk = tl.arange(0, K)[:, None]
    a = tl.load(a_ptr + m * K + ak)
    b = tl.load(b_ptr + bk * N + n)
    c = tl.dot(a, b, tl.zeros((M, N), tl.float32))
    tl.store(out_ptr + m * N + n, c)
    tl.store(last_ptr + lanes, x)


@pytest.mark.parametrize("m,n,k", [(8, 16, 8), (8, 32, 8), (4, 16, 16)])
def test_optional_a_cache_falls_back_for_live_rotation_accumulator(m, n, k):
    names = ("a_ptr", "b_ptr", "x_ptr", "tmp_ptr", "out_ptr", "last_ptr")
    ck = compile(_dot_with_live_rotation_source, {name: "*fp32" for name in names},
                 dict(M=m, N=n, K=k), grid=(1,))
    assert ck.stats["dot_a_cache_fallback"] == 1
    assert not ck.stats.get("dot_a_cache_registers", 0)
    rng = np.random.default_rng(134)
    values = dict(a_ptr=rng.integers(-3, 4, (m, k)).astype(np.float32),
                  b_ptr=rng.integers(-3, 4, (k, n)).astype(np.float32),
                  x_ptr=np.array([1, 2, 3, 4], np.float32), tmp_ptr=np.zeros((4, 16), np.float32),
                  out_ptr=np.zeros((m, n), np.float32), last_ptr=np.zeros(4, np.float32))
    run_emulator(ck, (1,), values)
    np.testing.assert_array_equal(values["out_ptr"], values["a_ptr"] @ values["b_ptr"])
    np.testing.assert_array_equal(values["tmp_ptr"], np.broadcast_to(values["x_ptr"][:, None], (4, 16)))
    np.testing.assert_array_equal(values["last_ptr"], values["x_ptr"])
