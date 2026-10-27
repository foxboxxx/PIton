"""Emulator tests for tile shapes that do not fill whole registers (reductions,
transposes, small dots, broadcasts) and for IEEE float comparisons."""
import numpy as np
import pytest

import piton
import piton.language as tl

rng = np.random.default_rng(7)


@piton.jit
def reduce2d(x_ptr, o_ptr, R: tl.constexpr, L: tl.constexpr, AXIS: tl.constexpr, KIND: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, L)[None, :]
    x = tl.load(x_ptr + r * L + c)
    if KIND == 0:
        y = tl.sum(x, axis=AXIS)
    elif KIND == 1:
        y = tl.max(x, axis=AXIS)
    else:
        y = tl.min(x, axis=AXIS)
    n: tl.constexpr = L if AXIS == 0 else R
    tl.store(o_ptr + tl.arange(0, n), y)


@pytest.mark.parametrize("R,L", [(8, 4), (16, 4), (8, 8), (32, 2), (4, 8), (2, 16), (64, 4), (16, 8)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_reduce_small_tiles(R, L, axis, kind):
    x = rng.integers(-50, 50, (R, L)).astype(np.float32)
    o = np.zeros(L if axis == 0 else R, np.float32)
    reduce2d[(1,)](x, o, R=R, L=L, AXIS=axis, KIND=kind)
    ref = [x.sum, x.max, x.min][kind](axis=axis)
    np.testing.assert_array_equal(o, ref)


@piton.jit
def reduce3d(x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, L: tl.constexpr, AXIS: tl.constexpr):
    i = tl.arange(0, A)[:, None, None]
    j = tl.arange(0, B)[None, :, None]
    k = tl.arange(0, L)[None, None, :]
    x = tl.load(x_ptr + i * (B * L) + j * L + k)
    y = tl.sum(x, axis=AXIS)
    if AXIS == 0:
        tl.store(o_ptr + tl.arange(0, B)[:, None] * L + tl.arange(0, L)[None, :], y)
    elif AXIS == 1:
        tl.store(o_ptr + tl.arange(0, A)[:, None] * L + tl.arange(0, L)[None, :], y)
    else:
        tl.store(o_ptr + tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :], y)


@pytest.mark.parametrize("A,B,L", [(2, 4, 4), (4, 2, 8), (2, 2, 4), (2, 8, 2)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reduce_3d(A, B, L, axis):
    x = rng.integers(-9, 9, (A, B, L)).astype(np.float32)
    ref = x.sum(axis=axis)
    o = np.zeros(ref.size, np.float32)
    reduce3d[(1,)](x, o, A=A, B=B, L=L, AXIS=axis)
    np.testing.assert_array_equal(o, ref.ravel())


@piton.jit
def trans_k(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + r * N + c) + 1.0
    t = tl.trans(x)
    tl.store(o_ptr + tl.arange(0, N)[:, None] * M + tl.arange(0, M)[None, :], t)


@pytest.mark.parametrize("M,N", [(8, 8), (4, 16), (16, 4), (2, 16), (16, 2), (4, 4), (16, 16), (8, 32)])
def test_trans(M, N):
    x = rng.integers(-100, 100, (M, N)).astype(np.float32)
    o = np.zeros(M * N, np.float32)
    trans_k[(1,)](x, o, M=M, N=N)
    np.testing.assert_array_equal(o, (x + 1.0).T.ravel())


@piton.jit
def bcast_k(x_ptr, o_ptr, R: tl.constexpr, L: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, L)
    rs = tl.load(x_ptr + r) * 2.0
    cs = tl.load(x_ptr + R + c) * 3.0
    y = rs[:, None] + cs[None, :]
    tl.store(o_ptr + r[:, None] * L + c[None, :], y)


@pytest.mark.parametrize("R,L", [(8, 8), (64, 4), (32, 2), (16, 4), (4, 16), (32, 16)])
def test_broadcast_small(R, L):
    x = rng.standard_normal(R + L).astype(np.float32)
    o = np.zeros(R * L, np.float32)
    bcast_k[(1,)](x, o, R=R, L=L)
    ref = (x[:R] * np.float32(2.0))[:, None] + (x[R:] * np.float32(3.0))[None, :]
    np.testing.assert_allclose(o, ref.ravel(), rtol=1e-5, atol=1e-6)   # the QPU rounds toward zero


@piton.jit
def dot_k(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          ACC: tl.constexpr):
    am = tl.arange(0, M)[:, None]
    ak = tl.arange(0, K)[None, :]
    bk = tl.arange(0, K)[:, None]
    bn = tl.arange(0, N)[None, :]
    a = tl.load(a_ptr + am * K + ak)
    b = tl.load(b_ptr + bk * N + bn)
    if ACC:
        c = tl.load(c_ptr + am * N + bn)
    else:
        c = tl.zeros((M, N), dtype=tl.float32)
    tl.store(o_ptr + am * N + bn, tl.dot(a, b, c))


@pytest.mark.parametrize("M,N,K", [(4, 8, 8), (2, 4, 8), (8, 8, 4), (1, 8, 16), (16, 2, 4), (4, 8, 32)])
@pytest.mark.parametrize("acc", [False, True])
def test_dot_small_n(M, N, K, acc):
    a = rng.integers(-4, 4, (M, K)).astype(np.float32)
    b = rng.integers(-4, 4, (K, N)).astype(np.float32)
    c = rng.integers(-4, 4, (M, N)).astype(np.float32)
    o = np.zeros(M * N, np.float32)
    dot_k[(1,)](a, b, c, o, M=M, N=N, K=K, ACC=acc)
    np.testing.assert_array_equal(o, (a @ b + (c if acc else 0)).ravel())


@piton.jit
def fcmp_k(x_ptr, y_ptr, o, C: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if MODE == 0:
        y = tl.load(y_ptr + offs)
    else:
        y = C
    tl.store(o + 0 * BLOCK + offs, (x < y).to(tl.int32))
    tl.store(o + 1 * BLOCK + offs, (x <= y).to(tl.int32))
    tl.store(o + 2 * BLOCK + offs, (x > y).to(tl.int32))
    tl.store(o + 3 * BLOCK + offs, (x >= y).to(tl.int32))
    tl.store(o + 4 * BLOCK + offs, (x == y).to(tl.int32))
    tl.store(o + 5 * BLOCK + offs, (x != y).to(tl.int32))


SPECIALS = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1.5, 1e-39, -1e-39,
                     1.1754944e-38, 1.7631416e-38, 3.4e38, -3.4e38, 2.0], np.float32)


@pytest.mark.parametrize("C", [None, 0.0, 1.5, -1.0, float("inf"), float("-inf"), 1e-39, float("nan")])
def test_float_compare_ieee(C):
    for _ in range(2):
        x = rng.permutation(SPECIALS)
        y = rng.permutation(SPECIALS) if C is None else np.full(16, C, np.float32)
        o = np.zeros(96, np.int32)
        fcmp_k[(1,)](x, y, o, C=0.0 if C is None else C, MODE=0 if C is None else 1, BLOCK=16)
        with np.errstate(invalid="ignore"):
            ref = np.concatenate([x < y, x <= y, x > y, x >= y, x == y, x != y]).astype(np.int32)
        np.testing.assert_array_equal(o, ref)


@piton.jit
def explog_k(x_ptr, o_ptr, n, F: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    if F == 0:
        y = tl.exp(x)
    elif F == 1:
        y = tl.exp2(x)
    elif F == 2:
        y = tl.log(x)
    else:
        y = tl.log2(x)
    tl.store(o_ptr + offs, y, mask=m)


EXPLOG_SPECIALS = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 88.7, 88.73,
                            -87.3, -87.4, -103.9, -150, 127.9, 128.0, 1e30, -1e30, 3.4e38, 1.17549435e-38],
                           np.float32)


@pytest.mark.parametrize("F", [0, 1, 2, 3])
def test_exp_log_accuracy(F):
    """exp/log run on the ALUs (not the ~13-bit SFU): within a few ulp of float64."""
    if F < 2:
        x = rng.uniform(-87 if F == 0 else -125, 88 if F == 0 else 127, 2000).astype(np.float32)
    else:
        x = np.exp(rng.uniform(-85, 85, 2000)).astype(np.float32)
    x = np.concatenate([x, EXPLOG_SPECIALS])
    o = np.zeros_like(x)
    explog_k[(piton.cdiv(len(x), 128),)](x, o, len(x), F=F, BLOCK=128)
    with np.errstate(all="ignore"):
        ref = [np.exp, np.exp2, np.log, np.log2][F](x.astype(np.float64)).astype(np.float32)
    ref[np.abs(ref) < 1.17549435e-38] = 0.0          # the QPU flushes denormals
    np.testing.assert_allclose(o, ref, rtol=3e-7, atol=1e-7 if F >= 2 else 0, equal_nan=True)


@piton.jit
def divsqrt_k(x_ptr, y_ptr, o_ptr, n, F: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    y = tl.load(y_ptr + offs, mask=m, other=1.0)
    if F == 0:
        r = x / y
    elif F == 1:
        r = tl.rsqrt(y)
    else:
        r = tl.sqrt(y)
    tl.store(o_ptr + offs, r, mask=m)


DIV_SPECIALS = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 4.0, 2.0 ** 127, 3.4e38,
                         1.2e-38, 0.25, -4.0, 9.0], np.float32)


@pytest.mark.parametrize("F", [0, 1, 2])
def test_div_sqrt_accuracy(F):
    """Division, rsqrt and sqrt run on the ALUs (Newton steps from an integer seed)."""
    y = np.exp(rng.uniform(-80, 80, 2000)).astype(np.float32)
    if F == 0:
        y *= rng.choice([-1, 1], len(y)).astype(np.float32)
    y = np.concatenate([y, DIV_SPECIALS])
    x = rng.standard_normal(len(y)).astype(np.float32)
    o = np.zeros_like(y)
    divsqrt_k[(piton.cdiv(len(y), 128),)](x, y, o, len(y), F=F, BLOCK=128)
    with np.errstate(all="ignore"):
        y64 = y.astype(np.float64)
        ref = [x / y64, 1 / np.sqrt(y64), np.sqrt(y64)][F].astype(np.float32)
    ref[np.abs(ref) < 1.17549435e-38] = 0.0
    np.testing.assert_allclose(o, ref, rtol=4e-7, atol=0, equal_nan=True)
