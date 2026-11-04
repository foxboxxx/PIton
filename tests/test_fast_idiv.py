"""PITON_FAST_IDIV: branch-free exact 32-bit // and % (lower.py _udivmod_fast).

The error bound in _udivmod_fast rests on the reciprocal's two inequalities,
checked here over every float mantissa; a numpy model of the instruction
sequence (the emulator's ALU functions) then runs every pair of edge values
and random pairs, and the strict emulator runs the compiled kernels."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
import piton.compiler  # noqa: E402,F401  (piton.compiler.compile below)
import piton.language as tl  # noqa: E402
from piton.backend import lower  # noqa: E402
from piton.qpu.emulator import alu_add, alu_mul  # noqa: E402
from primitive_kernels import idiv_kernel  # noqa: E402

rng = np.random.default_rng(23)
M32 = 0xFFFFFFFF


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setenv("PITON_FAST_IDIV", "1")
    for k in KERNELS:
        k._cache.clear()
    yield
    for k in KERNELS:
        k._cache.clear()


# -- the reciprocal and a model of the instruction sequence -----------------------

def _a(op, x, y=None):
    x = np.asarray(x, np.uint32)
    y = x if y is None else np.broadcast_to(np.asarray(y, np.uint32), x.shape).astype(np.uint32)
    return alu_add(op, x, y)[0]


def _m(op, x, y):
    x = np.asarray(x, np.uint32)
    return alu_mul(op, x, np.broadcast_to(np.asarray(y, np.uint32), x.shape).astype(np.uint32))[0]


def _recip(fb):
    """lower._div_recip"""
    r = _a("sub", np.full_like(fb, 0x7EF311C3), fb)
    for i in range(3):
        e = _a("fsub", np.full_like(fb, 0x40000000 if i < 2 else lower._DIV_LAST), _m("fmul", fb, r))
        r = _m("fmul", r, e)
    return r


def _trunc32(x):
    """Positive normal float64 holding an exact product or difference of float32
    values, rounded toward zero to float32 precision."""
    return (x.view(np.uint64) & ~np.uint64((1 << 29) - 1)).view(np.float64)


def test_reciprocal_bound_every_mantissa():
    """R * (m + ulp) <= 1 - 2.5 * 2**-23 and R * m >= 1 - 5.51 * 2**-23 for all 2**23
    mantissas m in [1, 2).  The sweep uses exact float64 products truncated like
    the QPU; a sample checks it against the emulator's ALUs bit for bit at the
    exponents of 1 .. 2**31 (the sequence scales exactly with the exponent)."""
    fb = (np.uint32(0x3F800000) + np.arange(1 << 23, dtype=np.uint32)).astype(np.uint32)
    m = fb.view(np.float32).astype(np.float64)
    r = (np.uint32(0x7EF311C3) - fb).astype(np.uint32).view(np.float32).astype(np.float64)
    last = float(np.uint32(lower._DIV_LAST).view(np.float32))
    for i in range(3):
        r = _trunc32(r * _trunc32((2.0 if i < 2 else last) - _trunc32(m * r)))
    assert (r * (m + 2.0 ** -23)).max() <= 1 - 2.5 * 2.0 ** -23       # (products exact in float64)
    assert (r * m).min() >= 1 - 5.51 * 2.0 ** -23
    bits = r.astype(np.float32).view(np.uint32)
    pick = rng.integers(0, 1 << 23, 4096)
    for e in (0, 1, 8, 23, 24, 29, 31):
        scaled = (fb[pick].astype(np.int64) + (e << 23)).astype(np.uint32)
        np.testing.assert_array_equal(_recip(scaled), (bits[pick].astype(np.int64) - (e << 23)).astype(np.uint32))


def _step(r1, ub, rcp, bhx, fix):
    """lower._div_step"""
    fr = _a("itof", r1)
    if fix == "abs":
        fr = _a("and", fr, 0x7FFFFFFF)
    elif fix == "u2f":
        fr = np.where(r1 >> 31 != 0, _a("fadd", fr, 0x4F7FFFFF), fr).astype(np.uint32)
    dq = _a("ftoi", _m("fmul", fr, rcp))
    r2 = _a("sub", r1, _m("mul24", dq, ub))
    if bhx is not None:
        r2 = _a("sub", r2, _m("mul24", _m("mul24", dq, 1 << 16), bhx))
    return dq, r2


def _model(a, b, signed, ua_max=None, ub_min=1, ub_max=None, const=False, paths=None):
    """lower._idiv_fast / _udivmod_fast with the path chosen as there from the
    magnitude bounds ua_max >= |a|, ub_min <= |b| <= ub_max (default: nothing
    known); const: b is a compile-time constant (every element the same value).
    The path taken is added to `paths`."""
    a, b = np.asarray(a, np.uint32), np.asarray(b, np.uint32)
    z = np.zeros_like(a)
    ua_max = (1 << 31 if signed else M32) if ua_max is None else ua_max
    ub_max = (1 << 31 if signed else M32) if ub_max is None else ub_max
    if signed:
        ua, ub = _a("max", a, _a("sub", z, a)), _a("max", b, _a("sub", z, b))
    else:
        ua, ub = a, b
    if const:
        rcp = np.full_like(ub, np.float32(lower._recip_down(int(ub[0]))).view(np.uint32))
    else:
        fb = _a("itof", ub)
        if not signed and ub_max >= 1 << 31:
            fb = np.where(ub >> 31 != 0, _a("fadd", fb, 0x4F800000), fb).astype(np.uint32)
        rcp = _recip(fb)
    bhx = _m("mul24", _a("shr", ub, 24), 256) if ub_max >= 1 << 24 else None
    one_step = ua_max < 1 << 31 and ua_max < ub_min << 20 and (ub_max < 1 << 24 or ua_max < 1 << 24)
    if one_step or ua_max < 1 << 24:
        q = _a("ftoi", _m("fmul", _a("itof", ua), rcp))
        r2 = _a("sub", ua, _m("mul24", q, ub))
        if not one_step:
            dq, r2 = _step(r2, ub, rcp, None, None)
            q = _a("add", q, dq)
    else:
        qh = _a("ftoi", _m("fmul", _a("itof", _a("shr", ua, 8)), rcp))
        r1 = _a("sub", ua, _m("mul24", _m("mul24", qh, ub), 256))
        dq, r2 = _step(r1, ub, rcp, bhx, None if ua_max < 1 << 31 else "abs" if signed else "u2f")
        q = _a("add", _m("mul24", qh, 256), dq)
    if paths is not None:
        paths.add("idiv_one_step" if one_step else "idiv_short" if ua_max < 1 << 24 else "idiv_full")
    if signed or ub_max <= 1 << 31:
        ge = _a("sub", r2, ub) >> 31 == 0
    else:
        k = np.uint32(0x80000000)
        ge = _a("min", r2 ^ k, ub ^ k) == ub ^ k
    q = np.where(ge, q + np.uint32(1), q).astype(np.uint32)
    r2 = np.where(ge, _a("sub", r2, ub), r2).astype(np.uint32)
    if signed:
        q = np.where((a ^ b) >> 31 != 0, _a("sub", z, q), q).astype(np.uint32)
        r2 = np.where(a >> 31 != 0, _a("sub", z, r2), r2).astype(np.uint32)
    return q, r2


def _ref(a, b, signed):
    """C semantics (INT_MIN / -1 wraps); uint32 bit patterns."""
    a, b = np.asarray(a, np.uint32), np.asarray(b, np.uint32)
    if signed:
        x, y = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
        q = np.sign(x) * np.sign(y) * (np.abs(x) // np.abs(y))
        r = x - q * y
    else:
        x, y = a.astype(np.int64), b.astype(np.int64)
        q, r = x // y, x % y
    return (q & M32).astype(np.uint32), (r & M32).astype(np.uint32)


def _edges(dense=True):
    v = {0, 1, 2, 3, 5, 7, 10, 255, 256, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000001}
    for e in range(33) if dense else (8, 16, 22, 23, 24, 25, 29, 30, 31):
        for d in (-3, -2, -1, 0, 1, 2, 3) if dense else (-1, 0, 1):
            v.add(((1 << e) + d) & M32)
            v.add((-(1 << e) + d) & M32)
    if dense:
        for e in (23, 24, 29, 31):
            for d in range(-40, 41):
                v.add(((1 << e) + d) & M32)
                v.add((-(1 << e) + d) & M32)
    return np.array(sorted(v), dtype=np.uint32)


def _random(n):
    """Dividends and divisors of every magnitude, plus suite-like divisors."""
    x = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    x = np.where(rng.random(n) < 0.5, x, x >> rng.integers(0, 32, n).astype(np.uint32)).astype(np.uint32)
    y = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32) >> rng.integers(0, 32, n).astype(np.uint32)
    y = np.where(rng.random(n) < 0.2, (y >> 8) | 1, y).astype(np.uint32)
    neg = rng.random((2, n)) < 0.5
    x = np.where(neg[0], -x.astype(np.int64) & M32, x).astype(np.uint32)
    y = np.where(neg[1], -y.astype(np.int64) & M32, y).astype(np.uint32)
    return x, np.where(y == 0, 1, y).astype(np.uint32)


@pytest.mark.parametrize("signed", [True, False])
def test_model_exact(signed):
    """Every pair of ~900 edge values (0, +-1, INT_MIN, INT_MAX, 2**k + -3..3, a band
    around 2**23, 2**24, 2**29, 2**31) and 2**18 random pairs."""
    e = _edges()
    x, y = (g.ravel() for g in np.meshgrid(e, e))
    x, y = x[y != 0], y[y != 0]
    rx, ry = _random(1 << 18)
    x, y = np.concatenate([x, rx]), np.concatenate([y, ry])
    q, r = _model(x, y, signed)
    want_q, want_r = _ref(x, y, signed)
    bad = np.nonzero((q != want_q) | (r != want_r))[0]
    assert bad.size == 0, [(hex(x[i]), hex(y[i])) for i in bad[:5]]


def _ranged(n, signed, ua_max, ub_min, ub_max):
    """Pairs with |a| <= ua_max and ub_min <= |b| <= ub_max (uint32 a, b when
    unsigned): uniform and log-uniform sizes, the bounds, and dividends next to
    multiples of the divisor (k b - 1, k b, k b + 1, half of them at the largest k)."""
    ua_max = (1 << 31 if signed else M32) if ua_max is None else ua_max
    ub_max = (1 << 31 if signed else M32) if ub_max is None else ub_max

    def mags(lo, hi):
        u = rng.integers(lo, hi + 1, n)
        g = np.clip(rng.integers(0, 1 << 33, n) >> rng.integers(0, 34, n), lo, hi)
        v = np.where(rng.random(n) < 0.5, u, g)
        v[:4] = np.clip([lo, lo + 1, hi - 1, hi], lo, hi)
        return v

    ma, mb = mags(0, ua_max), mags(ub_min, ub_max)
    kmax = ua_max // mb
    k = np.where(rng.random(n) < 0.5, kmax, (rng.random(n) * (kmax + 1)).astype(np.int64))
    near = np.clip(k * mb + rng.integers(-1, 2, n), 0, ua_max)
    ma = np.where(rng.random(n) < 0.4, near, ma)
    if signed:
        ma = np.where(rng.random(n) < 0.5, -ma, ma)
        mb = np.where(rng.random(n) < 0.5, -mb, mb)
    return (ma & M32).astype(np.uint32), (mb & M32).astype(np.uint32)


# (signed, ua_max, ub_min, ub_max, path): the bounds _idiv_fast derives from ranges
RANGED = [
    (True, None, 1, None, "idiv_full"),                     # nothing known
    (False, None, 1, None, "idiv_full"),
    (True, 2 ** 31 - 1, 2 ** 11 + 1, 2 ** 24 - 1, "idiv_one_step"),   # quotients up to 2**20
    (False, 2 ** 31 - 1, 2 ** 11 + 1, 2 ** 24 - 1, "idiv_one_step"),
    (True, 2 ** 20 - 1, 1, 2 ** 24 - 1, "idiv_one_step"),
    (True, 2 ** 24 - 1, 16, 2 ** 31, "idiv_one_step"),      # exact dividend, any divisor
    (False, 2 ** 24 - 1, 16, M32, "idiv_one_step"),
    (True, 2 ** 24 - 1, 1, 2 ** 31, "idiv_short"),
    (False, 2 ** 24 - 1, 1, M32, "idiv_short"),
    (True, 2 ** 31 - 1, 1, 2 ** 31, "idiv_full"),           # no r1 = 2**31 fix
    (True, 2 ** 31, 1, 2 ** 24 - 1, "idiv_full"),           # no high divisor bits
    (False, M32, 1, 2 ** 31, "idiv_full"),                  # unsigned, sign comparison
    (False, M32, 1, 2 ** 31 - 1, "idiv_full"),              # unsigned, plain itof
    (False, 2 ** 31 - 1, 1, M32, "idiv_full"),              # unsigned, no r1 fix
]


@pytest.mark.parametrize("cfg", RANGED)
def test_model_ranged_paths(cfg):
    """Each path and variant _udivmod_fast picks from known bounds, on 2**16 pairs
    inside the bounds (many at the largest quotient, next to a multiple)."""
    signed, ua_max, ub_min, ub_max, path = cfg
    x, y = _ranged(1 << 16, signed, ua_max, ub_min, ub_max)
    paths = set()
    q, r = _model(x, y, signed, ua_max, ub_min, ub_max, paths=paths)
    assert paths == {path}
    want_q, want_r = _ref(x, y, signed)
    bad = np.nonzero((q != want_q) | (r != want_r))[0]
    assert bad.size == 0, [(hex(x[i]), hex(y[i])) for i in bad[:5]]


CONSTS = [(True, c, None) for c in (-1, 3, 7, -10, 1000, 2 ** 23 + 1, -(2 ** 24 - 3), 123456789,
                                    2 ** 31 - 1, -2 ** 31 + 1, -2 ** 31)] + \
         [(False, c, None) for c in (3, 10, 1000, 2 ** 24 + 1, 2 ** 31 - 1, 2 ** 31 + 12345, M32, 0x80000001)] + \
         [(True, 1000, 2 ** 16 - 1), (True, -7, 2 ** 20 - 1), (False, 100003, 2 ** 31 - 1)]


@pytest.mark.parametrize("signed,c,ua_max", CONSTS)
def test_model_constant_divisor(signed, c, ua_max):
    """A constant divisor uses the largest float32 <= 1 / |c| (_recip_down)."""
    ub = abs(c) if signed else c & M32
    x, _ = _ranged(1 << 16, signed, ua_max, ub, ub)
    y = np.full(len(x), c & M32, np.uint32)
    q, r = _model(x, y, signed, ua_max, ub, ub, const=True)
    want_q, want_r = _ref(x, y, signed)
    bad = np.nonzero((q != want_q) | (r != want_r))[0]
    assert bad.size == 0, [(hex(x[i]), hex(y[i])) for i in bad[:5]]


def test_recip_down():
    for b in (1, 3, 7, 10, 255, 2 ** 23 + 1, 2 ** 24 - 3, 2 ** 31 - 1, 2 ** 31, 2 ** 31 + 1, 2 ** 32 - 1):
        r = lower._recip_down(b)
        assert r * b <= 1 < float(np.nextafter(np.float32(r), np.float32(2))) * b


# -- compiled kernels in the strict emulator --------------------------------------

@piton.jit
def divmod_k(x_ptr, y_ptr, o_ptr, n, U: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1)
    y = tl.load(y_ptr + offs, mask=m, other=1)
    if U:
        x = x.to(tl.uint32, bitcast=True)
        y = y.to(tl.uint32, bitcast=True)
    tl.store(o_ptr + offs, (x // y).to(tl.int32, bitcast=True), mask=m)
    tl.store(o_ptr + n + offs, (x % y).to(tl.int32, bitcast=True), mask=m)


def _run_divmod(x, y, signed, block=64):
    n = len(x)
    o = np.zeros(2 * n, np.int32)
    divmod_k[(piton.cdiv(n, block),)](x.view(np.int32), y.view(np.int32), o, n, U=int(not signed), BLOCK=block)
    q, r = _ref(x, y, signed)
    np.testing.assert_array_equal(o[:n].view(np.uint32), q)
    np.testing.assert_array_equal(o[n:].view(np.uint32), r)


@pytest.mark.parametrize("signed", [True, False])
def test_edge_pairs(fast, signed):
    e = _edges(dense=False)
    x, y = (g.ravel() for g in np.meshgrid(e, e))
    _run_divmod(x[y != 0], y[y != 0], signed)


@pytest.mark.parametrize("signed", [True, False])
def test_random_pairs(fast, signed):
    _run_divmod(*_random(4096), signed, block=256)


@piton.jit
def ranged_k(x_ptr, y_ptr, o_ptr, KIND: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    if KIND == 0:          # |a| < 2**20: one step
        a, b = x & 0xFFFFF, y
    elif KIND == 1:        # a < 2**24: unshifted two steps
        a, b = x & 0xFFFFFF, y
    elif KIND == 2:        # 0 <= a < 2**31: no fix for r1 = 2**31
        a, b = x & 0x7FFFFFFF, y
    elif KIND == 3:        # 1 <= b <= 2**23: no high divisor bits
        a, b = x, (y & 0x7FFFFF) + 1
    elif KIND == 4:        # constant divisor
        a, b = x, 7
    elif KIND == 5:        # constant divisor, short dividend: one step
        a, b = x & 0xFFFF, 1000
    elif KIND == 6:        # negative constant
        a, b = x, -10
    elif KIND == 7:        # constant above 2**24
        a, b = x, 123456789
    else:                  # INT_MIN constant
        a, b = x, -2147483648
    tl.store(o_ptr + offs, a // b)
    tl.store(o_ptr + BLOCK + offs, a % b)


PATHS = ["idiv_one_step", "idiv_short", "idiv_full", "idiv_full", "idiv_full", "idiv_one_step",
         "idiv_full", "idiv_full", "idiv_full"]


def ranged_inputs(kind, n=256):
    """ranged_k's inputs (random pairs and edge values) and its expected output."""
    x, y = _random(n)
    e = _edges(dense=False)[::3]
    x[:len(e)], y[:len(e)] = e, e[::-1]
    y = np.where(y == 0, 1, y).astype(np.uint32)
    xs, ys = x.view(np.int32).astype(np.int64), y.view(np.int32).astype(np.int64)
    a, b = {0: (xs & 0xFFFFF, ys), 1: (xs & 0xFFFFFF, ys), 2: (xs & 0x7FFFFFFF, ys),
            3: (xs, (ys & 0x7FFFFF) + 1), 4: (xs, 7), 5: (xs & 0xFFFF, 1000), 6: (xs, -10),
            7: (xs, 123456789), 8: (xs, -2 ** 31)}[kind]
    q, r = _ref(a & M32, np.broadcast_to(b, a.shape) & M32, True)
    return x.view(np.int32), y.view(np.int32), np.concatenate([q, r])


@pytest.mark.parametrize("kind", range(9))
def test_range_specialised_paths(fast, kind):
    x, y, want = ranged_inputs(kind)
    o = np.zeros(2 * len(x), np.int32)
    ck = ranged_k[(1,)](x, y, o, KIND=kind, BLOCK=len(x))
    assert [p for p in ("idiv_one_step", "idiv_short", "idiv_full") if p in ck.stats] == [PATHS[kind]]
    np.testing.assert_array_equal(o.view(np.uint32), want)


@piton.jit
def by_scalar_k(x_ptr, o_ptr, b, U: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if U:
        x = x.to(tl.uint32, bitcast=True)
        b = b.to(tl.uint32, bitcast=True)
    tl.store(o_ptr + offs, (x // b).to(tl.int32, bitcast=True))
    tl.store(o_ptr + BLOCK + offs, (x % b).to(tl.int32, bitcast=True))


@pytest.mark.parametrize("signed", [True, False])
def test_tile_by_scalar(fast, signed):
    """A uniform run-time divisor: one reciprocal for the whole tile, and x // b,
    x % b one division per register."""
    x = np.concatenate([_edges(dense=False), _random(128)[0]])[:128]
    for b in (2, 3, -1, -7, 2 ** 23 + 1, 2 ** 24 - 1, 2 ** 29 + 3, 2 ** 31 - 1, -2 ** 31, -2 ** 31 + 1, 1234567891):
        o = np.zeros(2 * len(x), np.int32)
        ck = by_scalar_k[(1,)](x.view(np.int32), o, b, U=int(not signed), BLOCK=len(x))
        assert ck.stats.get("idiv_full") == len(x) // 16
        q, r = _ref(x, np.full(len(x), b & M32, np.uint32), signed)
        np.testing.assert_array_equal(o.view(np.uint32), np.concatenate([q, r]), err_msg=str(b))


@piton.jit
def by_const_k(x_ptr, o_ptr, C: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x // C)
    tl.store(o_ptr + BLOCK + offs, x % C)


@pytest.mark.parametrize("c", [1, 2, 8, 2 ** 30, 3, 7, -1, -8, 1000, 2 ** 24 + 1, 2 ** 31 - 1, -2 ** 31])
def test_tile_by_constant(fast, c):
    """x // C and x % C share one division per register; 1 and powers of two
    take the shortcuts (no float estimate)."""
    x = np.concatenate([_edges(dense=False), _random(128)[0]])[:128]
    o = np.zeros(2 * len(x), np.int32)
    ck = by_const_k[(1,)](x.view(np.int32), o, C=c, BLOCK=len(x))
    pow2 = c > 0 and c & (c - 1) == 0
    assert sum(ck.stats.get(p, 0) for p in ("idiv_one_step", "idiv_short", "idiv_full")) == (0 if pow2 else len(x) // 16)
    q, r = _ref(x, np.full(len(x), c & M32, np.uint32), True)
    np.testing.assert_array_equal(o.view(np.uint32), np.concatenate([q, r]), err_msg=str(c))


@piton.jit
def scalar_k(o, a, b):
    tl.store(o, a // b)
    tl.store(o + 1, a % b)
    tl.store(o + 2, (a + 1) // b)


def test_scalar_division(fast):
    pairs = [(0, 3), (-1, 2), (7, -7), (-2 ** 31, -1), (-2 ** 31, -2 ** 31), (2 ** 31 - 1, -2 ** 31),
             (-2 ** 31, 2 ** 23 + 1), (2 ** 31 - 1, 2 ** 31 - 1), (1234567891, 2 ** 24), (-987654321, 2 ** 29 + 3),
             (2 ** 23 + 1, -3), (5, 2 ** 31 - 1)]
    for a, b in pairs:
        o = np.zeros(3, np.int32)
        scalar_k[(1,)](o, a, b)
        a1 = a + 1 if a < 2 ** 31 - 1 else -2 ** 31
        (q,), (r,) = _ref([a & M32], [b & M32], True)
        (q1,), _ = _ref([a1 & M32], [b & M32], True)
        assert list(o.view(np.uint32)) == [q, r, q1], (a, b)


@piton.jit
def regions_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    if n > 2:
        v = x // y
    else:
        v = x % y
    acc = v + x % y
    for i in range(n):
        acc += (x + i) // y - (x + i) % y
    tl.store(o_ptr + offs, acc + x // y)


@pytest.mark.parametrize("n", [1, 3])
def test_division_across_regions(fast, n):
    """// and % share a division only inside one region (if branches, loop body)."""
    x, y = _random(64)
    o = np.zeros(64, np.int32)
    regions_k[(1,)](x.view(np.int32), y.view(np.int32), o, n, BLOCK=64)
    with np.errstate(over="ignore"):
        q = lambda a: _ref(a, y, True)
        v = q(x)[0] if n > 2 else q(x)[1]
        acc = v + q(x)[1]
        for i in range(n):
            xi = (x.astype(np.int64) + i & M32).astype(np.uint32)
            acc = acc + q(xi)[0] - q(xi)[1]
        want = (acc + q(x)[0]).astype(np.uint32)
    np.testing.assert_array_equal(o.view(np.uint32), want)


@piton.jit
def index_split_k(o_ptr, n, N, BM: tl.constexpr, BN: tl.constexpr):
    offs = tl.program_id(0) * BM * BN + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(o_ptr + offs, (offs // N) * 3 + offs % N, mask=offs < n)


@piton.jit
def bcast_divmod_k(x_ptr, y_ptr, o_ptr, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    x = tl.load(x_ptr + rm)
    y = tl.load(y_ptr + rn)
    tl.store(o_ptr + rm[:, None] * BN + rn[None, :], x[:, None] // y[None, :] + x[:, None] % y[None, :])


def test_register_pressure_falls_back(fast):
    """Tiles that fit the registers only with the branching division (the
    straight-line one keeps the tile's operands, both results and the block's
    cached lane offsets live at once) compile with it instead of failing:
    offs // N and offs % N of a 16x16 tile, a broadcast 8x32 x // y + x % y."""
    n, N = 2 * 256 - 5, 37
    o = np.zeros(2 * 256, np.int32)
    ck = index_split_k[(2,)](o, n, N, BM=16, BN=16)
    assert ck.stats.get("fast_idiv_fallback") == 1
    offs = np.arange(n)
    np.testing.assert_array_equal(o[:n], offs // N * 3 + offs % N)
    assert (o[n:] == 0).all()
    x = rng.integers(-2 ** 31, 2 ** 31, 8).astype(np.int32)
    y = (rng.integers(1, 2 ** 31, 32) >> rng.integers(0, 31, 32)).astype(np.int32) * rng.choice([-1, 1], 32)
    y = np.where(y == 0, 7, y).astype(np.int32)
    o = np.zeros(8 * 32, np.int32)
    ck = bcast_divmod_k[(1,)](x, y, o, BM=8, BN=32)
    assert ck.stats.get("fast_idiv_fallback") == 1
    X, Y = (g.ravel() for g in np.meshgrid(x, y, indexing="ij"))
    q, r = _ref(X, Y, True)
    np.testing.assert_array_equal(o.view(np.uint32), q + r)


KERNELS = [divmod_k, ranged_k, by_scalar_k, by_const_k, scalar_k, regions_k, index_split_k, bcast_divmod_k]
IDIV_SIG = {"x_ptr": "*i32", "y_ptr": "*i32", "q_ptr": "*i32", "r_ptr": "*i32", "n": "i32"}


def test_quotient_and_remainder_share_one_division(fast):
    """x // y and x % y: one float estimate pair per register, as for x // y alone."""
    ck = piton.compiler.compile(idiv_kernel, IDIV_SIG, {"BLOCK": 64})
    words = ck.asm["qasm"]
    assert words.count("ftoi") == 2 * 4 and "div_full" not in words
    assert "fast_idiv_fallback" not in ck.stats


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("PITON_FAST_IDIV", raising=False)
    assert "div_full" in piton.compiler.compile(idiv_kernel, IDIV_SIG, {"BLOCK": 64}).asm["qasm"]
