"""PITON_MANDEL_FAST=1: `mandelbrot_fast` and the compiler's `fast_branches`.

Level 1 must give exactly the plain kernel's counts; the branch changes must
keep every kernel's results."""
import os
import sys

import numpy as np
import pytest

import piton
import piton.language as tl
from piton import compiler
from piton.qpu.emulator import _qpu_fadd, _qpu_fmul
from piton.runtime.launcher import run_emulator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import mandelbrot  # noqa: E402
from mandelbrot import TARGET, escape_counts  # noqa: E402

FOUR, QUARTER, ONE, TWO = (np.float32(v).view(np.uint32) for v in (4.0, 0.25, 1.0, 2.0))


@pytest.fixture
def fast(monkeypatch):
    """set(level) switches PITON_MANDEL_FAST; compiled kernels are not reused across levels."""
    def set_level(level):
        monkeypatch.setenv("PITON_MANDEL_FAST", str(level))
        monkeypatch.setattr(mandelbrot.mandelbrot_kernel, "_cache", {})
    yield set_level
    mandelbrot.mandelbrot_kernel._cache = {}


def counts(fast, level, *args, **kwargs):
    fast(level)
    return escape_counts(*args, **kwargs)


# -- exact float model of the QPU iteration (the emulator's float ops) ------------------

def _fadd(a, b):
    return np.ascontiguousarray(_qpu_fadd(a, b), np.float32).view(np.uint32)


def _fmul(a, b):
    return np.ascontiguousarray(_qpu_fmul(a, b), np.float32).view(np.uint32)


def _fsub(a, b):
    return _fadd(a, b ^ np.uint32(0x80000000))


def model_known_inside(cx, cy):
    """known_inside() with the QPU's float operations, on float32 bit patterns."""
    xq = _fsub(cx, QUARTER)
    yy = _fmul(cy, cy)
    q = _fadd(_fmul(xq, xq), yy)
    t = _fadd(_fmul(q, np.float32(2.0 / mandelbrot.CARDIOID_SHRINK).view(np.uint32)), xq)
    i32 = lambda v: v.view(np.int32)   # noqa: E731
    cardioid = (i32(t) | (i32(_fmul(t, t)) - i32(q))) >> 31
    xb = _fadd(cx, ONE)
    bulb = (i32(_fadd(_fmul(xb, xb), yy)) - np.int32(mandelbrot.BULB_R2_BITS)) >> 31
    return cardioid | bulb


def model_escape(cx, cy, iters):
    """Iterations before |z|^2 > 4 as the plain kernel computes them (iters: never)."""
    x = np.zeros(cx.shape, np.uint32)
    y = np.zeros(cx.shape, np.uint32)
    n = np.zeros(cx.shape, np.int32)
    alive = np.ones(cx.shape, bool)
    for _ in range(iters):
        xx, yy = _fmul(x, x), _fmul(y, y)
        alive &= (_fsub(FOUR, _fadd(xx, yy)) >> 31) == 0
        n += alive
        xy = _fmul(x, y)
        x, y = _fadd(_fsub(xx, yy), cx), _fadd(_fmul(xy, TWO), cy)
    return n


# -- level 1 gives the plain kernel's counts --------------------------------------------

def _view(center, zoom, width, height):
    step = np.float32(3.5 / zoom / 640)
    return (np.float32(center[0] - width / 2 * float(step)), np.float32(center[1] - height / 2 * float(step)),
            step, width, height)


VIEWS = {
    "whole set": (np.float32(-2.2), np.float32(-0.05 * 11.5), np.float32(0.05), 64, 24),
    "target zoom 1000": _view(TARGET, 1000, 64, 8),
    "target zoom 10": _view(TARGET, 10, 64, 8),
    "target zoom 10000": _view(TARGET, 10000, 64, 8),
    "cardioid edge": _view((-0.1, 0.65), 300, 64, 8),         # boundary crosses the tiles
    "cardioid cusp": _view((0.25, 0.0), 2000, 64, 8),
    "bulb edge": _view((-1.0, 0.25), 300, 64, 8),
    "all inside": _view((-0.1, 0.1), 50, 64, 8),              # every lane settled
    "outside": _view((1.0, 1.0), 10, 64, 8),                  # escapes in the first chunk
}


@pytest.mark.parametrize("view", list(VIEWS))
def test_level1_counts_match_plain_kernel(fast, view):
    x0, y0, step, w, h = VIEWS[view]
    plain = counts(fast, 0, x0, y0, step, w, h, 96, block=32, unroll=8, rows=8)
    np.testing.assert_array_equal(counts(fast, 1, x0, y0, step, w, h, 96, block=32, unroll=8, rows=8), plain)


@pytest.mark.parametrize("block,unroll,tile_h,rows", [(16, 8, 1, 1), (32, 4, 1, 2), (64, 8, 1, 1),
                                                      (16, 16, 2, 8), (32, 16, 1, 8), (8, 8, 1, 1)])
def test_level1_configurations(fast, block, unroll, tile_h, rows):
    x0, y0, step, w, h = VIEWS["whole set"]
    w = 40 if block <= 16 else w            # a partial last block where the grid allows
    plain = counts(fast, 0, x0, y0, step, w, h, 64, block=block, unroll=unroll, tile_h=tile_h, rows=rows)
    got = counts(fast, 1, x0, y0, step, w, h, 64, block=block, unroll=unroll, tile_h=tile_h, rows=rows)
    np.testing.assert_array_equal(got, plain)


@pytest.mark.parametrize("block,unroll,tile_h", [(32, 16, 1), (16, 16, 2)])
def test_level1_suite_and_demo_views(fast, block, unroll, tile_h):
    """perf/suite.py's emulator view (zoom 1000, 64x16, 128 iterations) and the
    patch the HDMI demo checks on boot (examples/mandelbrot_pi.py: zoom 100,
    64x8, 256 iterations), in the configurations both use."""
    suite = _view(TARGET, 1000, 64, 16) + (128,)
    step = np.float32(3.5 / 100.0 / 640)
    patch = (np.float32(TARGET[0] - 32 * float(step)), np.float32(TARGET[1] - 4 * float(step)), step, 64, 8, 256)
    for args in (suite, patch):
        plain = counts(fast, 0, *args, block=block, unroll=unroll, tile_h=tile_h, rows=8)
        got = counts(fast, 1, *args, block=block, unroll=unroll, tile_h=tile_h, rows=8)
        np.testing.assert_array_equal(got, plain)


def test_level1_palette_render(fast):
    """The colour path (palette lookup and the store) is the plain kernel's."""
    x0, y0, step, w, h = VIEWS["cardioid edge"]
    fast(0)
    want, _ = mandelbrot.render(x0, y0, step, w, h, 64, block=16, unroll=8)
    fast(1)
    got, _ = mandelbrot.render(x0, y0, step, w, h, 64, block=16, unroll=8)
    np.testing.assert_array_equal(got, want)


def test_known_inside_matches_float_model(fast):
    """The kernel's test, read back through a tiny kernel, equals the model used below."""
    @piton.jit
    def k(out_ptr, x0, y0, step, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        cx = x0 + (i % 16).to(tl.float32) * step
        cy = y0 + (i // 16).to(tl.float32) * step
        tl.store(out_ptr + i, mandelbrot.known_inside(cx, cy))

    fast(1)
    for x0, y0, step in ((-0.80, 0.05, 0.00625), (0.20, 0.0, 0.0005), (-1.26, -0.01, 0.0011), (-0.2, 0.62, 0.004)):
        out = np.zeros(256, np.int32)
        k[(1,)](out, np.float32(x0), np.float32(y0), np.float32(step), BLOCK=256)
        s = np.float32(step).view(np.uint32)
        cx = _fadd(np.float32(x0).view(np.uint32), _fmul(np.float32(np.arange(256) % 16).view(np.uint32), s))
        cy = _fadd(np.float32(y0).view(np.uint32), _fmul(np.float32(np.arange(256) // 16).view(np.uint32), s))
        np.testing.assert_array_equal(out, model_known_inside(cx, cy))


def test_known_inside_points_never_escape():
    """The assumption level 1 rests on, checked with the QPU's float arithmetic:
    points known_inside() accepts close to the edges of both components (and
    at the cusp) stay bounded for 2048 iterations."""
    rng = np.random.default_rng(5)
    n = 4000
    r = 1.0 - 0.02 * rng.uniform(0, 1, n) ** 2      # multipliers crowding toward 1
    t = rng.uniform(-np.pi, np.pi, n)
    mu = r * np.exp(1j * t)
    cusp = 0.25 - rng.uniform(0, 1e-3, n) + 1j * rng.uniform(-1e-4, 1e-4, n)
    c = np.concatenate([mu / 2 - mu ** 2 / 4, mu / 4 - 1, cusp])
    cx = c.real.astype(np.float32).view(np.uint32)
    cy = c.imag.astype(np.float32).view(np.uint32)
    accepted = model_known_inside(cx, cy) != 0
    assert accepted.sum() > 2000
    assert (model_escape(cx[accepted], cy[accepted], 2048) == 2048).all()


def test_known_inside_rejects_outside_points():
    c = np.array([0.2501 + 0j, -0.7501 + 0j, -1.2501 + 0j, 0.38 + 0.2j, -0.75 + 0.01j, 2.0 + 0j])
    cx = c.real.astype(np.float32).view(np.uint32)
    cy = c.imag.astype(np.float32).view(np.uint32)
    assert not model_known_inside(cx, cy).any()


def test_level1_does_less_work(fast):
    """Settled lanes and the early exit skip work the plain kernel does."""
    x0, y0, step, w, h = VIEWS["target zoom 10"]
    work = {}
    for level in (0, 1):
        fast(level)
        ck = mandelbrot.mandelbrot_kernel[(w // 32, h // 8)](
            np.zeros(w * h, np.uint32), np.arange(97, dtype=np.uint32), x0, y0, step, w, w, 96,
            BLOCK=32, UNROLL=8, ROWS=8, TILE_H=1)
        work[level] = ck.last_stats["instructions"]
    assert work[1] < 0.6 * work[0]


def test_switch_values(monkeypatch):
    words = {}
    for value in (None, "", "0", "1"):
        if value is None:
            monkeypatch.delenv("PITON_MANDEL_FAST", raising=False)
        else:
            monkeypatch.setenv("PITON_MANDEL_FAST", value)
        words[value] = compiler.compile(mandelbrot.mandelbrot_kernel, mandelbrot.SIGNATURE,
                                        dict(BLOCK=32, UNROLL=8, ROWS=8, TILE_H=1), grid=(2, 2)).words
    assert words[""] == words[None] == words["0"] != words["1"]
    monkeypatch.setenv("PITON_MANDEL_FAST", "")
    assert mandelbrot.fast_level() == 0
    monkeypatch.setenv("PITON_MANDEL_FAST", "1")
    assert mandelbrot.fast_level() == 1
    monkeypatch.setenv("PITON_MANDEL_FAST", "yes")
    with pytest.raises(ValueError):
        mandelbrot.fast_level()


# -- fast_branches: lane-flag conditions and early loop exits ---------------------------

def _run(monkeypatch, on, kernel, signature, constants, values, grid=(1,)):
    monkeypatch.setenv("PITON_MANDEL_FAST", "1" if on else "0")
    ck = compiler.compile(kernel, signature, constants, grid=grid)
    vals = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    stats = run_emulator(ck, grid, vals)
    return ck, vals, stats


@piton.jit
def cond_k(x_ptr, out_ptr, c, OP: tl.constexpr, RED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    m = tl.max(x) if RED == 0 else tl.min(x)
    r = 0
    if OP == 0:
        if m > c:
            r = 1
    elif OP == 1:
        if m >= c:
            r = 1
    elif OP == 2:
        if m < c:
            r = 1
    elif OP == 3:
        if m <= c:
            r = 1
    elif OP == 4:
        if c < m:
            r = 1
    else:
        if c >= m:
            r = 1
    tl.store(out_ptr + tl.program_id(0) + tl.arange(0, 1), tl.zeros([1], tl.int32) + r)


IMIN, IMAX = -2 ** 31, 2 ** 31 - 1


@pytest.mark.parametrize("red", [0, 1])
@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("block", [8, 16, 32, 64])
def test_lane_conditions_match_reductions(monkeypatch, red, op, block):
    rows = [np.full(block, IMIN), np.full(block, IMAX), np.arange(block) - 5, -np.arange(block),
            np.where(np.arange(block) == block - 1, IMAX, IMIN), np.where(np.arange(block) == 3, IMIN, 7)]
    x = np.concatenate(rows).astype(np.int32)
    for c in (IMIN, -1, 0, 7, IMAX):
        want = None
        for on in (False, True):
            out = np.full(len(rows), -9, np.int32)
            ck, vals, _ = _run(monkeypatch, on, cond_k, {"x_ptr": "*i32", "out_ptr": "*i32", "c": "i32"},
                               dict(OP=op, RED=red, BLOCK=block), {"x_ptr": x, "out_ptr": out, "c": c},
                               grid=(len(rows),))
            if on:
                np.testing.assert_array_equal(vals["out_ptr"], want, err_msg=f"c={c}")
                if block % 16 == 0:
                    assert " >> " not in ck.asm["qasm"]          # no lane rotations left
            want = vals["out_ptr"]
        ref = [{0: lambda m: m > c, 1: lambda m: m >= c, 2: lambda m: m < c, 3: lambda m: m <= c,
                4: lambda m: c < m, 5: lambda m: c >= m}[op](int(r.max() if red == 0 else r.min())) for r in rows]
        np.testing.assert_array_equal(want, np.array(ref, np.int32))


@piton.jit
def countdown_k(x_ptr, out_ptr, n, GUARD_IV: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i)
    trips = 0
    for k in range(n):
        if GUARD_IV:
            go = tl.max(v) < k           # false, then true: no early exit (depends on k)
        else:
            go = tl.max(v) > 0
        if go:
            v = v - 1
            trips += 1
    tl.store(out_ptr + i, v)
    tl.store(out_ptr + BLOCK + tl.arange(0, 1), tl.zeros([1], tl.int32) + trips)


@pytest.mark.parametrize("guard_iv", [0, 1])
@pytest.mark.parametrize("n", [0, 1, 5, 30, 60])
def test_guarded_loop_exit(monkeypatch, guard_iv, n):
    x = (np.arange(32) % 7 + 3).astype(np.int32)     # max 9
    results = []
    for on in (False, True):
        out = np.zeros(33, np.int32)
        ck, vals, stats = _run(monkeypatch, on, countdown_k, {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32"},
                               dict(GUARD_IV=guard_iv, BLOCK=32), {"x_ptr": x, "out_ptr": out, "n": n})
        results.append((vals["out_ptr"], stats["instructions"]))
    np.testing.assert_array_equal(results[1][0], results[0][0])
    trips = min(n, 9) if not guard_iv else max(0, n - 10)
    assert results[0][0][32] == trips
    if not guard_iv and n >= 30:
        assert results[1][1] < results[0][1]       # stopped after the ninth trip


@piton.jit
def not_identity_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i)
    s = 0
    for k in range(n):
        if tl.max(v) > 0:
            v = v - 1
        else:
            s += 1                   # the else branch changes state: the loop must go on
    tl.store(out_ptr + i, v)
    tl.store(out_ptr + BLOCK + tl.arange(0, 1), tl.zeros([1], tl.int32) + s)


def test_else_with_effects_keeps_looping(monkeypatch):
    x = (np.arange(16) % 5).astype(np.int32)
    outs = []
    for on in (False, True):
        out = np.zeros(17, np.int32)
        _, vals, _ = _run(monkeypatch, on, not_identity_k, {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32"},
                          dict(BLOCK=16), {"x_ptr": x, "out_ptr": out, "n": 10})
        outs.append(vals["out_ptr"])
    np.testing.assert_array_equal(outs[1], outs[0])
    assert outs[1][16] == 6


@piton.jit
def reused_max_k(x_ptr, out_ptr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i)
    m = tl.max(x)
    r = m * 0
    if m > 3:                        # the reduction is also stored: it is still computed
        r = m + 1
    tl.store(out_ptr + tl.arange(0, 1), tl.zeros([1], tl.int32) + r)


def test_reduction_with_other_uses(monkeypatch):
    for x in (np.arange(32, dtype=np.int32), np.zeros(32, np.int32)):
        outs = []
        for on in (False, True):
            out = np.zeros(1, np.int32)
            _, vals, _ = _run(monkeypatch, on, reused_max_k, {"x_ptr": "*i32", "out_ptr": "*i32"},
                              dict(BLOCK=32), {"x_ptr": x, "out_ptr": out})
            outs.append(vals["out_ptr"])
        np.testing.assert_array_equal(outs[1], outs[0])
