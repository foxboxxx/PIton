"""Issue efficiency (PITON_ISSUE=1): backend/lower.py "Issue", presched.py "Flag groups".

With the switch on, the math kernels must produce the same bits as the default
build for every input, special values included; every kernel here runs both ways
in the strict emulator and the outputs are compared bit for bit.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton.backend import opt, presched  # noqa: E402
from piton.backend.lower import Lowering  # noqa: E402
from piton.backend.mir import VReg  # noqa: E402
from piton.compiler import compile as pcompile  # noqa: E402
from piton import frontend, passes, reroll  # noqa: E402
from piton.qpu import emulator  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from kernels import softmax_kernel  # noqa: E402
from primitive_kernels import log_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402

rng = np.random.default_rng(41)
SIG = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}


def specials() -> np.ndarray:
    """Bit patterns of every special class, the ends of each kernel's ranges
    (exp overflow / underflow, recip's 2**126 limit) and random patterns."""
    fixed = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF, 0x00400000,
             0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7E7FFFFF, 0x7E800000, 0x7EC00000,
             0xFE800000, 0x7F000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
             0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001, 0xFFFFFFFF, 0x3F000000, 0x3FB504F3]
    near = np.array([88.0, 88.72283, 88.7229, 89.0, 89.5, -87.3, -103.9, -104.0, -104.5, 127.9, 128.0,
                     128.5, -149.0, -150.0, -151.0, 1e-38, 1.17549e-38, 2 ** 126, 2 ** 126 * 1.999, 0.5,
                     1.4142135, 0.70710677, 1e30, -1e30, 3.0e38, -3.0e38], np.float32).view(np.uint32)
    normals = (rng.integers(1, 255, 2000, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, 2000, dtype=np.uint32)
    normals |= rng.integers(0, 2, 2000, dtype=np.uint32) << 31
    rand = rng.integers(0, 2 ** 32, 1500, dtype=np.uint64).astype(np.uint32)
    small = rng.uniform(-8, 8, 500).astype(np.float32).view(np.uint32)
    bits = np.concatenate([np.array(fixed, np.uint32), near, normals, rand, small])
    return bits


def build(monkeypatch, issue, kernel, sig, cons, grid):
    monkeypatch.setenv("PITON_ISSUE", "1" if issue else "0")
    bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)
    return pcompile(kernel, sig, cons, grid=bound)


def run_bits(ck, grid, values, out="out_ptr", executed=None):
    """Output bits; `executed` (a list) gets the number of instructions executed."""
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    v3d = emulator.V3D()
    v3d.pc_counts = [0] * len(ck.program.instrs)
    run_emulator(ck, grid, values, v3d=v3d)
    if executed is not None:
        executed.append(sum(v3d.pc_counts))
    return values[out].view(np.uint32).copy()


def both_ways(monkeypatch, kernel, sig, cons, block, values, n, out="out_ptr"):
    grid = ((n + block - 1) // block,)
    off = build(monkeypatch, False, kernel, sig, cons, grid)
    on = build(monkeypatch, True, kernel, sig, cons, grid)
    executed = []
    a, b = run_bits(off, grid, values, out, executed), run_bits(on, grid, values, out, executed)
    bad = np.flatnonzero(a != b)
    assert bad.size == 0, (f"{bad.size} outputs differ, first at {bad[0]}: "
                           f"0x{a[bad[0]]:08x} (off) != 0x{b[bad[0]]:08x} (on)")
    return executed


def inputs(bits, n):
    x = np.resize(bits, n).view(np.float32)
    out = np.full(n + 64, 0x7FC00000, np.uint32).view(np.float32)
    return {"x_ptr": x, "out_ptr": out, "n": n}


# -- the switch --------------------------------------------------------------------------

def test_on_unless_zero(monkeypatch):
    """On by default (accepted on the Pi 2026-11-04); "" or "0" turn it off."""
    cons = {"OP": 4, "BLOCK": 256}
    words = []
    for value in ("", "0"):
        monkeypatch.setenv("PITON_ISSUE", value)
        words.append(pcompile(sfu_kernel, SIG, cons, grid=(16,)).words)
    assert words[0] == words[1]
    on = []
    for value in (None, "1"):
        if value is None:
            monkeypatch.delenv("PITON_ISSUE", raising=False)
        else:
            monkeypatch.setenv("PITON_ISSUE", value)
        on.append(pcompile(sfu_kernel, SIG, cons, grid=(16,)))
    assert on[0].words == on[1].words
    assert on[0].stats["cse"] >= 0 and len(on[0].program.instrs) < len(words[0])


# -- same bits ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", [0, 1, 2, 3, 4, 5])
def test_sfu_kernel_bits(monkeypatch, op):
    """exp2 log2 1/x rsqrt exp sqrt, rerolled (BLOCK 256) with a ragged tail."""
    n = 4096 - 37
    off, on = both_ways(monkeypatch, sfu_kernel, SIG, {"OP": op, "BLOCK": 256}, 256, inputs(specials(), n), n)
    assert on < off             # executed instructions


@pytest.mark.parametrize("op, lo, hi", [(4, 88.6, 89.1), (0, 127.9, 128.1), (4, -104.1, -103.0), (0, -150.1, -149.0)])
def test_exp_range_ends_bits(monkeypatch, op, lo, hi):
    """exp / exp2 densely around their overflow thresholds (where the upper
    clamp used to apply) and their lower clamps."""
    a, b = np.array([lo, hi], np.float32).view(np.uint32)
    a, b = min(a, b), max(a, b)
    bits = np.arange(a, b, max(1, (b - a) // 4000), dtype=np.uint32)
    n = bits.size
    both_ways(monkeypatch, sfu_kernel, SIG, {"OP": op, "BLOCK": 256}, 256, inputs(bits, n), n)


def test_log_kernel_bits(monkeypatch):
    n = 2048 + 5
    both_ways(monkeypatch, log_kernel, SIG, {"BLOCK": 256}, 256, inputs(specials(), n), n)


@piton.jit
def const_div_kernel(x_ptr, out_ptr, n, A: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(out_ptr + offs, A / x, mask=m)


@pytest.mark.parametrize("a", [1.0, 2.0, -1.0, 0.75, 0.0, -0.0, 1e-40, 3.0e38, float("inf"), float("-inf")])
def test_constant_numerator_bits(monkeypatch, a):
    """a / x for a constant a: normal, negative, zero, denormal, huge, infinite."""
    n = 1024 - 3
    off, on = both_ways(monkeypatch, const_div_kernel, SIG, {"A": a, "BLOCK": 64}, 64, inputs(specials(), n), n)
    assert on < off


@piton.jit
def scalar_div_kernel(x_ptr, out_ptr, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, x / s, mask=m)


@pytest.mark.parametrize("s", [0x3FC00000, 0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000,
                               0x7F7FFFFF, 0x7E800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFF800001,
                               0xC0490FDB])
def test_uniform_denominator_bits(monkeypatch, s):
    """x / s for a runtime scalar s of every special class."""
    n = 512 + 9
    values = inputs(specials(), n)
    values["s"] = np.uint32(s).view(np.float32)
    sig = dict(SIG, s="fp32")
    both_ways(monkeypatch, scalar_div_kernel, sig, {"BLOCK": 64}, 64, values, n)


def test_softmax_bits(monkeypatch):
    rows, cols = 16, 64
    x = rng.uniform(-4, 4, rows * cols).astype(np.float32)
    x[3 * cols:4 * cols] = -np.inf                     # a row of -inf: 0 / 0
    x[5 * cols + 7] = np.inf
    x[6 * cols + 1] = np.uint32(0x7FC00000).view(np.float32)
    x[7 * cols:8 * cols] = np.float32(80.0)
    values = {"output_ptr": np.zeros(rows * cols, np.float32), "input_ptr": x, "input_row_stride": cols,
              "output_row_stride": cols, "n_cols": cols - 5}
    sig = {"output_ptr": "*fp32", "input_ptr": "*fp32", "input_row_stride": "i32",
           "output_row_stride": "i32", "n_cols": "i32"}
    grid = (rows,)
    off = build(monkeypatch, False, softmax_kernel, sig, {"BLOCK_SIZE": cols}, grid)
    on = build(monkeypatch, True, softmax_kernel, sig, {"BLOCK_SIZE": cols}, grid)
    a = run_bits(off, grid, values, "output_ptr")
    b = run_bits(on, grid, values, "output_ptr")
    np.testing.assert_array_equal(a, b)


def test_recip_zero_case_is_the_newton_result():
    """lower.py fdiv_const drops recip's zero case: three Newton steps from the
    seed give +-inf (the sign of b) for every +-0 and denormal b in the
    emulator's float model (denormals read as 0, products of 2**128 or more
    are inf, 0 * inf is +0)."""
    for sign in (0, 0x80000000):
        b = np.arange(0, 1 << 23, dtype=np.uint32) | np.uint32(sign)
        r = (np.uint64(0x7EF311C3) - b.astype(np.uint64)).astype(np.uint32)
        two = np.full(b.shape, 0x40000000, np.uint32)
        for _ in range(3):
            t = emulator.alu_mul("fmul", b, r)[0]
            e = emulator.alu_add("fsub", two, t)[0]
            r = emulator.alu_mul("fmul", r, e)[0]
        assert np.all(r == np.uint32(0x7F800000 | sign))


def test_scaled_newton_rsqrt():
    """lower.py rsqrt_core (PITON_ISSUE): the scaled steps give the same bits as
    the plain ones for positive normals of every exponent (the emulator's
    float model: round toward zero), 1.0 ... 2.0 densely."""
    e = np.repeat(np.arange(1, 255, dtype=np.uint32), 4096)
    m = rng.integers(0, 1 << 23, e.size, dtype=np.uint32)
    x = np.concatenate([(e << 23) | m, np.arange(0x3F800000, 0x40000000, 7, dtype=np.uint32),
                        np.array([0x00800000, 0x7F7FFFFF, 0x00FFFFFF, 0x7F000000], np.uint32)])

    def fmul(a, b):
        return emulator.alu_mul("fmul", a, b)[0]

    def fsub(a, b):
        return emulator.alu_add("fsub", a, b)[0]

    def full(v):
        return np.full(x.shape, np.float32(v).view(np.uint32), np.uint32)

    seed = (np.uint64(0x5F375A86) - (x >> np.uint32(1)).astype(np.uint64)).astype(np.uint32)
    r = seed
    for _ in range(3):
        t2 = fmul(fmul(x, r), r)
        r = fmul(r, fsub(full(1.5), fmul(t2, full(0.5))))
    s, c = seed, 1
    for _ in range(3):
        t2 = fmul(fmul(x, s), s)
        s = fmul(s, fsub(full(3.0 * c * c), t2))
        c = 2 * c ** 3
    assert c == 8192
    np.testing.assert_array_equal(fmul(s, full(1.0 / c)), r)


def test_log_exponent_split():
    """lower.py _log_core (PITON_ISSUE): (x - bits(sqrt(1/2))) >> 23 and
    x - (e << 23) give the e and m of the masked version for every mantissa
    (three exponents) and random positive normals."""
    m = np.arange(1 << 23, dtype=np.uint32)
    x = np.concatenate([(np.uint32(e) << np.uint32(23)) | m for e in (1, 127, 254)] +
                       [rng.integers(0x00800000, 0x7F800000, 1 << 20, dtype=np.uint32)])
    mb = (x & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
    ge = mb >= np.uint32(0x3FB504F3)
    half = emulator.alu_mul("fmul", mb, np.full(x.shape, 0x3F000000, np.uint32))[0]
    m_old = np.where(ge, half, mb)
    e_old = (x >> np.uint32(23)).astype(np.int64) - np.where(ge, 126, 127)
    t = (x.astype(np.int64) - 0x3F3504F3).astype(np.int32)
    e_new = t >> 23
    shifted = emulator.alu_mul("mul24", e_new.astype(np.uint32), np.full(x.shape, 0x3F800000, np.uint32))[0]
    m_new = (x.astype(np.int64) - shifted.astype(np.int64)).astype(np.uint32)
    np.testing.assert_array_equal(e_new, e_old)
    np.testing.assert_array_equal(m_new, m_old)


def test_v8adds_and_mul24_identities():
    """The byte-wise add and the mul24 shift that exp and log use instead of or / shl."""
    sgn = np.array([0, 0x80000000] * 8, np.uint32)
    k = np.full(16, 0x3F000000, np.uint32)
    assert np.array_equal(emulator.alu_mul("v8adds", sgn, k)[0], sgn | k)
    mant = rng.integers(0, 1 << 23, 16, dtype=np.uint32)
    k = np.full(16, 0x3F800000, np.uint32)
    assert np.array_equal(emulator.alu_add("v8adds", mant, k)[0], mant | k)
    x = np.array(list(range(-127, 129, 16)), np.int64).astype(np.uint32)
    one = np.full(16, 0x3F800000, np.uint32)
    assert np.array_equal(emulator.alu_mul("mul24", x, one)[0], x << np.uint32(23))


@piton.jit
def lane_cmp_kernel(x_ptr, out_ptr, n, k, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    if OP == 0:
        c = offs < k
    elif OP == 1:
        c = offs <= k
    elif OP == 2:
        c = k > offs
    elif OP == 3:
        c = k <= offs
    elif OP == 4:
        c = offs == k
    else:
        c = k != offs
    tl.store(out_ptr + offs, tl.where(c, x, -x), mask=m)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("block", [64, 256])
def test_lane_compare(monkeypatch, op, block):
    """lower.py lane_cmp_operands: an index compared with a uniform as
    offset - (uniform - lane), either side, every signed and equality
    predicate, ragged n, k inside, at the ends of and outside the index range."""
    n = 4 * block + 7
    x = rng.uniform(1, 2, n).astype(np.float32)
    idx = np.arange(n)
    ref_ops = [np.less, np.less_equal, lambda a, b: b > a, lambda a, b: b <= a, np.equal, lambda a, b: b != a]
    sig = dict(SIG, k="i32")
    grid = ((n + block - 1) // block,)
    off = build(monkeypatch, False, lane_cmp_kernel, sig, {"OP": op, "BLOCK": block}, grid)
    on = build(monkeypatch, True, lane_cmp_kernel, sig, {"OP": op, "BLOCK": block}, grid)
    for k in (-2 ** 31, -1, 0, 5, block - 1, block, 2 * block + 3, n - 1, n, n + 40, 2 ** 31 - 1):
        values = {"x_ptr": x, "out_ptr": np.zeros(n + 16, np.float32), "n": n, "k": k}
        executed = []
        a = run_bits(off, grid, values, executed=executed)
        b = run_bits(on, grid, values, executed=executed)
        np.testing.assert_array_equal(a, b)
        ref = np.where(ref_ops[op](idx, k), x, -x)
        np.testing.assert_array_equal(b[:n], ref.view(np.uint32))
        assert executed[1] < executed[0]


@piton.jit
def other_kernel(x_ptr, out_ptr, n, CASE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=2.0)
    y = tl.where(x > 1.5, x * 3.0, -x)
    if CASE == 0:
        tl.store(out_ptr + offs, y, mask=m)                     # dead: only lanes the store skips
    elif CASE == 1:
        tl.store(out_ptr + offs, y, mask=offs < n + 3)          # another mask
    elif CASE == 2:
        tl.store(out_ptr + offs, y)                             # no mask
    elif CASE == 3:
        tl.store(out_ptr + offs, y - tl.max(y, axis=0), mask=m)     # a reduction reads every lane
    elif CASE == 4:
        tl.store(out_ptr + offs, x, mask=(x > 1.5) & m)         # the value decides the mask
    else:
        tl.store(out_ptr + offs, y, mask=m)
        tl.store(out_ptr + offs + 1, y, mask=offs + 1 < n)      # also under another mask


@pytest.mark.parametrize("case", range(6))
def test_drop_dead_other(monkeypatch, case):
    """passes.drop_dead_other: only where every use ends in a store under the
    load's own mask; the outputs keep their bits either way."""
    tile = frontend.generate(other_kernel.fn, SIG, {"CASE": case, "BLOCK": 64}, name=None)
    assert passes.drop_dead_other(tile) == (1 if case == 0 else 0)
    n = 4 * 64 - 5
    x = np.resize(np.array([0.5, 1.0, 1.75, 2.0, -3.0, 4.0], np.float32), n)
    values = {"x_ptr": x, "out_ptr": np.full(n + 64, np.nan, np.float32), "n": n}
    both_ways(monkeypatch, other_kernel, SIG, {"CASE": case, "BLOCK": 64}, 64, values, n)


def test_drop_dead_other_keeps_int_loads():
    @piton.jit
    def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, 7 // tl.load(x_ptr + offs, mask=m, other=1), mask=m)
    tile = frontend.generate(k.fn, {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32"}, {"BLOCK": 64}, name=None)
    assert passes.drop_dead_other(tile) == 0


@piton.jit
def swapped_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(out_ptr + offs, tl.where(offs % 3 == 0, x + y, y + x), mask=m)
    tl.store(out_ptr + n + offs, tl.where(offs % 2 == 0, tl.minimum(x, y), tl.minimum(y, x)), mask=m)
    tl.store(out_ptr + 2 * n + offs, tl.where(offs % 2 == 1, tl.maximum(x, y), tl.maximum(y, x)), mask=m)
    tl.store(out_ptr + 3 * n + offs, tl.where(offs % 2 == 1, x * y, y * x), mask=m)


def test_cse_keeps_operand_order(monkeypatch):
    """opt.cse merges swapped operands only where the result is the same bits:
    inf + -inf is the second operand, fmin / fmax of -0 and +0 the first."""
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    pairs = [(np.inf, -np.inf), (-np.inf, np.inf), (0.0, -0.0), (-0.0, 0.0), (1e-40, -1e-40), (1.5, 2.5)]
    x = np.array([p[0] for p in pairs] * 40, np.float32)
    y = np.array([p[1] for p in pairs] * 40, np.float32)
    n = x.size
    grid = ((n + 63) // 64,)
    outs = []
    for issue in (False, True):
        ck = build(monkeypatch, issue, swapped_kernel, sig, {"BLOCK": 64}, grid)
        values = {"x_ptr": x, "y_ptr": y, "out_ptr": np.zeros(4 * n, np.float32), "n": n}
        outs.append(run_bits(ck, grid, values))
    np.testing.assert_array_equal(outs[0], outs[1])
    ref = np.concatenate([np.where(np.arange(n) % 3 == 0, y, x), np.where(np.arange(n) % 2 == 0, x, y),
                          np.where(np.arange(n) % 2 == 1, y, x)])    # the second / first operand
    sel = np.isinf(x) | (x == 0) | (np.abs(x) < 1e-38)
    for k in range(3):
        got = outs[1][k * n:(k + 1) * n][sel[:n]]
        want = ref[k * n:(k + 1) * n].view(np.uint32)[sel[:n]]
        if k == 0:
            assert np.array_equal(got[np.isinf(x[sel])], want[np.isinf(x[sel])])
        else:
            assert np.array_equal(got[~np.isinf(x[sel])], want[~np.isinf(x[sel])])


# -- flag groups -------------------------------------------------------------------------

def _check_order(before, after):
    """`after` is a valid reordering of `before`: every use sees the same
    definition, every flag read the same flag write, writes to one register
    stay in order, and side effects keep their order."""
    assert sorted(map(id, before)) == sorted(map(id, after))

    def reaching(seq):
        last, flags, out = {}, None, {}
        for ins in seq:
            out[id(ins)] = ({v: last.get(v) for v in ins.uses()}, flags if ins.reads_flags else None)
            for v in ins.defs():
                last[v] = id(ins)
            if ins.writes_flags:
                flags = id(ins)
        return out

    assert reaching(before) == reaching(after)
    effects = [i for i in before if not (presched._pure(i) or presched._flag_only(i))]
    assert [id(i) for i in after if not (presched._pure(i) or presched._flag_only(i))] == list(map(id, effects))
    for v in {v for i in before for v in i.defs()}:
        assert [id(i) for i in before if v in i.defs()] == [id(i) for i in after if v in i.defs()]


@pytest.mark.parametrize("op", [2, 4, 5])
def test_flag_group_order_is_valid(op):
    tile = frontend.generate(sfu_kernel.fn, SIG, {"OP": op, "BLOCK": 256}, name=None)
    reroll.run(tile, 64, queue=True)
    ranges = passes.optimize(tile, (15, 0, 0))
    mf = Lowering(tile, ranges, {"num_stages": 2, "store_slots": True, "issue": True}).run()
    opt.run(mf, issue=True)
    for b in mf.blocks:
        before = list(b.insts)
        after = presched.schedule_block(before, flags=True)
        _check_order(before, after)
    loop = [b for b in mf.blocks if b.loop_depth == 1][0]
    after = presched.schedule_block(loop.insts, flags=True)
    # the first compare of the second register's chain no longer waits for every
    # flag group of the first
    writers = [i for i, ins in enumerate(loop.insts) if ins.writes_flags]
    moved = [after.index(loop.insts[w]) for w in writers]
    assert moved != sorted(moved)


@piton.jit
def div_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    y = tl.load(y_ptr + offs, mask=m, other=1.0)
    tl.store(out_ptr + offs, x / y, mask=m)


def test_cse_leaves_no_repeats(monkeypatch):
    """Both loads compare the same mask: after CSE no pure instruction of a block
    repeats another on the same operands."""
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    tile = frontend.generate(div_kernel.fn, sig, {"BLOCK": 64}, name=None)
    ranges = passes.optimize(tile, (15, 0, 0))
    mf = Lowering(tile, ranges, {"num_stages": 2, "store_slots": True, "issue": True}).run()
    stats = opt.run(mf, issue=True)
    assert stats["cse"] >= 1
    defs = {}
    for b in mf.blocks:
        for ins in b.insts:
            for op in ins.ops:
                defs[id(op.dst)] = defs.get(id(op.dst), 0) + 1
    for b in mf.blocks:
        keys = [(op.op, tuple(id(s) if isinstance(s, VReg) else repr(s) for s in op.srcs))
                for ins in b.insts if ins.kind == "alu" and opt._pure_inst(ins) for op in ins.ops
                if defs[id(op.dst)] == 1 and all(defs.get(id(s), 1) == 1 for s in op.srcs)]
        assert len(keys) == len(set(keys))
    n = 256 + 3
    values = {"x_ptr": np.resize(specials(), n).view(np.float32),
              "y_ptr": np.resize(specials()[::-1].copy(), n).view(np.float32),
              "out_ptr": np.zeros(n, np.float32), "n": n}
    both_ways(monkeypatch, div_kernel, sig, {"BLOCK": 64}, 64, values, n)


# -- loops, 2-D tiles, integer views ---------------------------------------------------

def same_bits(monkeypatch, kernel, sig, cons, grid, runs, out="out_ptr"):
    """Both builds on every argument set of `runs`, bit for bit; returns the on build."""
    off = build(monkeypatch, False, kernel, sig, cons, grid)
    on = build(monkeypatch, True, kernel, sig, cons, grid)
    for values in runs:
        a, b = run_bits(off, grid, values, out), run_bits(on, grid, values, out)
        bad = np.flatnonzero(a != b)
        assert bad.size == 0, (f"{bad.size} outputs differ, first at {bad[0]}: "
                               f"0x{a[bad[0]]:08x} (off) != 0x{b[bad[0]]:08x} (on)")
    return on


@piton.jit
def loop_kernel(x_ptr, out_ptr, n, LB: tl.constexpr, UB: tl.constexpr, PID_LB: tl.constexpr,
                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(pid if PID_LB else LB, UB):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=1.0)
        acc += tl.exp(x * 0.01) + 1.0 / x
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


@pytest.mark.parametrize("pid_lb", [0, 1])
def test_loop_prefetch_index(monkeypatch, pid_lb):
    """lower.py LoopStream (PITON_ISSUE): the prefetch index min(iv + 1, ub - 1)
    of a pipelined loop with a constant bound, with masks against a runtime n
    (loop_lanes candidates), ragged and empty data and zero-trip loops."""
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    UB, BLOCK = 6, 64
    cons = {"LB": 0, "UB": UB, "PID_LB": pid_lb, "BLOCK": BLOCK}
    grid = (8,)                     # programs 6 and 7 skip the loop with PID_LB
    x = rng.uniform(0.5, 2, UB * BLOCK).astype(np.float32)
    runs = [{"x_ptr": x, "out_ptr": np.zeros(grid[0] * BLOCK, np.float32), "n": n}
            for n in (UB * BLOCK, UB * BLOCK - 5, 3 * BLOCK + 1, 0)]
    on = same_bits(monkeypatch, loop_kernel, sig, cons, grid, runs)
    mir = [line.split() for line in on.asm["mir"].splitlines() if line.strip()]
    assert any(w[0] == "min" and w[1].startswith("%ivp") for w in mir)


@piton.jit
def carried_div_kernel(x_ptr, s_ptr, out_ptr, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    s = tl.load(s_ptr)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(ITERS):
        acc += x / s
        s = s * 3.0 - 1.0
        acc += (x + 1.0) / s
    tl.store(out_ptr + offs, acc, mask=m)


def test_loop_carried_uniform_denominator(monkeypatch):
    """lower.py fdiv_uniform: the reciprocal and limits are per block and per
    register, so a loop-carried scalar divisor that changes every iteration
    (and within one) keeps its bits."""
    sig = {"x_ptr": "*fp32", "s_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    n = 64 * 3 - 5
    runs = [{"x_ptr": np.resize(specials(), n).view(np.float32), "s_ptr": np.array([s0], np.float32),
             "out_ptr": np.zeros(n + 64, np.float32), "n": n} for s0 in (0.75, 0.0, -1e-40, float("inf"))]
    same_bits(monkeypatch, carried_div_kernel, sig, {"ITERS": 3, "BLOCK": 64}, (3,), runs)


@piton.jit
def tile2d_kernel(x_ptr, out_ptr, M, N, stride, OP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptr + rm[:, None] * stride + rn[None, :], mask=m, other=-1.0)
    if OP == 0:
        y = x / tl.sum(tl.where(m, x, 0.0), axis=1)[:, None]
    else:
        y = tl.where(rn[None, :] >= N - 3, x, 2.0 / x)
    tl.store(out_ptr + rm[:, None] * stride + rn[None, :], y, mask=m)


@pytest.mark.parametrize("op", [0, 1])
def test_tile2d_bits(monkeypatch, op):
    """Row and column masks of a 2-D tile, a per-row denominator and lane
    compares against N - 3."""
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32", "stride": "i32"}
    stride = 70
    x = rng.uniform(-3, 3, 16 * stride + 64).astype(np.float32)
    x[::11] = np.resize(specials(), x[::11].size).view(np.float32)
    runs = [{"x_ptr": x, "out_ptr": np.zeros(16 * stride + 64, np.float32), "M": M, "N": N, "stride": stride}
            for M, N in ((13, 64), (5, 17))]
    same_bits(monkeypatch, tile2d_kernel, sig, {"OP": op, "BM": 4, "BN": 64}, (4,), runs)


@piton.jit
def int_view_kernel(x_ptr, out_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    if MODE == 0:
        q = (1 << 30) // x.to(tl.int32, bitcast=True)
    else:
        q = 1000 % x.to(tl.int32)
    tl.store(out_ptr + offs, q.to(tl.float32), mask=m)


@pytest.mark.parametrize("mode", [0, 1])
def test_drop_dead_other_through_int_views(monkeypatch, mode):
    """passes.drop_dead_other follows bitcasts and conversions to integers:
    masked lanes then divide by 0 (garbage in lanes the store skips, no hang)."""
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    tile = frontend.generate(int_view_kernel.fn, sig, {"MODE": mode, "BLOCK": 64}, name=None)
    assert passes.drop_dead_other(tile) == 1
    n = 4 * 64 - 13
    values = {"x_ptr": rng.uniform(1, 50, n).astype(np.float32), "out_ptr": np.zeros(n + 64, np.float32), "n": n}
    same_bits(monkeypatch, int_view_kernel, sig, {"MODE": mode, "BLOCK": 64}, ((n + 63) // 64,), [values])
