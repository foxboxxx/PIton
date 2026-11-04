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


def run_bits(ck, grid, values, out="out_ptr"):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    run_emulator(ck, grid, values)
    return values[out].view(np.uint32).copy()


def both_ways(monkeypatch, kernel, sig, cons, block, values, n, out="out_ptr"):
    grid = ((n + block - 1) // block,)
    off = build(monkeypatch, False, kernel, sig, cons, grid)
    on = build(monkeypatch, True, kernel, sig, cons, grid)
    a, b = run_bits(off, grid, values, out), run_bits(on, grid, values, out)
    bad = np.flatnonzero(a != b)
    assert bad.size == 0, (f"{bad.size} outputs differ, first at {bad[0]}: "
                           f"0x{a[bad[0]]:08x} (off) != 0x{b[bad[0]]:08x} (on)")
    return off, on


def inputs(bits, n):
    x = np.resize(bits, n).view(np.float32)
    out = np.full(n + 64, 0x7FC00000, np.uint32).view(np.float32)
    return {"x_ptr": x, "out_ptr": out, "n": n}


# -- the switch --------------------------------------------------------------------------

def test_off_unless_one(monkeypatch):
    cons = {"OP": 2, "BLOCK": 256}
    words = []
    for value in (None, "", "0"):
        if value is None:
            monkeypatch.delenv("PITON_ISSUE", raising=False)
        else:
            monkeypatch.setenv("PITON_ISSUE", value)
        words.append(pcompile(sfu_kernel, SIG, cons, grid=(16,)).words)
    assert words[0] == words[1] == words[2]
    monkeypatch.setenv("PITON_ISSUE", "1")
    on = pcompile(sfu_kernel, SIG, cons, grid=(16,))
    assert on.stats["cse"] >= 0 and len(on.program.instrs) < len(words[0])


# -- same bits ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", [0, 1, 2, 3, 4, 5])
def test_sfu_kernel_bits(monkeypatch, op):
    """exp2 log2 1/x rsqrt exp sqrt, rerolled (BLOCK 256) with a ragged tail."""
    n = 4096 - 37
    off, on = both_ways(monkeypatch, sfu_kernel, SIG, {"OP": op, "BLOCK": 256}, 256, inputs(specials(), n), n)
    assert len(on.program.instrs) <= len(off.program.instrs) + 8


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
    assert len(on.program.instrs) < len(off.program.instrs)


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


def test_cse_shares_abs(monkeypatch):
    """recip's keep_nan and the quotient's checks both take |y|: computed once."""
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    tile = frontend.generate(div_kernel.fn, sig, {"BLOCK": 64}, name=None)
    ranges = passes.optimize(tile, (15, 0, 0))
    mf = Lowering(tile, ranges, {"num_stages": 2, "store_slots": True, "issue": True}).run()
    stats = opt.run(mf, issue=True)
    assert stats["cse"] >= 4                  # one per register
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
