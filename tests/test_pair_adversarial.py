"""Adversarial checks of the pairing schedules (PITON_PAIR=1, sched.py
"Pairing") from review: every instruction the move to the mul ALU forms
behaves like its two ops and assembles to the same bits with vc4asm (r5,
conditions, both register files), special float values come out bit-identical
to the switch off, and the loop-entry padding leaves no latency hazard on any
control-flow path (taken or not, zero-trip loops included)."""
import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton import compiler  # noqa: E402
from piton.backend import sched  # noqa: E402
from piton.backend.mir import MInst, MOp  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from piton.qpu.emulator import V3D, Memory  # noqa: E402
from piton.qpu.isa import Imm, Instr  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from dot_tests import dot_loop, dot_tile  # noqa: E402

rng = np.random.default_rng(1105)


# ---------------------------------------------------------------------------
# The move to the mul ALU, instruction by instruction
# ---------------------------------------------------------------------------


def _node(op, dst, srcs, cond="always", unit="any"):
    return sched._node_for(0, MInst(ops=[MOp(op, dst, list(srcs), cond=cond, unit=unit)]))


_V3D = []


def _qpu(seed):
    if not _V3D:
        _V3D.append(V3D(num_qpus=1, mem=Memory(1 << 16)))
    q = _V3D[0].qpus[0]
    q.reset()
    r = np.random.default_rng(seed)

    def words():
        return r.integers(0, 2 ** 32, 16, dtype=np.uint64).astype(np.uint32)
    q.ra = [words() for _ in range(32)]
    q.rb = [words() for _ in range(32)]
    q.acc = [words() for _ in range(6)]
    q.flags = tuple(r.random(16) < .5 for _ in range(3))
    return q


def _execute(q, ins):
    q.last_written_now, q.last_acc_now, q.last_r5_now = set(), set(), False
    writes = []
    q.exec_alu(ins, writes)
    for dst, file, value, mask in writes:
        q.commit(dst, file, value, mask)
    return [a.copy() for a in q.ra + q.rb + q.acc]


# ".ifnc" is left out: format_instr prints condition 7 (C clear) that way, which
# vc4asm reads as N clear (5); no PIton build uses the carry conditions
_SWAP_GRID = list(itertools.product(
    [isa.RA[1], isa.RB[2], isa.R0, isa.R4, isa.R5, isa.ELEM_NUM, isa.QPU_NUM],      # move source
    [isa.RA[5], isa.RB[6], isa.R1, isa.R5QUAD, isa.R5REP],                          # move destination
    ["always", "ifz", "ifnn"],
    [("add", [isa.R2, isa.R3]), ("fadd", [isa.RA[7], isa.R3]), ("shr", [isa.R2, Imm(3)]),
     ("fmax", [isa.RA[1], isa.RB[2]])],
    [isa.RA[10], isa.RB[11], isa.R3, isa.R1, isa.NOP]))                             # add-op destination


def _swapped():
    """(instruction, move node, add node) for every grid entry the swap forms."""
    out = []
    for src, dst, cond, (op, srcs), adst in _SWAP_GRID:
        move, add = _node("mov", dst, [src], cond=cond), _node(op, adst, srcs, unit="add")
        bundle = sched.Bundle(swap=True)
        assert bundle.try_add(move)
        if bundle.try_add(add):
            assert bundle.mul is move.alu_ops[0] and bundle.add is add.alu_ops[0]
            out.append((isa.canonicalize(bundle.to_instr()), move, add))
    return out


def test_swapped_instructions_behave_like_their_two_ops():
    """Run from the same registers, the joint instruction (decoded from its
    word) leaves every register as the move alone on the add ALU and the
    add op alone do together: r5quad/r5rep replication, write swap and
    conditions included."""
    formed = _swapped()
    assert len(formed) > 300
    for ins, move, add in formed:
        word = isa.encode(ins)
        for seed in range(2):
            base = _execute(_qpu(seed), Instr())
            joint = _execute(_qpu(seed), isa.decode(word))
            alone = [_execute(_qpu(seed), isa.decode(isa.encode(isa.canonicalize(Instr(add=n.alu_ops[0])))))
                     for n in (move, add)]
            want = [next((s[i] for s in alone if not np.array_equal(s[i], base[i])), base[i])
                    for i in range(len(base))]
            for i, (got, exp) in enumerate(zip(joint, want)):
                np.testing.assert_array_equal(got, exp, err_msg=f"{isa.format_instr(ins)} register {i}")


@pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")
def test_swapped_instructions_assemble_to_our_words():
    formed = _swapped()
    text = "\n".join(isa.format_instr(ins) for ins, _, _ in formed) + "\n"
    words, warnings = vc4asm_assemble(text, verify=False)
    ours = [isa.encode(ins) for ins, _, _ in formed]
    bad = [(isa.format_instr(formed[i][0]), f"{w:016x}", f"{o:016x}")
           for i, (w, o) in enumerate(zip(words, ours)) if w != o]
    assert len(words) == len(ours) and not bad, bad[:5]


# ---------------------------------------------------------------------------
# Special float values, ragged tails, dots: bit-identical to the switch off
# ---------------------------------------------------------------------------


SPECIAL = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-40, -1e-40, 3e38, -3e38, 1.0, -1.0,
                    1e-38, 0.5, 2.0, 65504.0, -7.25], np.float32)


def _special(n, scale=1.0):
    x = (rng.standard_normal(n) * scale).astype(np.float32)
    idx = rng.choice(n, size=max(1, n // 3), replace=False)
    x[idx] = SPECIAL[rng.integers(0, len(SPECIAL), len(idx))]
    bits = x.view(np.uint32)
    bits[:2] = [0x7FC12345, 0xFFC00001][:n]       # NaNs with a payload, one negative
    return x


@piton.jit
def select_k(x_ptr, y_ptr, o_ptr, n, a, b, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = tl.load(y_ptr + offs, mask=m, other=-0.0)
    u = tl.where(x > y, x * a, y - b)
    v = tl.where(x == x, u, y)
    w = tl.maximum(u, v) + tl.minimum(x, y)
    tl.store(o_ptr + offs, tl.where(w < 0.0, -w, w * 0.5), mask=m)
    tl.store(o_ptr + n + offs, tl.abs(x) - y * y + tl.exp(-tl.abs(x)) + 1.0 / (tl.abs(y) + a), mask=m)


@piton.jit
def int_bits_k(x_ptr, y_ptr, o_ptr, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0)
    y = tl.load(y_ptr + offs, mask=m, other=1)
    r = ((x << 3) ^ (y >> 2)) + (x & y) - (x | 7)
    tl.store(o_ptr + offs, tl.where(r < 0, -r, r) * s, mask=m)
    tl.store(o_ptr + n + offs, (tl.maximum(x, y) - tl.minimum(x, s)) ^ (offs * 3), mask=m)


def _both(monkeypatch, kernel, grid, make, **constants):
    """Outputs of kernel[grid] with the switch off and on, compared bit for bit."""
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        kernel._cache.clear()
        args = make()
        kernel[grid](*args, **constants)
        out[env] = [a for a in args if isinstance(a, np.ndarray)]
    kernel._cache.clear()
    for i, (a, b) in enumerate(zip(out["0"], out["1"])):
        np.testing.assert_array_equal(b.view(np.uint32), a.view(np.uint32), err_msg=f"output {i}")


@pytest.mark.parametrize("n,block", [(1, 64), (100, 64), (16 * 37 + 3, 16)])
def test_special_values_bit_identical(monkeypatch, n, block):
    x, y = _special(n, 4), _special(n, 4)
    grid = (piton.cdiv(n, block),)
    _both(monkeypatch, select_k, grid, lambda: (x.copy(), y.copy(), np.zeros(2 * n, np.float32), n, -1.5, 0.25), BLOCK=block)
    xi = rng.integers(-2 ** 31, 2 ** 31, n).astype(np.int32)
    yi = rng.integers(-2 ** 31, 2 ** 31, n).astype(np.int32)
    xi[:1] = -2 ** 31
    _both(monkeypatch, int_bits_k, grid, lambda: (xi.copy(), yi.copy(), np.zeros(2 * n, np.int32), n, -3), BLOCK=block)


@pytest.mark.parametrize("M,N,Kd", [(4, 16, 8), (2, 64, 4), (16, 16, 16), (1, 16, 16), (4, 8, 8)])
def test_dot_special_values_bit_identical(monkeypatch, M, N, Kd):
    """NaN, +-inf, -0.0 and denormal operands through the split multiply-add
    pairs (and the small-N dot)."""
    a, b = _special(M * Kd), _special(Kd * N)
    _both(monkeypatch, dot_tile, (1,), lambda: (a.copy(), b.copy(), np.zeros(M * N, np.float32)), M=M, N=N, K=Kd)


@pytest.mark.parametrize("M,N,Kd,BK", [(8, 16, 32, 8), (4, 32, 64, 16)])
def test_dot_loop_special_values_bit_identical(monkeypatch, M, N, Kd, BK):
    a, b = _special(M * Kd), _special(Kd * N)
    _both(monkeypatch, dot_loop, (1,), lambda: (a.copy(), b.copy(), np.zeros(M * N, np.float32)), M=M, N=N, K=Kd, BK=BK)


# ---------------------------------------------------------------------------
# Loop entries: no latency hazard on any path
# ---------------------------------------------------------------------------


@piton.jit
def carried_k(x_ptr, o_ptr, n_iter, a, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32) + a
    top = tl.zeros((BLOCK,), dtype=tl.float32) - 1e30
    for i in range(n_iter):
        v = tl.load(x_ptr + i * BLOCK + offs)
        acc = acc * 0.5 + v * v
        top = tl.maximum(top, v)
    tl.store(o_ptr + offs, acc + top)


@piton.jit
def int_loop_k(o_ptr, n_iter, a, b, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = offs * a
    y = offs + b
    for i in range(n_iter):
        t = x ^ (y << 1)
        y = (y + t) & 0xFFFF
        x = t - i
    tl.store(o_ptr + offs, x)
    tl.store(o_ptr + BLOCK + offs, y)


@piton.jit
def nested_loop_k(x_ptr, o_ptr, n_outer, n_inner, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tot = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n_outer):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for j in range(n_inner):
            acc = acc * 0.5 + tl.load(x_ptr + (i * n_inner + j) * BLOCK + offs)
        tot += acc * (i + 1)
    tl.store(o_ptr + offs, tot)


def _next(prog, i):
    """Instructions that may run right after instruction i (a branch takes
    effect after its three delay slots; thrend ends two after it)."""
    instrs, n = prog.instrs, len(prog.instrs)
    if i >= 3 and instrs[i - 3].kind == "branch":
        br = instrs[i - 3]
        return [prog.labels[br.target]] + ([i + 1] if br.bcond != "always" and i + 1 < n else [])
    if any(i >= k and instrs[i - k].kind == "alu" and instrs[i - k].sig == "thrend" for k in (2,)):
        return []
    return [i + 1] if i + 1 < n else []


def path_violations(prog) -> list:
    """(i, j, distance) closer than sched.required_distance on some path."""
    res = [sched.instr_res(x) for x in prog.instrs]
    bad = []
    for i in range(len(res)):
        todo, seen = [(j, 1) for j in _next(prog, i)], set()
        while todo:
            j, d = todo.pop()
            if d > 4 or (j, d) in seen:
                continue
            seen.add((j, d))
            if sched.required_distance(res[i], res[j]) > d:
                bad.append((i, j, d))
            todo += [(k, d + 1) for k in _next(prog, j)]
    return bad


@pytest.mark.parametrize("env", ["0", "1"])
def test_loop_entries_hazard_free_on_every_path(monkeypatch, env):
    monkeypatch.setenv("PITON_PAIR", env)
    builds = [
        compiler.compile(carried_k, {"x_ptr": "*fp32", "o_ptr": "*fp32", "n_iter": "i32", "a": "fp32"},
                         {"BLOCK": 32}, grid=(1,)),
        compiler.compile(int_loop_k, {"o_ptr": "*i32", "n_iter": "i32", "a": "i32", "b": "i32"},
                         {"BLOCK": 16}, grid=(1,)),
        compiler.compile(nested_loop_k, {"x_ptr": "*fp32", "o_ptr": "*fp32", "n_outer": "i32", "n_inner": "i32"},
                         {"BLOCK": 32}, grid=(1,)),
    ]
    for ck in builds:
        assert not path_violations(ck.program), ck.name


@pytest.mark.parametrize("trips", [0, 1, 5])
def test_loops_bit_identical_at_every_trip_count(monkeypatch, trips):
    x = _special(max(1, trips * 4) * 32)
    _both(monkeypatch, carried_k, (1,), lambda: (x.copy(), np.zeros(32, np.float32), trips, 0.75), BLOCK=32)
    _both(monkeypatch, int_loop_k, (1,), lambda: (np.zeros(32, np.int32), trips, 12345, -77), BLOCK=16)
    _both(monkeypatch, nested_loop_k, (1,), lambda: (x.copy(), np.zeros(32, np.float32), trips, 4), BLOCK=32)
    _both(monkeypatch, nested_loop_k, (1,), lambda: (x.copy(), np.zeros(32, np.float32), 4, trips), BLOCK=32)


def test_suite_loop_builds_hazard_free_on_every_path(monkeypatch):
    """The suite's loop kernels (Pi sizes) with the switch on, the paths the
    emulator-size runs never take included."""
    from perf.suite import CASES_BY_NAME, compile_case
    monkeypatch.setenv("PITON_PAIR", "1")
    for name in ("matmul_8x64x8", "mandel_16x16x2", "softmax", "row_sum"):
        ck, _ = compile_case(CASES_BY_NAME[name], "pi")
        assert not path_violations(ck.program), name


def test_run_emulator_on_carried_loop(monkeypatch):
    """The emulator agrees with numpy on the carried loop (the bit-identical
    tests above only compare the two switch settings)."""
    monkeypatch.setenv("PITON_PAIR", "1")
    x = rng.standard_normal(5 * 32).astype(np.float32)
    ck = compiler.compile(carried_k, {"x_ptr": "*fp32", "o_ptr": "*fp32", "n_iter": "i32", "a": "fp32"},
                          {"BLOCK": 32}, grid=(1,))
    out = np.zeros(32, np.float32)
    run_emulator(ck, (1,), {"x_ptr": x, "o_ptr": out, "n_iter": 5, "a": 0.75})
    acc = np.full(32, 0.75, np.float32)
    top = np.full(32, -1e30, np.float32)
    for i in range(5):
        v = x[i * 32:(i + 1) * 32]
        acc = acc * np.float32(0.5) + v * v
        top = np.maximum(top, v)
    np.testing.assert_allclose(out, acc + top, rtol=1e-6)


@pytest.mark.parametrize("swap", [False, True])
def test_refused_pair_leaves_no_flags_behind(swap):
    """A two-op node whose second op hits a destination conflict after its
    first set the flags: try_add restores everything, so a later op of that
    instruction does not set the flags from its own result."""
    pair_ = sched._node_for(0, MInst(ops=[MOp("add", isa.R1, [isa.R2, isa.R3], setf=True, unit="add"),
                                          MOp("fmul", isa.R1, [isa.R2, isa.R3], unit="mul")], atomic=True))
    bundle = sched.Bundle(swap=swap)
    assert not bundle.try_add(pair_)
    assert (bundle.add, bundle.mul, bundle.setf, bundle.setf_unit) == (None, None, False, None)
    assert bundle.try_add(_node("mov", isa.R0, [isa.RA[1]]))
    assert not bundle.to_instr().setf
