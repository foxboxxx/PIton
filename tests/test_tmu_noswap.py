"""PITON_TMU_NOSWAP: TMU_NOSWAP in programs that are not coarsened and spread
their loads over TMU0 and TMU1 (lower.py Lowering.run), so each load reaches
the same physical TMU of the slice on all four of its QPUs; the emulator's
TMU0/TMU1 model with the QPU2/QPU3 swap (guide p40)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import piton  # noqa: E402
import piton.compiler as pc  # noqa: E402
import piton.language as tl  # noqa: E402
from examples.kernels import add_kernel, matmul_kernel  # noqa: E402
from perf.suite import CASES, CASES_BY_NAME, compile_case, default_check  # noqa: E402
from piton.backend.sched import fix_block_boundaries, instr_res, required_distance  # noqa: E402
from piton.compiler import compile as pcompile  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from piton.qpu.emulator import EmulatorError, HazardError, V3D  # noqa: E402
from piton.qpu.isa import AluOp, Imm, Instr  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402

PTR = "*fp32"


@pytest.fixture
def noswap_on(monkeypatch):
    monkeypatch.setenv("PITON_TMU_NOSWAP", "1")


# ---------------------------------------------------------------------------
# Emulator: two TMUs per slice, per-unit FIFOs, the QPU2/QPU3 swap
# ---------------------------------------------------------------------------


def mov(dst, src):
    return Instr(add=AluOp("mov", dst, src))


def unif(i):
    return mov(isa.RA[i], isa.UNIF)


def request(unit, i):
    return mov(isa.TMU_S[unit], isa.RA[i])


def noswap():
    return mov(isa.TMU_NOSWAP, Imm(1))


LOAD_A = [unif(0), Instr.nop()]          # ra0 = address (not read by the next instruction)


END = [Instr(sig="thrend"), Instr.nop(), Instr.nop()]


def run_raw(instrs, uniforms, qpu=0, v=None):
    """Run a hand-written program on one QPU; returns (v3d, qpu)."""
    v = v or V3D()
    prog = v.load([isa.encode(isa.canonicalize(i)) for i in instrs + END])
    a = v.mem.alloc(4 * len(uniforms), 16)
    v.mem.write_words(a, np.array(uniforms, dtype=np.uint32))
    q = v.qpus[qpu]
    q.start(prog, 0, a)
    while q.running:
        q.step()
    return v, q


def words_at(v, values):
    a = v.mem.alloc(4 * len(values), 64)
    v.mem.write_words(a, np.array(values, dtype=np.uint32))
    return [a + 4 * i for i in range(len(values))]


ORDER = [unif(0), unif(1), unif(2), request(0, 0), request(1, 1), request(0, 2),
         Instr(sig="ldtmu1"),
         Instr(add=AluOp("mov", isa.R0, isa.R4), sig="ldtmu0"),
         Instr(add=AluOp("mov", isa.R1, isa.R4), sig="ldtmu0"),
         mov(isa.R2, isa.R4)]


@pytest.mark.parametrize("qpu,swapped", [(0, False), (1, False), (2, True), (3, True), (6, True), (9, False)])
@pytest.mark.parametrize("with_noswap", [False, True])
def test_tmu_results_come_back_in_order_per_unit(qpu, swapped, with_noswap):
    """ldtmu1 returns the TMU1 request although an older TMU0 one is queued;
    the swap only changes which physical TMU of the slice serves each FIFO."""
    v = V3D()
    a, b, c = words_at(v, [111, 222, 333])
    prog = ([noswap(), Instr.nop(), Instr.nop()] if with_noswap else []) + ORDER
    v, q = run_raw(prog, [a, b, c], qpu=qpu, v=v)
    assert [int(q.acc[r][0]) for r in range(3)] == [222, 111, 333]
    s = qpu // 4
    phys = {0: 1, 1: 0} if swapped and not with_noswap else {0: 0, 1: 1}
    assert v.tmu_requests == {(s, phys[0]): 2, (s, phys[1]): 1}


def test_tmu_fifos_hold_four_requests_each():
    v = V3D()
    (a,) = words_at(v, [5])
    four_each = LOAD_A + [request(u, 0) for _ in range(4) for u in (0, 1)] + \
        [Instr(sig=f"ldtmu{u}") for _ in range(4) for u in (0, 1)]
    run_raw(four_each, [a], v=v)
    for qpu in (0, 2):         # QPU2's swapped FIFOs overflow the same way
        five = LOAD_A + [request(0, 0)] * 5 + [Instr(sig="ldtmu0")] * 5
        with pytest.raises(HazardError, match="TMU0 request FIFO overflow"):
            run_raw(five, [a], qpu=qpu, v=V3D())


def test_ldtmu_of_the_other_unit_is_an_error():
    v = V3D()
    (a,) = words_at(v, [5])
    with pytest.raises(EmulatorError, match="ldtmu1 with an empty TMU FIFO"):
        run_raw(LOAD_A + [request(0, 0), Instr(sig="ldtmu1"), Instr(sig="ldtmu0")], [a], qpu=2, v=v)


@pytest.mark.parametrize("gap,ok", [(0, False), (1, False), (2, True), (5, True)])
def test_tmu_noswap_three_instructions_before_the_first_request(gap, ok):
    """Guide p37: TMU_NOSWAP must be written three instructions before the
    first TMU write (gap = instructions in between)."""
    v = V3D()
    (a,) = words_at(v, [5])
    prog = [unif(0), noswap()] + [Instr.nop()] * gap + [request(0, 0), Instr(sig="ldtmu0")]
    if ok:
        run_raw(prog, [a], qpu=2, v=v)
        assert v.tmu_requests == {(0, 0): 1}
    else:
        with pytest.raises(HazardError, match="three instructions after TMU_NOSWAP"):
            run_raw(prog, [a], qpu=2, v=v)


def test_tmu_noswap_after_a_request_is_a_hazard():
    v = V3D()
    (a,) = words_at(v, [5])
    prog = LOAD_A + [request(0, 0), Instr(sig="ldtmu0"), noswap()]
    with pytest.raises(HazardError, match="TMU_NOSWAP written after a TMU request"):
        run_raw(prog, [a], qpu=0, v=v)


def test_tmu_noswap_is_reset_for_the_next_program():
    v = V3D()
    (a,) = words_at(v, [5])
    run_raw([noswap()] + LOAD_A + [request(0, 0), Instr(sig="ldtmu0")], [a], qpu=2, v=v)
    run_raw(LOAD_A + [request(0, 0), Instr(sig="ldtmu0")], [a], qpu=2, v=v)
    assert v.tmu_requests == {(0, 0): 1, (0, 1): 1}


def test_scheduler_keeps_noswap_three_instructions_ahead_across_blocks():
    first = noswap()
    req = request(0, 0)
    assert required_distance(instr_res(first), instr_res(req)) == 3
    blocks = {"entry": [unif(0), first], "main": [req]}
    fix_block_boundaries(blocks, {"entry": [], "main": ["entry"]}, ["entry", "main"])
    assert blocks["main"][-1] is req and len(blocks["main"]) == 3


def _matmul(M, N, K, bm=8, bn=64, bk=4, group=4, stats=None):
    sig = {"a_ptr": PTR, "b_ptr": PTR, "c_ptr": PTR, "M": "i32", "N": "i32",
           "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}
    cons = {"K": K, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": group,
            "stride_ak": 1, "stride_bn": 1, "stride_cn": 1}
    grid = (-(-M // bm) * -(-N // bn),)
    ck = pcompile(matmul_kernel, sig, cons, grid=grid)
    rng = np.random.default_rng(M * 1000 + N * 10 + K)
    a = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    b = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    vals = {"a_ptr": a, "b_ptr": b, "c_ptr": np.full((M, N), np.nan, np.float32), "M": M, "N": N,
            "stride_am": K, "stride_bk": N, "stride_cm": N}
    (stats if stats is not None else {}).update(run_emulator(ck, grid, vals))
    np.testing.assert_allclose(vals["c_ptr"], a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4 * K)
    return ck


def test_noswap_keeps_shared_lines_in_one_tmu_cache(monkeypatch):
    """matmul: the four programs of a slice read the same B panel; with the
    swap every line is fetched into both TMUs of the slice."""
    lines = {}
    for value in ("0", "1"):
        monkeypatch.setenv("PITON_TMU_NOSWAP", value)
        stats = {}
        ck = _matmul(32, 64, 64, stats=stats)
        assert ck.stats.get("tmu_noswap") == (1 if value == "1" else None)
        lines[value] = sum(stats["tmu_unit_lines"].values())
        assert sum(stats["tmu_unit_requests"].values()) == stats["tmu_loads"]
        assert len(stats["tmu_unit_requests"]) == 2            # 4 programs: one slice, both TMUs
    assert lines["1"] < 0.7 * lines["0"]


# ---------------------------------------------------------------------------
# Compiler: the switch and where TMU_NOSWAP goes
# ---------------------------------------------------------------------------

ADD_SIG = {"x_ptr": PTR, "y_ptr": PTR, "output_ptr": PTR, "n_elements": "i32"}


def _first(instrs, pred):
    return next(i for i, ins in enumerate(instrs) if pred(ins))


def _writes(ins, regs):
    return any(op is not None and op.dst in regs for op in (ins.add, ins.mul))


def _noswap_writes(ck):
    return [i for i, ins in enumerate(ck.program.instrs) if _writes(ins, (isa.TMU_NOSWAP,))]


@pytest.mark.parametrize("value,on", [(None, False), ("", False), ("0", False), ("1", True)])
def test_switch_values(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("PITON_TMU_NOSWAP", raising=False)
    else:
        monkeypatch.setenv("PITON_TMU_NOSWAP", value)
    assert pc.tmu_noswap_enabled() == on
    ck = pcompile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256})
    assert ck.coarsen == 1 and bool(ck.stats.get("tmu_noswap")) == on
    instrs = ck.program.instrs
    assert _noswap_writes(ck) == ([0] if on else [])
    if on:
        assert _first(instrs, lambda ins: _writes(ins, isa.TMU_S)) >= 3


def test_switch_off_is_the_default(monkeypatch):
    monkeypatch.delenv("PITON_TMU_NOSWAP", raising=False)
    unset = pcompile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256})
    for value in ("0", ""):
        monkeypatch.setenv("PITON_TMU_NOSWAP", value)
        off = pcompile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256})
        assert unset.words == off.words and unset.uniforms == off.uniforms


@pytest.mark.parametrize("value", ["2", "yes", "on", "noswap"])
def test_unknown_switch_value_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PITON_TMU_NOSWAP", value)
    with pytest.raises(ValueError, match="PITON_TMU_NOSWAP"):
        pcompile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256})


def _without_noswap(prog):
    """The program's words with the TMU_NOSWAP op taken out of its instruction."""
    from dataclasses import replace
    addrs = prog.label_addrs()
    out = []
    for pc_, ins in enumerate(prog.instrs):
        for slot in ("add", "mul"):
            op = getattr(ins, slot)
            if op is not None and op.dst is isa.TMU_NOSWAP:
                ins = replace(ins, **{slot: None})
        out.append(isa.encode(isa.canonicalize(ins), 8 * pc_, addrs))
    return out


@pytest.mark.parametrize("name", ["row_sum", "add", "exp", "matmul_8x64x8", "dense_relu"])
def test_noswap_shares_an_instruction_with_a_uniform_read(monkeypatch, name):
    """TMU_NOSWAP is no closely coupled peripheral access: it pairs with one of
    the entry block's uniform reads and costs no instruction; the rest of the
    program is the one built without it."""
    monkeypatch.setenv("PITON_TMU_NOSWAP", "0")
    off, _ = compile_case(CASES_BY_NAME[name], "emu")
    monkeypatch.setenv("PITON_TMU_NOSWAP", "1")
    on, _ = compile_case(CASES_BY_NAME[name], "emu")
    instrs = on.program.instrs
    assert on.stats.get("tmu_noswap") == 1 and len(instrs) == len(off.program.instrs)
    w = _first(instrs, lambda ins: _writes(ins, (isa.TMU_NOSWAP,)))
    assert any(op is not None and isa.UNIF in (op.a, op.b) for op in (instrs[w].add, instrs[w].mul))
    assert _first(instrs, lambda ins: _writes(ins, isa.TMU_S)) - w >= 3
    assert _without_noswap(on.program) == list(off.words) and on.uniforms == off.uniforms


@pytest.mark.parametrize("name", ["add", "dense_relu"])
def test_coarsened_programs_keep_the_hardware_swap(monkeypatch, name):
    """TMU_NOSWAP only in programs that are not coarsened (PITON_PERSISTENT):
    it slowed the coarsened ones on the Pi (dense_relu with k=2 0.960x) and
    sped up the same kernels built plain (dense_relu 1.026x)."""
    monkeypatch.setenv("PITON_TMU_NOSWAP", "1")
    coarsened, _ = compile_case(CASES_BY_NAME[name], "pi")
    monkeypatch.setenv("PITON_PERSISTENT", "0")
    plain, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert coarsened.coarsen > 1 and plain.coarsen == 1
    assert not coarsened.stats.get("tmu_noswap") and plain.stats.get("tmu_noswap") == 1
    assert not _noswap_writes(coarsened) and _noswap_writes(plain) == [0]
    monkeypatch.setenv("PITON_TMU_NOSWAP", "0")
    monkeypatch.delenv("PITON_PERSISTENT")
    off, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert list(off.words) == list(coarsened.words)


@piton.jit
def one_load(x_ptr, out_ptr):
    tl.store(out_ptr + tl.arange(0, 16), tl.load(x_ptr + tl.arange(0, 16)) + 1.0)


def test_programs_using_only_tmu0_keep_the_hardware_swap(noswap_on):
    """One request per program: the swap is what spreads it over both TMUs."""
    ck = pcompile(one_load, {"x_ptr": PTR, "out_ptr": PTR}, {})
    assert not ck.stats.get("tmu_noswap") and not _noswap_writes(ck)


@piton.jit
def three_loads(x_ptr, y_ptr, z_ptr, out_ptr):
    offs = tl.arange(0, 16)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs) * tl.load(z_ptr + offs))


def test_unbalanced_programs_keep_the_hardware_swap(noswap_on):
    """Two requests on TMU0, one on TMU1: without the swap physical TMU0 would
    serve 8 of each slice's 12 requests."""
    ck = pcompile(three_loads, {"x_ptr": PTR, "y_ptr": PTR, "z_ptr": PTR, "out_ptr": PTR}, {})
    assert not ck.stats.get("tmu_noswap") and not _noswap_writes(ck)


# ---------------------------------------------------------------------------
# Every suite kernel with the switch on, against numpy
# ---------------------------------------------------------------------------


def _check(case, ck, run, grid=None):
    """Run in the strict emulator against the suite's reference; grid: the
    first programs of a 1-D case (their outputs come first)."""
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    run_emulator(ck, grid or run.grid, values)
    for name, expected in run.expected.items():
        got = values[name]
        if grid is not None:
            n = expected.size * grid[0] // run.grid[0]
            got, expected = got.ravel()[:n], expected.ravel()[:n]
        error = default_check(run, name, got, expected)
        assert error is None, f"{case.name}.{name}: {error}"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_suite_case_with_noswap(noswap_on, case):
    ck, run = compile_case(case, "emu")
    assert ck.coarsen == 1 and ck.stats.get("tmu_noswap") == 1 and _noswap_writes(ck) == [0]
    _check(case, ck, run)


# Pi-size builds the strict emulator runs in seconds (~20k instructions/s):
# row_sum, row_max 32k instructions, transpose 242k, softmax 300k.  The other
# plain ones run 0.8-1.7M (exp ... idiv: 1024 programs, the first 16 here;
# compute_peak, matmul) or more (mandel); copy ... scaled_relu and dense_relu
# are coarsened at Pi size and keep the swap.
PI_RUNS = {"row_sum": None, "row_max": None, "transpose": None, "softmax": None,
           **dict.fromkeys(("exp", "exp2", "log", "log2", "recip", "rsqrt", "sqrt", "idiv"), (16,))}


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_suite_case_at_pi_size_with_noswap(noswap_on, case):
    ck, run = compile_case(case, "pi")
    assert bool(ck.stats.get("tmu_noswap")) == (ck.coarsen == 1)
    assert _noswap_writes(ck) == ([0] if ck.coarsen == 1 else [])
    if case.name in PI_RUNS and ck.coarsen == 1:          # (PITON_ISSUE coarsens recip at Pi size)
        _check(case, ck, run, PI_RUNS[case.name])


@pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")
@pytest.mark.parametrize("name", ["row_sum", "exp", "softmax", "transpose", "matmul_8x64x4", "matmul_8x64x8",
                                  "dense_relu", "mandel_16x16x2"])
def test_noswap_pi_builds_assemble_with_vc4asm(noswap_on, monkeypatch, name):
    """Pi-size builds with TMU_NOSWAP (dense_relu built plain): vc4asm
    assembles our .qasm to the same words and its verifier (-V) reports
    nothing."""
    if name == "dense_relu":
        monkeypatch.setenv("PITON_PERSISTENT", "0")
    ck, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert ck.stats.get("tmu_noswap") == 1
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert words == [int(w) for w in ck.words]
    assert not warnings, warnings[:5]
