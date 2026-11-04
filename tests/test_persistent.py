"""PITON_PERSISTENT: several program ids per hardware program.

compile() wraps the kernel in a loop over k consecutive axis-0 program ids
(Lowering.begin_programs); the host launches ceil(gx / k) programs and passes
each its own index as pid0 and the logical grid as nprog0. Every test runs the
strict emulator, so a store DMA that is not waited for (before the next VPM
write or before thread end), or a uniform read too soon after a unif_addr
write, fails the run. Outputs are compared bit for bit with the plain build.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import piton
import piton.language as tl
from piton import compiler
from piton.backend.regalloc import RegAllocError
from piton.qpu import isa
from piton.qpu.emulator import V3D, HazardError
from piton.qpu.isa import f2u
from piton.runtime.launcher import program_ids, run_emulator, semaphore_programs, slot_semaphores, uniforms_for
from piton.tools.compile import emit_c

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from kernels import add_kernel, matmul_kernel, row_sum_kernel  # noqa: E402
from primitive_kernels import copy_kernel, fma_kernel, idiv_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perf.suite import CASES_BY_NAME, compile_case  # noqa: E402

ADD_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
COPY_SIG = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
rng = np.random.default_rng(5)


@pytest.fixture
def persistent(monkeypatch):
    """persistent(value): PITON_PERSISTENT=value; None = the plain build ("0")."""
    def set_k(value):
        monkeypatch.setenv("PITON_PERSISTENT", "0" if value is None else str(value))
    return set_k


def _fail_builds(monkeypatch, when):
    """Make _compile_tile raise RegAllocError for the option sets `when` accepts."""
    real = compiler._compile_tile

    def build(fn, signature, constants, name, pid_max, options, *variant, **kw):
        if when(options):
            raise RegAllocError("forced by the test")
        return real(fn, signature, constants, name, pid_max, options, *variant, **kw)
    monkeypatch.setattr(compiler, "_compile_tile", build)


def _run(ck, grid, values):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    stats = run_emulator(ck, grid, values)
    return values, stats


def _same_as_plain(persistent, k, kernel, sig, cons, grid, values, compile_grid="launch"):
    """Compile plain and with PITON_PERSISTENT=k (for `compile_grid`, by default
    the launch grid), run both on `grid`; every array must come back
    bit-identical. Returns (coarsened kernel, its outputs, both runs' stats)."""
    compile_grid = grid if compile_grid == "launch" else compile_grid
    persistent(None)
    plain = compiler.compile(kernel, sig, cons, grid=compile_grid)
    assert plain.coarsen == 1
    want, plain_stats = _run(plain, grid, values)
    persistent(k)
    ck = compiler.compile(kernel, sig, cons, grid=compile_grid)
    got, stats = _run(ck, grid, values)
    for name, v in want.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(got[name].view(np.uint32), v.view(np.uint32), err_msg=name)
    return ck, got, stats, plain_stats


def _add_values(n):
    return {"x_ptr": rng.standard_normal(n).astype(np.float32), "y_ptr": rng.standard_normal(n).astype(np.float32),
            "output_ptr": np.full(n + 16, 7.0, dtype=np.float32), "n_elements": n}   # guard band: masked tail


@pytest.mark.parametrize("blocks,k", [(10, 4), (12, 4), (3, 4), (7, 3), (9, 2), (1, 4), (5, 5), (11, 6)])
def test_add_runs_k_ids_per_program(persistent, blocks, k):
    """Grids that are not a multiple of k: the last program runs fewer ids."""
    values = _add_values(blocks * 64 - 5)
    ck, got, _, _ = _same_as_plain(persistent, k, add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, (blocks,), values)
    assert ck.coarsen == min(k, blocks)
    # the kernel's own uniforms first, then the hardware program's index and the logical grid
    assert ck.uniforms[:4] == [("arg", n) for n in ADD_SIG]
    assert ck.coarsen == 1 or ck.uniforms[4:] in ([("zero", "zero_ptr"), ("pid", "pid0"), ("nprog", "nprog0")],
                               [("zero", "zero_ptr"), ("pid", "pid0"), ("nprog", "nprog0"), ("self", "unif_base")])
    n = values["n_elements"]
    np.testing.assert_allclose(got["output_ptr"][:n], values["x_ptr"] + values["y_ptr"], rtol=1e-6, atol=1e-6)
    assert np.all(got["output_ptr"][n:] == 7.0)


def test_smaller_grid_than_compiled_for(persistent):
    """The bounds check uses the launch grid (nprog0), not the compile-time bound."""
    for blocks in (1, 5, 13, 16):
        ck, _, _, _ = _same_as_plain(persistent, 4, add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, (blocks,),
                                     _add_values(blocks * 64), compile_grid=(16,))
        assert ck.coarsen == 4


def test_without_a_grid_bound(persistent):
    ck, _, _, _ = _same_as_plain(persistent, 3, add_kernel, ADD_SIG, {"BLOCK_SIZE": 32}, (8,), _add_values(250),
                                 compile_grid=None)
    assert ck.coarsen == 3


@pytest.mark.parametrize("kernel,cons", [(add_kernel, {"BLOCK_SIZE": 256}), (copy_kernel, {"BLOCK": 256})])
def test_coarsening_executes_fewer_instructions(persistent, kernel, cons):
    sig = ADD_SIG if kernel is add_kernel else COPY_SIG
    n = 16 * 256
    values = _add_values(n) if kernel is add_kernel else \
        {"x_ptr": rng.standard_normal(n).astype(np.float32), "out_ptr": np.zeros(n, np.float32), "n": n}
    ck, _, stats, plain_stats = _same_as_plain(persistent, 4, kernel, sig, cons, (16,), values)
    assert ck.coarsen == 4
    assert stats["instructions"] < plain_stats["instructions"]


@piton.jit
def ids_kernel(out_ptr, flag_ptr, ROUNDS: tl.constexpr):
    lanes = tl.arange(0, 16)
    acc = tl.zeros((16,), dtype=tl.int32)
    for r in range(ROUNDS):                    # program id read inside a loop
        acc += tl.program_id(0) + r
    pid = tl.program_id(0)                     # and again after it
    row = tl.program_id(1)
    flat = row * tl.num_programs(0) + pid
    v = acc * 1000000 + tl.num_programs(0) * 10000 + tl.num_programs(1) * 100 + row
    if pid % 3 == 0:
        tl.store(flag_ptr + flat, pid + 1)
    tl.store(out_ptr + flat * 16 + lanes, v)


@pytest.mark.parametrize("grid,k", [((10, 3), 4), ((5, 2), 2), ((7,), 3), ((4, 2, 3), 3)])
def test_program_id_and_num_programs_are_logical(persistent, grid, k):
    persistent(k)
    rounds = 3
    gx, gy, gz = (list(grid) + [1, 1])[:3]
    ck = compiler.compile(ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": rounds}, grid=grid)
    assert ck.coarsen == min(k, gx)
    out = np.full(gx * gy * 16, -1, dtype=np.int32)
    flags = np.zeros(gx * gy, dtype=np.int32)
    values, _ = _run(ck, grid, {"out_ptr": out, "flag_ptr": flags})
    # every (z) slice writes the same values: z is not part of the address
    pid = np.tile(np.arange(gx), gy)
    row = np.repeat(np.arange(gy), gx)
    expected = (rounds * pid + rounds * (rounds - 1) // 2) * 1000000 + gx * 10000 + gy * 100 + row
    np.testing.assert_array_equal(values["out_ptr"].reshape(-1, 16), np.repeat(expected[:, None], 16, axis=1))
    np.testing.assert_array_equal(values["flag_ptr"], np.where(pid % 3 == 0, pid + 1, 0))


@piton.jit
def early_exit_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_blocks:
        return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)


def test_early_return(persistent):
    """`return` skips to the next program id, not to the end of the program."""
    x = rng.standard_normal(320).astype(np.float32)
    ck, got, _, _ = _same_as_plain(persistent, 4, early_exit_kernel,
                                   {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_blocks": "i32"}, {"BLOCK": 32}, (10,),
                                   {"x_ptr": x, "out_ptr": np.zeros(320, np.float32), "n_blocks": 7})
    assert ck.coarsen == 4
    np.testing.assert_array_equal(got["out_ptr"][:224], x[:224] * 2)
    assert np.all(got["out_ptr"][224:] == 0)


@piton.jit
def grid_stride_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK: tl.constexpr):
    """A hand-written persistent loop: the program id and grid size are loop bounds,
    with a nested loop inside."""
    lanes = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), n_rows, tl.num_programs(0)):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, n_cols, BLOCK):
            acc += tl.load(x_ptr + row * n_cols + c + lanes, mask=c + lanes < n_cols, other=0.0)
        tl.store(out_ptr + row * BLOCK + lanes, acc)


@pytest.mark.parametrize("grid,k", [((5,), 2), ((6,), 4), ((3,), 3)])
def test_program_ids_in_loop_bounds(persistent, grid, k):
    rows, cols, block = 13, 40, 16
    x = rng.standard_normal(rows * cols).astype(np.float32)
    ck, got, _, _ = _same_as_plain(persistent, k, grid_stride_kernel,
                                   {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_rows": "i32", "n_cols": "i32"},
                                   {"BLOCK": block}, grid,
                                   {"x_ptr": x, "out_ptr": np.zeros(rows * block, np.float32), "n_rows": rows,
                                    "n_cols": cols})
    assert ck.coarsen == k
    xp = np.pad(x.reshape(rows, cols), ((0, 0), (0, 48 - cols))).reshape(rows, 3, block).sum(axis=1)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), xp, rtol=1e-5, atol=1e-5)


@piton.jit
def mixed_store_kernel(x_ptr, out_ptr, rev_ptr, last_ptr, n, BLOCK: tl.constexpr):
    """Scalar, scatter (reversed, masked) and contiguous stores in one program."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, x + 1.0, mask=m)
    tl.store(rev_ptr + pid * BLOCK + (BLOCK - 1 - tl.arange(0, BLOCK)), x, mask=m)
    tl.store(last_ptr + pid, tl.sum(x, axis=0))


def test_scalar_and_scatter_stores(persistent):
    n, block, blocks = 7 * 32 - 9, 32, 7
    x = rng.standard_normal(n).astype(np.float32)
    ck, _, _, _ = _same_as_plain(
        persistent, 3, mixed_store_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "rev_ptr": "*fp32", "last_ptr": "*fp32", "n": "i32"},
        {"BLOCK": block}, (blocks,),
        {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "rev_ptr": np.zeros(blocks * block, np.float32),
         "last_ptr": np.zeros(blocks, np.float32), "n": n})
    assert ck.coarsen == 3


MATMUL_SIG = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
              "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}


@pytest.mark.parametrize("mode", ["registers", "reload"])
def test_matmul_k_loop(persistent, monkeypatch, mode):
    """A nested runtime loop with loop-carried accumulators and arguments re-read
    after it; k=3 does not divide the 10 tiles. Each mode forces one way to keep
    the kernel's uniforms: in registers, or re-read for every program id."""
    reload = mode == "reload"
    _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1 and o["coarsen_reload"] != reload)
    M, N, K = 40, 128, 16
    cons = dict(K=K, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=4, GROUP_SIZE_M=2)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    ck, got, _, _ = _same_as_plain(persistent, 3, matmul_kernel, MATMUL_SIG, cons, ((M // 8) * (N // 64),),
                                   {"a_ptr": a, "b_ptr": b, "c_ptr": np.full(M * N, np.nan, np.float32), "M": M,
                                    "N": N, "stride_am": K, "stride_bk": N, "stride_cm": N})
    assert ck.coarsen == 3 and ck.stats["coarsen_reload"] == reload
    if reload:
        assert ck.uniforms[-3:] == [("pid", "pid0"), ("nprog", "nprog0"), ("self", "unif_base")]
    np.testing.assert_allclose(got["c_ptr"].reshape(M, N), a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)


def _first_wait_after_loads(ck) -> bool:
    """In the program-id loop, a TMU request comes before the first DMA wait:
    the next id's loads overlap the previous id's last store."""
    prog = ck.program
    start = prog.labels[next(lbl for lbl in prog.labels if lbl.startswith("programs"))]
    text = [isa.format_instr(prog.instrs[i]) for i in range(start, len(prog.instrs))]
    first_wait = next(i for i, t in enumerate(text) if "vw_wait" in t)
    return any("t0s" in t or "t1s" in t for t in text[:first_wait])


def _waits_after_last_kick(ck, block_prefix) -> int:
    """DMA waits after the last DMA start (vw_addr write) of each block whose
    label starts with block_prefix."""
    prog = ck.program
    starts = sorted(prog.labels.values())
    total = 0
    for label, at in prog.labels.items():
        if not label.startswith(block_prefix):
            continue
        end = next((i for i in starts if i > at), len(prog.instrs))
        text = [isa.format_instr(prog.instrs[i]) for i in range(at, end)]
        kick = max(i for i, t in enumerate(text) if "vw_addr" in t)
        total += sum("vw_wait" in t for t in text[kick + 1:])
    return total


@pytest.mark.parametrize("kernel,cons", [(sfu_kernel, {"OP": 4, "BLOCK": 256}), (copy_kernel, {"BLOCK": 64})])
def test_last_store_overlaps_the_next_id(persistent, monkeypatch, kernel, cons):
    """Rerolled (exp) and straight-line versioned (copy) bodies: no wait for the
    previous id's DMA before the next id's loads. Private VPM rows only: a store
    through a slot waits for its DMA before it releases the slot
    (tests/test_persistent_slots.py)."""
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")      # (copy at BLOCK 64 has no slot stores anyway)
    n = 16 * cons["BLOCK"]
    x = rng.uniform(-8, 8, n).astype(np.float32)
    ck, got, _, _ = _same_as_plain(persistent, 4, kernel, COPY_SIG, cons, (16,),
                                   {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "n": n})
    assert ck.coarsen == 4 and bool(ck.stats["rerolled"]) == (kernel is sfu_kernel)
    assert _first_wait_after_loads(ck)
    if kernel is copy_kernel:
        # the unmasked store path leaves its DMA running (the plain build waits there)
        persistent(None)
        assert _waits_after_last_kick(compiler.compile(kernel, COPY_SIG, cons, grid=(16,)), "store_full") == 1
        assert _waits_after_last_kick(ck, "store_full") == 0
    if kernel is sfu_kernel:
        np.testing.assert_allclose(got["out_ptr"], np.exp(x.astype(np.float64)), rtol=4e-6)
    else:
        np.testing.assert_array_equal(got["out_ptr"], x)


def test_port_aware_clones_keep_the_id_loop_out_of_licm(persistent, monkeypatch):
    """Port-aware allocation (PITON_PORT_ALLOC) works on clones of the MIR;
    a clone keeps the loop headers LICM must leave alone (coarsen_hoist=False)."""
    from piton.backend.regalloc import clone_function
    persistent(4)
    _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1 and o["coarsen_hoist"])
    real = compiler._build
    seen = []

    def build(tile, ranges, options, interleave):
        out = real(tile, ranges, options, interleave)
        seen.append(out[1])
        return out
    monkeypatch.setattr(compiler, "_build", build)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(16,))
    assert ck.coarsen == 4 and ck.stats["coarsen_hoist"] == 0
    mf = next(f for f in seen if f.no_licm)
    assert clone_function(mf).no_licm == mf.no_licm


def test_falls_back_when_no_version_allocates(persistent, monkeypatch):
    persistent(None)
    plain = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    persistent(1)
    _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    assert ck.coarsen == 1 and "coarsen" not in ck.stats
    assert list(ck.words) == list(plain.words) and ck.uniforms == plain.uniforms


def test_automatic_by_default(monkeypatch):
    monkeypatch.delenv("PITON_PERSISTENT", raising=False)
    assert compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, grid=(1024,)).coarsen == 4


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_off(persistent, value):
    persistent(value)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    assert ck.coarsen == 1 and "coarsen" not in ck.stats
    assert ck.uniforms == [("arg", n) for n in ADD_SIG] + [("pid", "pid0"), ("zero", "zero_ptr")]
    assert not any(lbl.startswith("programs") for lbl in ck.program.labels)


@piton.jit
def no_pid_kernel(out_ptr):
    tl.store(out_ptr + tl.arange(0, 16), tl.arange(0, 16))


IDIV_SIG = {"x_ptr": "*i32", "y_ptr": "*i32", "q_ptr": "*i32", "r_ptr": "*i32", "n": "i32"}
ROW_SIG = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_cols": "i32", "stride": "i32"}


@pytest.mark.parametrize("value,kernel,sig,cons,grid,k", [
    (None, add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024,), 1),
    ("0", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024,), 1),
    # automatic: known 1-D grids of >= 256 programs, keeping >= 128 hardware programs
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (1024,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (4096,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (512,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (256,), 2),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (255,), 1),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, None, 1),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, (1024, 2), 1),
    # store slots (1 KiB DMAs): ids as short as without
    ("1", copy_kernel, COPY_SIG, {"BLOCK": 256}, (1024,), 4),
    # compute bound (under COARSEN_MIN_BYTES per instruction: rerolled exp, a
    # 16-element add whose id is mostly program-id and address arithmetic)
    ("1", sfu_kernel, COPY_SIG, {"OP": 4, "BLOCK": 256}, (1024,), 1),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024,), 1),
    # run-time loops: at most 2 (long ids: test_factor_long_ids)
    ("1", row_sum_kernel, ROW_SIG, {"BLOCK_M": 4, "BLOCK_N": 64}, (1024,), 2),
    ("1", row_sum_kernel, ROW_SIG, {"BLOCK_M": 4, "BLOCK_N": 64}, (64,), 1),
    # a forced factor: any grid, at most the grid
    ("8", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024,), 8),
    ("8", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (5,), 5),
    ("3", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, None, 3),
    ("4", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (64, 8), 4),
])
def test_factor(persistent, value, kernel, sig, cons, grid, k):
    persistent(value)
    assert compiler.compile(kernel, sig, cons, grid=grid).coarsen == k


def test_factor_long_ids(persistent, monkeypatch):
    """Long program ids get at most 2 (the suite's dense layer, ~2100
    instructions and ~18 KiB per id); compute-bound ones none, however long:
    exact division with either lowering (~6000 and ~1500 instructions per id
    for 4 KiB)."""
    persistent("1")
    assert compile_case(CASES_BY_NAME["dense_relu"], "pi")[0].coarsen == 2
    for fast in ("0", "1"):
        monkeypatch.setenv("PITON_FAST_IDIV", fast)
        assert compiler.compile(idiv_kernel, IDIV_SIG, {"BLOCK": 256}, grid=(1024,)).coarsen == 1


@pytest.mark.parametrize("slots", ["0", "1"])
def test_id_length(persistent, monkeypatch, slots):
    """Per-id instruction estimate: code outside loops once, loop bodies times
    their constant trip counts (the reroll chunk loop), unknown for run-time
    loops. It bounds what an id executes from above, and closely: with store
    slots the queued store after the rerolled loop (four slot copies of acquire
    and release, both store versions) is ~40% of the program, and multiplying it
    by the 4 chunks too took exp past COARSEN_SHORT (k 2 instead of 4)."""
    monkeypatch.setenv("PITON_STORE_SLOTS", slots)
    persistent(None)
    seen = []
    real = compiler._coarsening

    def spy(tile, pid_max, id_length):
        seen.append(id_length())
        return real(tile, pid_max, id_length)
    monkeypatch.setattr(compiler, "_coarsening", spy)
    ck = compiler.compile(sfu_kernel, COPY_SIG, {"OP": 4, "BLOCK": 256}, grid=(16,))
    assert ck.stats["rerolled"] == 64 and bool(ck.stats.get("store_slots")) == (slots == "1")
    n = 16 * 256 - 40
    _, stats = _run(ck, (16,), {"x_ptr": rng.uniform(-8, 8, n).astype(np.float32),
                                "out_ptr": np.zeros(n, np.float32), "n": n})
    per_id = stats["instructions"] / 16
    assert per_id <= seen[-1] <= 1.25 * per_id < compiler.COARSEN_SHORT
    compiler.compile(row_sum_kernel, ROW_SIG, {"BLOCK_M": 4, "BLOCK_N": 64}, grid=(64,))
    assert seen[-1] is None


def test_kernels_without_program_ids_are_left_alone(persistent):
    persistent(4)
    assert compiler.compile(no_pid_kernel, {"out_ptr": "*i32"}, {}, grid=(64,)).coarsen == 1


def test_bad_value(persistent):
    persistent("yes")
    with pytest.raises(ValueError, match="PITON_PERSISTENT"):
        compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, grid=(64,))


def test_program_ids():
    assert list(program_ids((5, 2))) == [((x, y, 0), (5, 2, 1)) for y in range(2) for x in range(5)]
    assert list(program_ids((5, 2), 2)) == [((x, y, 0), (5, 2, 1)) for y in range(2) for x in range(3)]
    assert list(program_ids((8,), 4)) == [((0, 0, 0), (8, 1, 1)), ((1, 0, 0), (8, 1, 1))]


@piton.jit
def jit_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask),
             mask=mask)


def test_launcher(persistent):
    """kernel[grid](...) launches ceil(grid / k) programs (compiled for the grid
    rounded up to a power of two)."""
    persistent(4)
    n = 11 * 32 - 3
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    ck = jit_add_kernel[(piton.cdiv(n, 32),)](x, y, out, n, BLOCK_SIZE=32)
    assert ck.coarsen == 4
    np.testing.assert_allclose(out, x + y, rtol=1e-6, atol=1e-6)


def test_dma_running_at_thread_end_is_a_hazard(persistent):
    """The emulator rejects a program that ends while its VDW DMA still runs
    (the final wait the coarsened loop relies on). The last block is partial,
    so it takes the masked store path, which leaves its DMA to that wait."""
    persistent(None)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 64}, grid=(4,))
    prog = ck.program
    end = next(i for i, ins in enumerate(prog.instrs) if ins.sig == "thrend")
    wait = max(i for i in range(end) if "vw_wait" in isa.format_instr(prog.instrs[i]))
    assert isa.format_instr(prog.instrs[wait]) == "mov -, vw_wait"
    x = np.arange(250, dtype=np.float32)
    run_emulator(ck, (4,), {"x_ptr": x, "out_ptr": np.zeros(250, np.float32), "n": 250})
    prog.instrs[wait] = isa.Instr.nop()
    with pytest.raises(HazardError, match="without waiting for its VDW DMA"):
        run_emulator(ck, (4,), {"x_ptr": x, "out_ptr": np.zeros(250, np.float32), "n": 250})


def test_vpm_write_before_waiting_for_the_previous_id_is_a_hazard(persistent):
    """Id 15 takes the masked store path while id 14's unmasked DMA is left
    running. Without the wait at the start of that path its VPM writes overwrite
    rows the DMA may still be reading on hardware (the engine is shared by 12
    QPUs). The id's loads outlast the emulator's modelled DMA time, so only a
    rule that wants a VW_WAIT since the DMA started catches it (a time check
    accepted this program)."""
    persistent(4)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 64}, grid=(16,))
    assert ck.coarsen == 4
    prog = ck.program
    part = prog.labels[next(lbl for lbl in prog.labels if lbl.startswith("store_part"))]
    wait = next(i for i in range(part, len(prog.instrs)) if "vw_wait" in isa.format_instr(prog.instrs[i]))
    assert isa.format_instr(prog.instrs[wait]) == "mov -, vw_wait"
    assert not any("mov vpm," in isa.format_instr(prog.instrs[i]) for i in range(part, wait))
    n = 16 * 64 - 5
    x = np.arange(n, dtype=np.float32)
    out = np.zeros(n, np.float32)
    run_emulator(ck, (16,), {"x_ptr": x, "out_ptr": out, "n": n})
    np.testing.assert_array_equal(out, x)
    prog.instrs[wait] = isa.Instr.nop()
    with pytest.raises(HazardError, match="VPM written while this QPU's VDW DMA"):
        run_emulator(ck, (16,), {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "n": n})


def _variants(monkeypatch, k, kernel, sig, cons, grid, values, compile_grid="launch"):
    """_same_as_plain for each of the four coarsened builds (uniforms in
    registers or re-read; constants hoisted out of the id loop or not) that
    allocates."""
    compile_grid = grid if compile_grid == "launch" else compile_grid
    monkeypatch.setenv("PITON_PERSISTENT", "0")
    want, _ = _run(compiler.compile(kernel, sig, cons, grid=compile_grid), grid, values)
    monkeypatch.setenv("PITON_PERSISTENT", str(k))
    real = compiler._compile_tile
    built = 0
    for hoist in (True, False):
        for reload in (False, True):
            def only(fn, signature, constants, name, pid_max, options, *variant, _want=(hoist, reload), **kw):
                if options.get("coarsen", 1) > 1 and (options["coarsen_hoist"], options["coarsen_reload"]) != _want:
                    raise RegAllocError("forced by the test")
                return real(fn, signature, constants, name, pid_max, options, *variant, **kw)
            monkeypatch.setattr(compiler, "_compile_tile", only)
            ck = compiler.compile(kernel, sig, cons, grid=compile_grid)
            monkeypatch.setattr(compiler, "_compile_tile", real)
            if ck.coarsen == 1:
                continue
            built += 1
            got, _ = _run(ck, grid, values)
            for name, v in want.items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_array_equal(got[name].view(np.uint32), v.view(np.uint32),
                                                  err_msg=f"{name}: hoist={hoist} reload={reload}")
    assert built


@piton.jit
def store_first_kernel(x_ptr, a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    """A store before any load (it waits for the previous id's last DMA), then
    two masked stores, the last one rewriting the first one's rows."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(a_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32) + pid, mask=m)
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(b_ptr + offs, x * 3.0, mask=m)
    tl.store(a_ptr + offs, x + 0.5, mask=offs < n - 3)


@pytest.mark.parametrize("k,blocks", [(3, 7), (4, 9), (5, 11)])
def test_store_before_loads(monkeypatch, k, blocks):
    n = blocks * 32 - 7
    _variants(monkeypatch, k, store_first_kernel, {"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32", "n": "i32"},
              {"BLOCK": 32}, (blocks,),
              {"x_ptr": rng.standard_normal(n).astype(np.float32), "a_ptr": np.full(n + 32, -9.0, np.float32),
               "b_ptr": np.full(n + 32, -9.0, np.float32), "n": n})


@piton.jit
def inner_store_kernel(x_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
    """A run-time loop with a store inside, run 0, 1 or 2 times depending on the
    program id, then a store after it."""
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    trips = pid % 3
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in range(0, trips):
        v = tl.load(x_ptr + (pid * 3 + r) * BLOCK + lanes)
        acc += v
        tl.store(out_ptr + (pid * 3 + r) * BLOCK + lanes, v * 2.0)
    tl.store(sum_ptr + pid * BLOCK + lanes, acc + trips)


@pytest.mark.parametrize("k,blocks", [(3, 7), (4, 8)])
def test_store_loop_with_zero_trips(monkeypatch, k, blocks):
    _variants(monkeypatch, k, inner_store_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "sum_ptr": "*fp32"},
              {"BLOCK": 16}, (blocks,),
              {"x_ptr": rng.standard_normal(blocks * 48).astype(np.float32),
               "out_ptr": np.zeros(blocks * 48, np.float32), "sum_ptr": np.zeros(blocks * 16, np.float32)})


@piton.jit
def tile_store_kernel(x_ptr, out_ptr, M, N, stride_x, stride_o, BM: tl.constexpr, BN: tl.constexpr):
    """2-D tiles with a run-time row stride and ragged edges."""
    pid = tl.program_id(0)
    nb = tl.cdiv(N, BN)
    rm = (pid // nb) * BM + tl.arange(0, BM)
    rn = (pid % nb) * BN + tl.arange(0, BN)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptr + rm[:, None] * stride_x + rn[None, :], mask=msk, other=0.0)
    tl.store(out_ptr + rm[:, None] * stride_o + rn[None, :], x * 2.0 + 1.0, mask=msk)


# 0x4000 + 40: the row gaps do not fit the VDW stride field (the out-of-line one-DMA-per-row store)
@pytest.mark.parametrize("stride_o", [40, 0x4000 + 40])
def test_tile_store_with_runtime_row_stride(monkeypatch, stride_o):
    M, N = 13, 37
    grid = (4 * 3,)
    _variants(monkeypatch, 3, tile_store_kernel,
              {"x_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32", "stride_x": "i32", "stride_o": "i32"},
              {"BM": 4, "BN": 16}, grid,
              {"x_ptr": rng.standard_normal(M * N).astype(np.float32),
               "out_ptr": np.full((M - 1) * stride_o + N + 16, -3.0, np.float32), "M": M, "N": N,
               "stride_x": N, "stride_o": stride_o})


@piton.jit
def late_args_kernel(x_ptr, y_ptr, out_ptr, side_ptr, n_iter, scale, BLOCK: tl.constexpr):
    """Arguments used only after a top-level loop (re-read from the uniform list)
    on a 2-D grid, with program_id(1) read again after the loop."""
    pid = tl.program_id(0)
    row = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, n_iter):
        acc += tl.load(x_ptr + i * BLOCK + lanes) * (pid + 1)
    base = (row * tl.num_programs(0) + pid) * BLOCK
    tl.store(out_ptr + base + lanes, acc * scale + tl.load(y_ptr + base + lanes))
    tl.store(side_ptr + row * tl.num_programs(0) + pid, tl.program_id(1) * 1000 + pid)


@pytest.mark.parametrize("k,grid", [(3, (7, 2)), (2, (5, 3))])
def test_arguments_reread_after_a_loop(monkeypatch, k, grid):
    gx, gy = grid
    _variants(monkeypatch, k, late_args_kernel,
              {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "side_ptr": "*i32", "n_iter": "i32",
               "scale": "fp32"}, {"BLOCK": 16}, grid,
              {"x_ptr": rng.standard_normal(48).astype(np.float32),
               "y_ptr": rng.standard_normal(gx * gy * 16).astype(np.float32),
               "out_ptr": np.zeros(gx * gy * 16, np.float32), "side_ptr": np.zeros(gx * gy, np.int32),
               "n_iter": 3, "scale": 0.75})


@piton.jit
def id_slot_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(out_ptr + (pid % 1021) * BLOCK + tl.arange(0, BLOCK), tl.zeros((BLOCK,), dtype=tl.int32) + pid)


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("G", [(1 << 24) + 77, (1 << 27) + 5])
def test_first_id_beyond_24_bits(persistent, k, G):
    """pid0 * k past the 24-bit multiplier range (a few hardware programs of a
    huge grid, including the partial last one)."""
    persistent(k)
    ck = compiler.compile(id_slot_kernel, {"out_ptr": "*i32"}, {"BLOCK": 16}, grid=(G,))
    assert ck.coarsen == k
    hx = -(-G // k)
    expect, hws = {}, []
    for hw in (0, (1 << 24) // k, (1 << 24) // k + 1, hx - 2, hx - 1):
        ids = range(hw * k, min(hw * k + k, G))
        if not any(p % 1021 in expect for p in ids):
            hws.append(hw)
            expect.update((p % 1021, p) for p in ids)
    v = V3D(num_qpus=4, strict=True)
    out = v.mem.alloc(1021 * 16 * 4, 64)
    zero, scratch = v.mem.alloc(64, 64), v.mem.alloc(64, 64)
    unifs = []
    for hw in hws:
        a = v.mem.alloc(4 * len(ck.uniforms), 16)
        v.mem.write_words(a, np.array(uniforms_for(ck, {"out_ptr": np.zeros(1, np.int32)}, {"out_ptr": out},
                                                   (hw, 0, 0), (G, 1, 1), zero, scratch, a), dtype=np.uint32))
        unifs.append(a)
    v.run(v.load(ck.words), unifs)
    got = v.mem.read_array(out, (1021, 16), np.int32)
    assert len(hws) >= 4
    for slot, p in expect.items():
        assert np.all(got[slot] == p), (slot, p)


STUB_RUNTIME = r"""
#include <stdio.h>
#include <stdlib.h>
#include "kern.h"
static piton_buf_t bufs[16];
static unsigned nbufs;
static uint32_t next_bus = 0x40000000u;
piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b = {.handle = nbufs + 1, .bus = next_bus, .cpu = calloc(size, 1), .size = size};
    next_bus += (size + 4095u) & ~4095u;
    printf("ALLOC %u %u\n", b.bus, size);
    bufs[nbufs++] = b;
    return b;
}
void piton_free(piton_buf_t *b) { b->handle = 0; }       /* memory stays readable below */
static const volatile uint32_t *at(uint32_t bus) {
    for (unsigned i = nbufs; i-- > 0;)
        if (bus >= bufs[i].bus && bus < bufs[i].bus + bufs[i].size)
            return (const volatile uint32_t *)bufs[i].cpu + (bus - bufs[i].bus) / 4;
    abort();
}
uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n) {
    (void)code;
    printf("LAUNCH %u\n", n);
    for (unsigned i = 0; i < n; i++) {
        printf("PROGRAM %u", unifs[i]);
        for (unsigned j = 0; j < KERN_NUM_UNIFORMS; j++) printf(" %u", at(unifs[i])[j]);
        printf("\n");
    }
    return 0;
}
uint32_t piton_launch_slots(uint32_t code, const uint32_t *unifs, uint32_t n, uint32_t init, uint32_t fini) {
    printf("SLOTS %u %u\n", init - code, fini - code);
    return piton_launch(code, unifs, n);
}
"""


@pytest.mark.parametrize("kernel,sig,cons,args,grids,k", [
    (fma_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "alpha": "fp32"},
     {"BLOCK": 64}, {"x_ptr": 0x1000, "y_ptr": 0x2000, "out_ptr": 0x3000, "n": 777, "alpha": 1.5},
     [(13, 1, 1), (64, 1, 1), (1, 1, 1)], 4),
    (ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": 2}, {"out_ptr": 0x1000, "flag_ptr": 0x2000},
     [(10, 3, 1), (7, 2, 2)], 3),
    (copy_kernel, COPY_SIG, {"BLOCK": 256}, {"x_ptr": 0x1000, "out_ptr": 0x9000, "n": 2000},
     [(9, 1, 1), (2, 1, 1)], 4),                 # store slots: piton_launch_slots
])
def test_generated_c_launcher(persistent, monkeypatch, tmp_path, kernel, sig, cons, args, grids, k):
    """tools/compile.py's C launcher writes the same uniforms as the Python
    launch paths (program_ids + uniforms_for): ceil(gx / k) programs, pid0 =
    the hardware program, nprog0 = gx. Run with both uniform protocols. A
    kernel with store slots goes through piton_launch_slots, whose init / fini
    programs follow the kernel's code."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    persistent(k)
    real = compiler._compile_tile
    for reload in (False, True):
        monkeypatch.setattr(compiler, "_compile_tile", _compile_tile_only(real, reload))
        ck = compiler.compile(kernel, sig, cons)
        assert ck.coarsen == k and ck.stats["coarsen_reload"] == reload
        header, source = emit_c(ck, "kern")
        assert f"hardware program runs {k} consecutive x ids" in header
        d = tmp_path / str(int(reload))
        d.mkdir()
        (d / "kern.h").write_text(header)
        (d / "kern.c").write_text(source)
        fields = ", ".join(f".{n} = {v}" + ("f" if isinstance(v, float) else "") for n, v in args.items())
        calls = "\n".join(f"    kern_launch(&a, {gx}, {gy}, {gz});" for gx, gy, gz in grids)
        (d / "main.c").write_text(STUB_RUNTIME + f"int main(void) {{\n    kern_args_t a = {{{fields}}};\n"
                                  f"{calls}\n    return 0;\n}}\n")
        runtime = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pi", "runtime"))
        subprocess.run([cc, "-std=c99", "-O1", "-I", runtime, "-I", str(d), str(d / "main.c"), str(d / "kern.c"),
                        "-o", str(d / "run")], check=True, capture_output=True, text=True)
        log = subprocess.run([str(d / "run")], check=True, capture_output=True, text=True).stdout.splitlines()
        zero = int([ln for ln in log if ln.startswith("ALLOC")][1].split()[1])     # code, zero, uniforms
        values = {n: (f2u(v) if isinstance(v, float) else v) for n, v in args.items()}
        launches = [i for i, ln in enumerate(log) if ln.startswith("LAUNCH")]
        assert len(launches) == len(grids)
        sems = slot_semaphores(ck.words)
        assert bool(sems) == (kernel is copy_kernel)
        slot_lines = [ln for ln in log if ln.startswith("SLOTS")]
        if sems:
            init, _ = semaphore_programs(sems)
            assert slot_lines == [f"SLOTS {8 * len(ck.words)} {8 * (len(ck.words) + len(init))}"] * len(grids)
        else:
            assert not slot_lines
        for grid, at in zip(grids, launches):
            ids = list(program_ids(grid, ck.coarsen))
            assert int(log[at].split()[1]) == len(ids) == -(-grid[0] // k) * grid[1] * grid[2]
            for (pid, g), line in zip(ids, log[at + 1:at + 1 + len(ids)]):
                words = [int(w) for w in line.split()[1:]]
                assert words[1:] == uniforms_for(ck, values, {}, pid, g, zero, zero + 2048, words[0])


def _compile_tile_only(real, reload):
    """_compile_tile that refuses the coarsened builds with the other uniform protocol."""
    def build(fn, signature, constants, name, pid_max, options, chunk=0):
        if options.get("coarsen", 1) > 1 and options["coarsen_reload"] != reload:
            raise RegAllocError("forced by the test")
        return real(fn, signature, constants, name, pid_max, options, chunk)
    return build
