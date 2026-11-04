"""Stores through shared 16-row VPM slots under hardware semaphores (PITON_STORE_SLOTS=1).

Every kernel runs in the strict emulator with all 12 QPUs; the contended
timing serialises DMAs on one engine with long, jittered transfers, so the
three QPUs sharing a slot really do wait for each other, and any VPM write
to rows of a DMA that has not been waited for raises a HazardError.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton.backend.lower import SLOT_SEMAPHORES  # noqa: E402
from piton.compiler import compile as pcompile  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import Program  # noqa: E402
from piton.qpu.emulator import V3D, EmulatorError, HazardError, TimingModel  # noqa: E402
from piton.runtime import launcher  # noqa: E402
from piton.runtime.launcher import run_emulator, semaphore_programs, slot_semaphores  # noqa: E402
from kernels import add_kernel, matmul_kernel  # noqa: E402
from primitive_kernels import copy_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402

rng = np.random.default_rng(23)
SIG_COPY = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}


@pytest.fixture
def slots(monkeypatch):
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")


def contended(seed, num_qpus=12, per_row=40):
    """One VDW engine, DMAs of 40 cycles per row plus up to 300 cycles of jitter."""
    return V3D(num_qpus=num_qpus, seed=seed,
               timing=TimingModel(vdw_serial=True, vdw_cycles_per_row=per_row, vdw_jitter=300,
                                  start_jitter=400))


def build(kernel, sig, cons, grid):
    bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)
    return pcompile(kernel, sig, cons, grid=bound)


def launch(ck, grid, values, v3d):
    run_emulator(ck, grid, values, v3d=v3d)
    assert v3d.sems == [0] * 16           # init / kernel / fini leave them as after a reset
    return values


def sems_in(ck):
    return [(i.acquire, i.sema) for i in ck.program.instrs if i.kind == "sem"]


# -- the switch ------------------------------------------------------------------------

def test_on_unless_zero(monkeypatch):
    monkeypatch.setenv("PITON_STORE_SLOTS", "")
    off = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (16,))
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")
    zero = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (16,))
    assert not sems_in(off) and "store_slots" not in off.stats
    assert len(off.program.instrs) == len(zero.program.instrs) and not sems_in(zero)
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")
    on = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (16,))
    assert on.stats["store_slots"] == 1
    # one store: an acquire and a release per slot, each in its own copy
    assert sorted(sems_in(on)) == sorted([(True, s) for s in SLOT_SEMAPHORES] +
                                         [(False, s) for s in SLOT_SEMAPHORES])
    assert slot_semaphores(on.words) == SLOT_SEMAPHORES and slot_semaphores(off.words) == ()
    monkeypatch.delenv("PITON_STORE_SLOTS", raising=False)
    assert build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (16,)).words == on.words


def test_small_stores_keep_private_rows(slots, monkeypatch):
    """A 64-element store fits the 4 private rows: the code is unchanged."""
    on = build(copy_kernel, SIG_COPY, {"BLOCK": 64}, (16,))
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")
    off = build(copy_kernel, SIG_COPY, {"BLOCK": 64}, (16,))
    assert on.words == off.words and "store_slots" not in on.stats


def test_release_waits_and_semaphores_stay_out_of_delay_slots(slots):
    ck = build(add_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
               {"BLOCK_SIZE": 256}, (16,))
    instrs = ck.program.instrs
    for pc, ins in enumerate(instrs):
        if ins.kind != "sem":
            continue
        assert not any(instrs[p].kind == "branch" for p in range(max(0, pc - 3), pc)), pc
        if not ins.acquire:       # the slot is free only once its last DMA has been read out
            prev = instrs[pc - 1]
            assert prev.kind == "alu" and isa.VW_WAIT in [s for op in (prev.add, prev.mul) if op
                                                          for s in op.srcs()], pc


# -- suite kernels -----------------------------------------------------------------------

AFFECTED = ["copy", "add", "mul", "fma", "scaled_relu", "transpose", "matmul_8x64x4", "matmul_8x64x8",
            "exp", "exp2", "log", "log2", "recip", "rsqrt", "sqrt"]     # (the last 7: rerolled loops)
UNAFFECTED = ["idiv", "row_sum", "row_max", "dense_relu", "compute_peak"]


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("name", AFFECTED + ["softmax"])
def test_suite_case_contended(slots, name, seed):
    from perf.suite import CASES_BY_NAME, compile_case, default_check
    case = CASES_BY_NAME[name]
    size = "pi" if name == "softmax" else "emu"     # softmax rows are 256 wide only at Pi size
    run = case.make(size)
    if name == "softmax":
        run.grid = (24,)                            # 24 of its 256 rows
        run.expected = {k: v[:24 * 256] for k, v in run.expected.items()}
    ck, _ = compile_case(case, size, run)
    assert ck.stats.get("store_slots") == 1
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    launch(ck, run.grid, values, contended(seed))
    for out, expected in run.expected.items():
        got = values[out][:expected.size]
        assert default_check(run, out, got, expected) is None


@pytest.mark.parametrize("name", UNAFFECTED)
def test_suite_cases_without_large_stores(slots, name, monkeypatch):
    """Stores of at most 4 rows per DMA (and idiv's 16-element rerolled chunks;
    PITON_FAST_IDIV queues its stores, tests/test_fast_idiv.py) keep the private
    rows: no semaphores, nothing for the runtime to set up."""
    from perf.suite import CASES_BY_NAME, compile_case
    if name == "idiv":
        monkeypatch.setenv("PITON_FAST_IDIV", "0")
    ck, _ = compile_case(CASES_BY_NAME[name], "emu")
    assert "store_slots" not in ck.stats and not sems_in(ck) and slot_semaphores(ck.words) == ()


# -- 1-D: ragged tails, tiny and large grids ---------------------------------------------------

@pytest.mark.parametrize("n,extra", [(1, 0), (15, 0), (16, 1), (17, 0), (255, 0), (256, 2), (257, 0),
                                     (1000, 1), (4096 - 37, 3)])
def test_ragged_1d(slots, n, extra):
    """Masked tails (full rows, then a partial row), and programs past the end that store nothing."""
    grid = (piton.cdiv(n, 256) + extra,)
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (32,))
    x = rng.standard_normal(n).astype(np.float32)
    out = np.full(n + 64, 7.0, np.float32)          # padding after n must stay untouched
    launch(ck, grid, {"x_ptr": x, "out_ptr": out, "n": n}, contended(n))
    np.testing.assert_array_equal(out[:n], x)
    np.testing.assert_array_equal(out[n:], 7.0)


@piton.jit
def copy_sum_k(x_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x)
    tl.store(sum_ptr + tl.program_id(0), tl.sum(x, axis=0))      # (a reduction: not rerolled)


@pytest.mark.parametrize("programs,num_qpus", [(1, 12), (3, 1), (7, 5), (300, 12)])
def test_grids_and_qpu_counts(slots, programs, num_qpus):
    """One program, one QPU (no contention), 5 QPUs (slot 0 shared by QPUs 0 and 4),
    300 programs; 512 elements, so each store is two 1 KiB DMAs from one slot."""
    block = 512
    ck = build(copy_sum_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "sum_ptr": "*fp32"}, {"BLOCK": block},
               (programs,))
    assert ck.stats.get("store_slots") == 1 and not ck.stats["rerolled"]
    x = rng.standard_normal(programs * block).astype(np.float32)
    out, sums = np.zeros_like(x), np.zeros(programs, np.float32)
    launch(ck, (programs,), {"x_ptr": x, "out_ptr": out, "sum_ptr": sums},
           contended(programs, num_qpus, per_row=8))
    np.testing.assert_array_equal(out, x)
    np.testing.assert_allclose(sums, x.reshape(programs, block).sum(axis=1), rtol=1e-5, atol=1e-4)


# -- 2-D: run-time row stride (fast / one DMA per row), constant strides, masks -------------

@piton.jit
def tile_k(x_ptr, out_ptr, n_rows, n_cols, stride, R: tl.constexpr, L: tl.constexpr, MASK: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, L)
    x = tl.load(x_ptr + rows[:, None] * L + cols[None, :])
    if MASK:
        m = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        tl.store(out_ptr + rows[:, None] * stride + cols[None, :], x * 2.0, mask=m)
    else:
        tl.store(out_ptr + rows[:, None] * stride + cols[None, :], x * 2.0)


@piton.jit
def tile_const_k(x_ptr, out_ptr, R: tl.constexpr, L: tl.constexpr, STRIDE: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, L)
    x = tl.load(x_ptr + rows[:, None] * L + cols[None, :])
    tl.store(out_ptr + rows[:, None] * STRIDE + cols[None, :], x * 0.5)


def _tile_ref(x, n_rows, n_cols, stride, out, R, L):
    for r in range(x.shape[0]):
        for c in range(L):
            if r < n_rows and c < n_cols:
                out[r * stride + c] = x[r, c] * np.float32(2.0)
    return out


@pytest.mark.parametrize("R,L", [(16, 16), (16, 32), (8, 64)])
@pytest.mark.parametrize("stride", [48, 20000])          # VDW stride field fits / does not
@pytest.mark.parametrize("mask", [False, True])
def test_2d_runtime_stride(slots, R, L, stride, mask):
    tiles = 3
    n_rows, n_cols = (tiles * R - 5, L - 3) if mask else (tiles * R, L)
    stride = max(stride, L)
    x = rng.standard_normal((tiles * R, L)).astype(np.float32)
    out = np.full(stride * tiles * R, -9.0, np.float32)
    ck = build(tile_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_rows": "i32", "n_cols": "i32", "stride": "i32"},
               {"R": R, "L": L, "MASK": mask}, (tiles,))
    assert ck.stats.get("store_slots") == 1
    launch(ck, (tiles,), {"x_ptr": x, "out_ptr": out, "n_rows": n_rows, "n_cols": n_cols, "stride": stride},
           contended(stride + R))
    want = _tile_ref(x, n_rows, n_cols, stride, np.full_like(out, -9.0), R, L)
    np.testing.assert_array_equal(out, want)


@pytest.mark.parametrize("stride", [16, 40, 20000])       # contiguous / gaps / one DMA per row
def test_2d_constant_stride(slots, stride):
    R, L, tiles = 16, 16, 4
    x = rng.standard_normal((tiles * R, L)).astype(np.float32)
    out = np.zeros(stride * tiles * R, np.float32)
    ck = build(tile_const_k, {"x_ptr": "*fp32", "out_ptr": "*fp32"}, {"R": R, "L": L, "STRIDE": stride}, (tiles,))
    launch(ck, (tiles,), {"x_ptr": x, "out_ptr": out}, contended(stride))
    np.testing.assert_array_equal(out.reshape(tiles * R, stride)[:, :L], x * np.float32(0.5))


@piton.jit
def tile_const_mask_k(x_ptr, out_ptr, n_rows, R: tl.constexpr, L: tl.constexpr, STRIDE: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, L)
    x = tl.load(x_ptr + rows[:, None] * L + cols[None, :])
    tl.store(out_ptr + rows[:, None] * STRIDE + cols[None, :], x * 0.5, mask=rows[:, None] < n_rows)


@pytest.mark.parametrize("stride", [40, 20000])
def test_2d_constant_stride_masked(slots, stride):
    """A run-time number of rows; with 20000 every row is its own guarded DMA."""
    R, L, tiles = 16, 16, 3
    n_rows = tiles * R - 7
    x = rng.standard_normal((tiles * R, L)).astype(np.float32)
    out = np.full(stride * tiles * R, -1.0, np.float32)
    ck = build(tile_const_mask_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_rows": "i32"},
               {"R": R, "L": L, "STRIDE": stride}, (tiles,))
    assert ck.stats.get("store_slots") == 1
    launch(ck, (tiles,), {"x_ptr": x, "out_ptr": out, "n_rows": n_rows}, contended(stride + 1))
    got = out.reshape(tiles * R, stride)
    np.testing.assert_array_equal(got[:n_rows, :L], x[:n_rows] * np.float32(0.5))
    np.testing.assert_array_equal(got[n_rows:], -1.0)
    np.testing.assert_array_equal(got[:, L:], -1.0)


@piton.jit
def tile_neg_k(x_ptr, out_ptr, base, stride, R: tl.constexpr, L: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, L)
    x = tl.load(x_ptr + rows[:, None] * L + cols[None, :])
    tl.store(out_ptr + base + rows[:, None] * stride + cols[None, :], x * 2.0)


@pytest.mark.parametrize("L", [16, 32])
def test_2d_negative_runtime_stride(slots, L):
    """Rows stored in reverse order: the gap does not fit the VDW stride field,
    so the slow version stores one DMA per row from the slot."""
    R, tiles = 16, 3
    stride = -(L + 16)
    base = (tiles * R - 1) * -stride
    x = rng.standard_normal((tiles * R, L)).astype(np.float32)
    out = np.full(base + L, -3.0, np.float32)
    ck = build(tile_neg_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "base": "i32", "stride": "i32"},
               {"R": R, "L": L}, (tiles,))
    assert ck.stats.get("store_slots") == 1
    launch(ck, (tiles,), {"x_ptr": x, "out_ptr": out, "base": base, "stride": stride}, contended(L))
    want = np.full_like(out, -3.0)
    for r in range(tiles * R):
        want[base + r * stride:base + r * stride + L] = x[r] * np.float32(2.0)
    np.testing.assert_array_equal(out, want)


# -- scalar and scatter stores, stores in loops and branches ------------------------------------

@piton.jit
def mixed_k(x_ptr, out_ptr, sum_ptr, odd_ptr, n, limit, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, x * 2.0, mask=m)                        # 16 rows: one slot DMA
    tl.store(sum_ptr + pid, tl.sum(x, axis=0))                       # scalar
    tl.store(sum_ptr + 64 + pid, tl.max(x, axis=0), mask=pid < limit)   # masked scalar
    head = tl.load(x_ptr + pid * BLOCK + tl.arange(0, 16))
    tl.store(odd_ptr + pid * 32 + 2 * tl.arange(0, 16), head * 4.0)     # stride 2: scatter


def test_scalar_and_scatter_stores(slots):
    programs, block = 20, 256
    n = programs * block - 100
    x = rng.standard_normal(programs * block).astype(np.float32)
    out = np.zeros(n, np.float32)
    sums = np.zeros(128, np.float32)
    odd = np.zeros(programs * 32, np.float32)
    ck = build(mixed_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "sum_ptr": "*fp32", "odd_ptr": "*fp32",
                         "n": "i32", "limit": "i32"}, {"BLOCK": block}, (programs,))
    assert ck.stats.get("store_slots") == 1
    launch(ck, (programs,), {"x_ptr": x, "out_ptr": out, "sum_ptr": sums, "odd_ptr": odd, "n": n,
                             "limit": 13}, contended(5))
    np.testing.assert_array_equal(out, x[:n] * np.float32(2.0))
    xm = np.where(np.arange(programs * block) < n, x, 0.0).reshape(programs, block)
    np.testing.assert_allclose(sums[:programs], xm.sum(axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(sums[64:64 + 13], xm.max(axis=1)[:13])
    np.testing.assert_array_equal(sums[64 + 13:], 0.0)
    np.testing.assert_array_equal(odd.reshape(programs, 16, 2)[:, :, 0],
                                  x.reshape(programs, block)[:, :16] * np.float32(4.0))
    np.testing.assert_array_equal(odd.reshape(programs, 16, 2)[:, :, 1], 0.0)


@piton.jit
def loop_branch_k(x_ptr, out_ptr, n, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(ITERS):
        offs = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.store(out_ptr + offs, x * 4.0, mask=m)
    tail = pid * ITERS * BLOCK + tl.arange(0, BLOCK)
    if pid % 2 == 0:
        tl.store(out_ptr + n + tail, tl.load(x_ptr + tail) * 2.0)
    else:
        tl.store(out_ptr + n + tail, tl.load(x_ptr + tail) * 0.5)


def test_stores_in_loops_and_branches(slots):
    programs, iters, block = 9, 5, 256
    n = programs * iters * block - 300
    x = rng.standard_normal(programs * iters * block).astype(np.float32)
    out = np.zeros(n + programs * iters * block, np.float32)
    ck = build(loop_branch_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
               {"ITERS": iters, "BLOCK": block}, (programs,))
    assert ck.stats.get("store_slots") == 1
    launch(ck, (programs,), {"x_ptr": x, "out_ptr": out, "n": n}, contended(9))
    np.testing.assert_array_equal(out[:n], x[:n] * np.float32(4.0))
    for p in range(programs):
        lo = p * iters * block
        scale = np.float32(2.0 if p % 2 == 0 else 0.5)
        np.testing.assert_array_equal(out[n + lo:n + lo + block], x[lo:lo + block] * scale)


# -- rerolled loops: the block's store after the loop (reroll.py queue) -----------------------

SIG_SFU = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}


def _sacq_count(ck):
    return sum(1 for acquire, _ in sems_in(ck) if acquire)


@pytest.mark.parametrize("n,extra", [(1, 0), (100, 1), (256, 0), (1000, 2), (4096 - 37, 0)])
def test_rerolled_queue_ragged(slots, n, extra):
    """exp over BLOCK=256 is rerolled into four 64-element chunks; with slots
    the chunks are shifted into carried registers and the masked block is
    stored once after the loop (one acquire per slot: one store)."""
    ck = build(sfu_kernel, SIG_SFU, {"OP": 4, "BLOCK": 256}, (32,))
    assert ck.stats["rerolled"] == 64 and ck.stats.get("store_slots") == 1
    assert "piton.shift_in" in ck.asm["ttir_rerolled"] and _sacq_count(ck) == 4
    x = rng.uniform(-8, 8, n).astype(np.float32)
    out = np.full(n + 64, 5.0, np.float32)
    launch(ck, (piton.cdiv(n, 256) + extra,), {"x_ptr": x, "out_ptr": out, "n": n}, contended(n + 3))
    np.testing.assert_allclose(out[:n], np.exp(x.astype(np.float64)), rtol=4e-6, atol=1e-30)
    np.testing.assert_array_equal(out[n:], 5.0)


def test_reroll_without_switch_has_no_queue(monkeypatch):
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")
    ck = build(sfu_kernel, SIG_SFU, {"OP": 4, "BLOCK": 256}, (32,))
    assert ck.stats["rerolled"] == 64 and "piton.shift_in" not in ck.asm["ttir_rerolled"] and not sems_in(ck)


@piton.jit
def exp_where_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + offs, tl.exp(x), mask=(offs < n) & (x > 0.0))


def test_rerolled_store_mask_from_loaded_data(slots):
    """A mask computed from loaded data would have to be carried too: the
    store stays in the loop (64-element chunks, private rows)."""
    ck = build(exp_where_k, SIG_SFU, {"BLOCK": 256}, (8,))
    assert ck.stats["rerolled"] and "piton.shift_in" not in ck.asm["ttir_rerolled"]
    n = 1500
    x = rng.uniform(-4, 4, n).astype(np.float32)
    out = np.full(n, -2.0, np.float32)
    launch(ck, (6,), {"x_ptr": x, "out_ptr": out, "n": n}, contended(2))
    want = np.where(x > 0, np.exp(x.astype(np.float64)), -2.0)
    np.testing.assert_allclose(out, want, rtol=4e-6, atol=1e-30)


@piton.jit
def exp_sqrt_k(x_ptr, e_ptr, s_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    tl.store(e_ptr + offs, tl.exp(x), mask=m)
    tl.store(s_ptr + offs, tl.sqrt(x * x), mask=m)


def test_rerolled_two_stores(slots):
    """Two stored blocks: two queues (or, when they do not fit the registers,
    the build without slots); either way the results are right."""
    ck = build(exp_sqrt_k, {"x_ptr": "*fp32", "e_ptr": "*fp32", "s_ptr": "*fp32", "n": "i32"},
               {"BLOCK": 256}, (8,))
    assert ck.stats["rerolled"]
    n = 1800
    x = rng.uniform(-6, 6, n).astype(np.float32)
    e, sq = np.zeros(n, np.float32), np.zeros(n, np.float32)
    launch(ck, (8,), {"x_ptr": x, "e_ptr": e, "s_ptr": sq, "n": n}, contended(7))
    np.testing.assert_allclose(e, np.exp(x.astype(np.float64)), rtol=4e-6, atol=1e-30)
    np.testing.assert_allclose(sq, np.abs(x), rtol=4e-6)


# -- the protocol around a launch, and what the emulator catches ----------------------------

def _copy_launch(n=24 * 256):
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (n // 256,))
    x = rng.standard_normal(n).astype(np.float32)
    return ck, x, {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "n": n}


def test_init_and_fini_programs():
    init, fini = semaphore_programs(SLOT_SEMAPHORES)
    v = V3D()
    v.run(v.load(init), [v.mem.alloc(64)])
    assert v.sems[:4] == [1, 1, 1, 1] and v.sems[4:] == [0] * 12
    v.run(v.load(fini), [v.mem.alloc(64)])
    assert v.sems == [0] * 16


def test_semaphores_persist_across_launches(slots):
    ck, x, values = _copy_launch()
    v = contended(3)
    for _ in range(3):                    # one V3D: the counts carry over, as on the Pi
        values["out_ptr"][:] = 0
        launch(ck, (24,), values, v)
        np.testing.assert_array_equal(values["out_ptr"], x)


def test_missing_init_is_a_deadlock(slots, monkeypatch):
    """A slot kernel launched without init waits for semaphores nobody releases."""
    ck, x, values = _copy_launch()
    monkeypatch.setattr(launcher, "slot_semaphores", lambda words: ())
    with pytest.raises(EmulatorError, match="deadlock: QPU0 waits for the semaphore 0"):
        run_emulator(ck, (24,), values)


def test_shared_slot_without_exclusion_is_caught(slots):
    """With a slot semaphore at 2, two QPUs fill one slot at once: the emulator
    sees a QPU overwrite rows another QPU's DMA has not finished reading."""
    ck, x, values = _copy_launch(48 * 256)
    v = contended(4, per_row=200)
    v.sems[SLOT_SEMAPHORES[0]] = 1        # init makes it 2
    with pytest.raises(HazardError, match="has not been waited for"):
        run_emulator(ck, (48,), values, v3d=v)


def test_vpm_rows_stay_reserved_until_the_dma_is_waited_for():
    v = V3D(num_qpus=2)
    a, b = v.qpus
    for q in (a, b):
        q.program, q.pc, q.running = [(isa.Instr.nop(), None)], 0, True
    out = v.mem.alloc(1024, 64)
    a.write_vw_setup(0x00001A00 | 16)                       # VPM row 16
    a.write_vpm(np.arange(16, dtype=np.uint32))
    a.write_vw_setup(0x80000000 | (1 << 23) | (16 << 16) | 0x4000 | (16 << 7))
    a.start_vdw(out)
    b.write_vw_setup(0x00001A00 | 17)                       # another row: fine
    b.write_vpm(np.zeros(16, np.uint32))
    b.write_vw_setup(0x00001A00 | 16)
    with pytest.raises(HazardError, match="VPM row 16 while QPU0"):
        b.write_vpm(np.zeros(16, np.uint32))
    a.wait_vdw()
    with pytest.raises(HazardError):                        # a's wait returns only when the DMA ends
        b.write_vpm(np.zeros(16, np.uint32))
    b.cycle = a.cycle
    b.write_vpm(np.zeros(16, np.uint32))                    # waited for: the row is free again
    np.testing.assert_array_equal(v.mem.read_array(out, (16,), np.uint32), np.arange(16))


def test_dma_of_rows_another_qpu_overwrote():
    """The other direction of the race: rows filled by one QPU and overwritten
    by another before the first QPU's DMA has read them."""
    v = V3D(num_qpus=2)
    a, b = v.qpus
    for q in (a, b):
        q.program, q.pc, q.running = [(isa.Instr.nop(), None)], 0, True
    out = v.mem.alloc(1024, 64)
    a.write_vw_setup(0x00001A00 | 32)                       # rows 32, 33
    a.write_vpm(np.arange(16, dtype=np.uint32))
    a.write_vpm(np.arange(16, dtype=np.uint32))
    b.write_vw_setup(0x00001A00 | 33)                       # no DMA reads row 33 yet: not caught here
    b.write_vpm(np.zeros(16, np.uint32))
    a.write_vw_setup(0x80000000 | (1 << 23) | (16 << 16) | 0x4000 | (32 << 7))
    a.start_vdw(out)                                        # row 32 only: still a's data
    a.wait_vdw()
    a.write_vw_setup(0x80000000 | (2 << 23) | (16 << 16) | 0x4000 | (32 << 7))
    with pytest.raises(HazardError, match="VPM row 33, last written by QPU1"):
        a.start_vdw(out)


def test_waited_reservations_end_with_their_run(slots):
    """Reservations from a finished launch must not leak into the next one,
    whose cycles start again at 0 (here 3 programs: QPUs 3-11 stay idle and
    keep the state of their last program of the 48-program launch)."""
    ck, x, values = _copy_launch(48 * 256)
    v = contended(6)
    launch(ck, (48,), values, v)
    assert any(q.vdw_rows_until for q in v.qpus[3:])
    small = {"x_ptr": x[:3 * 256], "out_ptr": np.zeros(3 * 256, np.float32), "n": 3 * 256}
    launch(ck, (3,), small, v)
    np.testing.assert_array_equal(small["out_ptr"], x[:3 * 256])


# -- the Pi side -----------------------------------------------------------------------------

def test_generated_launch_runs_init_kernel_fini(slots, tmp_path, monkeypatch):
    """The generated firmware calls piton_launch_slots with the init / fini
    programs it embedded after the kernel; run it against a stand-in runtime."""
    monkeypatch.setenv("PITON_SLOTS_ONCE", "0")      # the per-launch init / fini (test_slots_once_launches: the other)
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    from piton.runtime.pi import Buffer, PiRunner
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (4,))
    sems = slot_semaphores(ck.words)
    runner = object.__new__(PiRunner)
    source = runner._generate(ck.words, [Buffer("x", 4096), Buffer("out", 4096)],
                              [[("buf", "x", 0), ("buf", "out", 0), 1024, p, 0] for p in range(4)],
                              ["out"], 2, "slots", {}, semaphores=sems)
    assert "piton_launch_slots(" in source and " piton_launch(" not in source
    source = source.replace('#include "rpi.h"', '''#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define printk printf
static uint32_t timer_get_usec(void) { static uint32_t t; return ++t; }
''')
    source = source.replace('#include "piton_rt.h"', '''#include "piton_rt.h"
static uint32_t *memory;
void piton_init(void) {}
uint32_t piton_qpu_count(void) { return 12; }
piton_buf_t piton_alloc(uint32_t size) {
    memory = calloc(size, 1);
    return (piton_buf_t){.bus = 0x40000000, .cpu = memory, .size = size};
}
void piton_free(piton_buf_t *b) { free((void *)b->cpu); }
static void dump(const char *what, uint32_t code, unsigned words) {
    printf("%s", what);
    for (unsigned i = 0; i < words; i++) printf(" %08x", memory[(code - 0x40000000) / 4 + i]);
    printf("\\n");
}
uint32_t piton_launch_slots(uint32_t code, const uint32_t *unifs, uint32_t n, uint32_t init, uint32_t fini) {
    dump("INIT", init, 2 * INIT_WORDS);
    dump("KERNEL", code, 2 * KERNEL_WORDS);
    dump("FINI", fini, 2 * INIT_WORDS);
    printf("PROGRAMS %u\\n", n);
    return 5;
}
''')
    init, fini = semaphore_programs(sems)
    main = tmp_path / "main.c"
    main.write_text(f"#define INIT_WORDS {len(init)}\n#define KERNEL_WORDS {len(ck.words)}\n" + source +
                    "\nint main(void) { notmain(); return 0; }\n")
    runtime_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pi", "runtime"))
    subprocess.run([cc, "-std=c99", "-O2", "-I", runtime_dir, str(main), "-o", str(tmp_path / "run")],
                   check=True, capture_output=True, text=True)
    log = subprocess.run([str(tmp_path / "run")], check=True, capture_output=True, text=True).stdout

    def words(tag):
        line = next(x for x in log.splitlines() if x.startswith(tag + " "))
        lo_hi = [int(t, 16) for t in line.split()[1:]]
        return [lo | (hi << 32) for lo, hi in zip(lo_hi[0::2], lo_hi[1::2])]
    assert words("INIT") == init and words("FINI") == fini and words("KERNEL") == list(ck.words)
    assert log.count("PROGRAMS 4") == 2 and log.count("PITON_TIME 5") == 2


def test_aot_launcher_uses_launch_slots(slots, monkeypatch):
    from piton.tools.compile import emit_c
    monkeypatch.setenv("PITON_SLOTS_ONCE", "0")      # the per-launch init / fini (test_slots_once_launches: the other)
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (4,))
    header, body = emit_c(ck, "copyk")
    init, fini = semaphore_programs(slot_semaphores(ck.words))
    total = len(ck.words) + len(init) + len(fini)
    assert f"copyk_code[{2 * total}]" in header and f"copyk_code[{2 * total}]" in body
    assert (f"piton_launch_slots(code_buf.bus, unif_ptrs, n, code_buf.bus + {8 * len(ck.words)}, "
            f"code_buf.bus + {8 * (len(ck.words) + len(init))})") in body
    w = init[0]
    assert f"0x{w & 0xFFFFFFFF:08x}, 0x{w >> 32:08x}," in body


def test_slot_kernel_qasm_assembles_with_vc4asm(slots):
    from piton.qpu.asm import find_vc4asm, vc4asm_assemble
    if find_vc4asm() is None:
        pytest.skip("vc4asm not installed")
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (4,))
    words, _ = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert list(words) == list(ck.words)
    for prog in semaphore_programs(SLOT_SEMAPHORES):
        text = Program("p", [isa.decode(w) for w in prog]).qasm()
        assert list(vc4asm_assemble(text, verify=True)[0]) == list(prog)


def test_slots_once_launches(slots, monkeypatch):
    """PITON_SLOTS_ONCE=1: both generated launchers call piton_launch_slots_once
    with the init program only (pi/runtime/piton_rt.c runs it once per boot)."""
    from piton.runtime.pi import Buffer, PiRunner
    from piton.tools.compile import emit_c
    ck = build(copy_kernel, SIG_COPY, {"BLOCK": 256}, (4,))
    sems = slot_semaphores(ck.words)
    runner = object.__new__(PiRunner)
    words = 8 * len(ck.words)
    for once in ("0", "1"):
        monkeypatch.setenv("PITON_SLOTS_ONCE", once)
        source = runner._generate(ck.words, [Buffer("x", 4096), Buffer("out", 4096)],
                                  [[("buf", "x", 0), ("buf", "out", 0), 1024, p, 0] for p in range(4)],
                                  ["out"], 2, "slots", {}, semaphores=sems)
        _, body = emit_c(ck, "copyk")
        if once == "1":
            assert f"piton_launch_slots_once(m.bus, unif_ptrs, 4, m.bus + {words}u);" in source
            assert f"piton_launch_slots_once(code_buf.bus, unif_ptrs, n, code_buf.bus + {words});" in body
        else:
            assert "piton_launch_slots_once" not in source + body


MATMUL_SIG = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
              "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}


def _matmul(n):
    cons = dict(K=n, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=8, GROUP_SIZE_M=4)
    return pcompile(matmul_kernel, MATMUL_SIG, cons, grid=((n // 8) * (n // 64),))


def test_sparse_stores_keep_private_rows(monkeypatch):
    """A tile stored once after a long K loop keeps the private rows (a slot
    store waits for its DMAs; Pi: 1024^3 0.95x with slots, 256^3 1.05x):
    compiler.SLOTS_MIN_STORE_BYTES. PITON_STORE_SLOTS=all keeps the slots."""
    monkeypatch.delenv("PITON_STORE_SLOTS", raising=False)
    assert _matmul(256).stats.get("store_slots") == 1             # 0.17 bytes stored per instruction
    for n in (512, 1024):                                         # 0.08, 0.04
        ck = _matmul(n)
        assert "store_slots" not in ck.stats and not sems_in(ck)
    monkeypatch.setenv("PITON_STORE_SLOTS", "all")
    assert _matmul(1024).stats.get("store_slots") == 1

