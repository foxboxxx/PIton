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
from piton.runtime.launcher import program_ids, run_emulator, uniforms_for
from piton.tools.compile import emit_c

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from kernels import add_kernel, matmul_kernel, row_sum_kernel  # noqa: E402
from primitive_kernels import copy_kernel, fma_kernel, idiv_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402

ADD_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
COPY_SIG = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
rng = np.random.default_rng(5)


@pytest.fixture
def persistent(monkeypatch):
    def set_k(value):
        if value is None:
            monkeypatch.delenv("PITON_PERSISTENT", raising=False)
        else:
            monkeypatch.setenv("PITON_PERSISTENT", str(value))
    return set_k


def _fail_builds(monkeypatch, when):
    """Make _compile_tile raise RegAllocError for the option sets `when` accepts."""
    real = compiler._compile_tile

    def build(fn, signature, constants, name, pid_max, options, chunk=0):
        if when(options):
            raise RegAllocError("forced by the test")
        return real(fn, signature, constants, name, pid_max, options, chunk)
    monkeypatch.setattr(compiler, "_compile_tile", build)


def _run(ck, grid, values):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    stats = run_emulator(ck, grid, values)
    return values, stats


def _same_as_plain(persistent, k, kernel, sig, cons, grid, values, compile_grid=None):
    """Compile plain and with PITON_PERSISTENT=k, run both on `grid`; every
    array must come back bit-identical. Returns (coarsened kernel, outputs, stats)."""
    compile_grid = grid if compile_grid is None else compile_grid
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


@pytest.mark.parametrize("kernel,cons", [(sfu_kernel, {"OP": 4, "BLOCK": 256}), (copy_kernel, {"BLOCK": 64})])
def test_last_store_overlaps_the_next_id(persistent, kernel, cons):
    """Rerolled (exp) and straight-line versioned (copy) bodies: no wait for the
    previous id's DMA before the next id's loads."""
    n = 16 * cons["BLOCK"]
    x = rng.uniform(-8, 8, n).astype(np.float32)
    ck, got, _, _ = _same_as_plain(persistent, 4, kernel, COPY_SIG, cons, (16,),
                                   {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "n": n})
    assert ck.coarsen == 4 and bool(ck.stats["rerolled"]) == (kernel is sfu_kernel)
    assert _first_wait_after_loads(ck)
    if kernel is sfu_kernel:
        np.testing.assert_allclose(got["out_ptr"], np.exp(x.astype(np.float64)), rtol=4e-6)
    else:
        np.testing.assert_array_equal(got["out_ptr"], x)


def test_falls_back_when_no_version_allocates(persistent, monkeypatch):
    persistent(None)
    plain = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    persistent(1)
    _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    assert ck.coarsen == 1 and "coarsen" not in ck.stats
    assert list(ck.words) == list(plain.words) and ck.uniforms == plain.uniforms


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
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (4096,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (512,), 4),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (256,), 2),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (255,), 1),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, None, 1),
    ("1", add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, (1024, 2), 1),
    # long program ids (exact division: ~6000 instructions) and run-time loops: at most 2
    ("1", idiv_kernel, IDIV_SIG, {"BLOCK": 256}, (1024,), 2),
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


def test_id_length():
    """Per-id instruction estimate: straight-line length, times constant trip
    counts (the reroll chunk loop), unknown for run-time loops."""
    from piton import frontend
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    ck = compiler.compile(sfu_kernel, sig, {"OP": 4, "BLOCK": 256})
    assert ck.stats["rerolled"] == 64
    tile = frontend.generate(sfu_kernel.fn, sig, {"OP": 4, "BLOCK": 256})
    from piton import reroll
    reroll.run(tile, 64)
    assert compiler._id_length(tile, ck.program) == 4 * len(ck.program.instrs)
    tile = frontend.generate(row_sum_kernel.fn, ROW_SIG, {"BLOCK_M": 4, "BLOCK_N": 64})
    assert compiler._id_length(tile, ck.program) is None


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


def test_dma_running_at_thread_end_is_a_hazard():
    """The emulator rejects a program that ends while its VDW DMA still runs
    (the final wait the coarsened loop relies on). The last block is partial,
    so it takes the masked store path, which leaves its DMA to that wait."""
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
"""


@pytest.mark.parametrize("kernel,sig,cons,args,grids,k", [
    (fma_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "alpha": "fp32"},
     {"BLOCK": 64}, {"x_ptr": 0x1000, "y_ptr": 0x2000, "out_ptr": 0x3000, "n": 777, "alpha": 1.5},
     [(13, 1, 1), (64, 1, 1), (1, 1, 1)], 4),
    (ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": 2}, {"out_ptr": 0x1000, "flag_ptr": 0x2000},
     [(10, 3, 1), (7, 2, 2)], 3),
])
def test_generated_c_launcher(persistent, monkeypatch, tmp_path, kernel, sig, cons, args, grids, k):
    """tools/compile.py's C launcher writes the same uniforms as the Python
    launch paths (program_ids + uniforms_for): ceil(gx / k) programs, pid0 =
    the hardware program, nprog0 = gx. Run with both uniform protocols."""
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
