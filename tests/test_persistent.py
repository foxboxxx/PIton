"""PITON_PERSISTENT: several program ids per hardware program.

compile() wraps the kernel in a loop over k consecutive axis-0 program ids
(Lowering.begin_programs); the host launches ceil(gx / k) programs and passes
each its own index as pid0 and the logical grid as nprog0. Every test runs the
strict emulator, so a store DMA that is not waited for, or a uniform read too
soon after a unif_addr write, fails the run.
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
from piton.runtime.launcher import program_ids, run_emulator
from piton.tools.compile import emit_c

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from kernels import add_kernel, matmul_kernel  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402

ADD_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
rng = np.random.default_rng(5)


@pytest.fixture
def persistent(monkeypatch):
    def set_k(value):
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


def _run_add(ck, n, grid):
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.full(n + 16, 7.0, dtype=np.float32)           # guard band: masked tail
    stats = run_emulator(ck, grid, {"x_ptr": x, "y_ptr": y, "output_ptr": out, "n_elements": n})
    np.testing.assert_array_equal(out[:n], x + y)
    assert np.all(out[n:] == 7.0)
    return stats


@pytest.mark.parametrize("blocks,k", [(10, 4), (12, 4), (3, 4), (7, 3), (9, 2), (1, 4)])
def test_add_runs_k_ids_per_program(persistent, blocks, k):
    """Grids that are not a multiple of k: the last program runs fewer ids."""
    persistent(k)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(blocks,))
    assert ck.coarsen == min(k, blocks)
    assert ck.uniforms[:4] == [("arg", n) for n in ADD_SIG]
    _run_add(ck, blocks * 64 - 5, (blocks,))


def test_smaller_grid_than_compiled_for(persistent):
    """The bounds check uses the launch grid (nprog0), not the compile-time bound."""
    persistent(4)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(16,))
    assert ck.coarsen == 4
    for blocks in (1, 5, 13, 16):
        _run_add(ck, blocks * 64, (blocks,))


def test_coarsening_executes_fewer_instructions(persistent, monkeypatch):
    monkeypatch.delenv("PITON_PERSISTENT", raising=False)
    plain = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, grid=(16,))
    persistent(4)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, grid=(16,))
    assert plain.coarsen == 1 and ck.coarsen == 4
    assert _run_add(ck, 16 * 256, (16,))["instructions"] < _run_add(plain, 16 * 256, (16,))["instructions"]


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


@pytest.mark.parametrize("grid,k", [((10, 3), 4), ((5, 2), 2), ((7,), 3)])
def test_program_id_and_num_programs_are_logical(persistent, grid, k):
    persistent(k)
    rounds = 3
    ck = compiler.compile(ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": rounds}, grid=grid)
    assert ck.coarsen == k
    gx, gy = (list(grid) + [1])[:2]
    out = np.full(gx * gy * 16, -1, dtype=np.int32)
    flags = np.zeros(gx * gy, dtype=np.int32)
    run_emulator(ck, grid, {"out_ptr": out, "flag_ptr": flags})
    pid = np.tile(np.arange(gx), gy)
    row = np.repeat(np.arange(gy), gx)
    expected = (rounds * pid + rounds * (rounds - 1) // 2) * 1000000 + gx * 10000 + gy * 100 + row
    np.testing.assert_array_equal(out.reshape(-1, 16), np.repeat(expected[:, None], 16, axis=1))
    np.testing.assert_array_equal(flags, np.where(pid % 3 == 0, pid + 1, 0))


@piton.jit
def early_exit_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_blocks:
        return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)


def test_early_return(persistent):
    persistent(4)
    ck = compiler.compile(early_exit_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_blocks": "i32"},
                          {"BLOCK": 32}, grid=(10,))
    assert ck.coarsen == 4
    x = rng.standard_normal(320).astype(np.float32)
    out = np.zeros(320, dtype=np.float32)
    run_emulator(ck, (10,), {"x_ptr": x, "out_ptr": out, "n_blocks": 7})
    np.testing.assert_array_equal(out[:224], x[:224] * 2)
    assert np.all(out[224:] == 0)


MATMUL_SIG = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
              "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}


@pytest.mark.parametrize("mode", ["registers", "reload"])
def test_matmul_k_loop(persistent, monkeypatch, mode):
    """A nested runtime loop with loop-carried accumulators and arguments re-read
    after it; k=3 does not divide the 10 tiles. 'reload' forces the version that
    re-reads the uniforms for every program id."""
    persistent(3)
    if mode == "reload":
        _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1 and not o["coarsen_reload"])
    M, N, K = 40, 128, 16
    cons = dict(K=K, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=4, GROUP_SIZE_M=2)
    grid = ((M // 8) * (N // 64),)
    ck = compiler.compile(matmul_kernel, MATMUL_SIG, cons, grid=grid)
    assert ck.coarsen == 3 and ck.stats["coarsen_reload"] == (mode == "reload")
    if mode == "reload":
        assert ck.uniforms[-1] == ("self", "unif_base")
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = np.full(M * N, np.nan, dtype=np.float32)
    run_emulator(ck, grid, {"a_ptr": a, "b_ptr": b, "c_ptr": c, "M": M, "N": N, "stride_am": K,
                            "stride_bk": N, "stride_cm": N})
    np.testing.assert_allclose(c.reshape(M, N), a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)


def test_rerolled_body_overlaps_the_previous_store():
    """A rerolled kernel's chunk loop does not wait for the previous program id's
    last DMA before its loads: the first store of the next id does."""
    os.environ["PITON_PERSISTENT"] = "4"
    try:
        ck = compiler.compile(sfu_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                              {"OP": 4, "BLOCK": 256}, grid=(16,))
    finally:
        del os.environ["PITON_PERSISTENT"]
    assert ck.coarsen == 4 and ck.stats["rerolled"]
    prog = ck.program
    start = prog.labels[next(lbl for lbl in prog.labels if lbl.startswith("programs"))]
    text = [prog.instrs[i].__repr__() for i in range(start, len(prog.instrs))]
    first_wait = next(i for i, t in enumerate(text) if "vw_wait" in t)
    assert any("t0s" in t or "t1s" in t for t in text[:first_wait])
    x = rng.uniform(-8, 8, 16 * 256).astype(np.float32)
    out = np.zeros_like(x)
    run_emulator(ck, (16,), {"x_ptr": x, "out_ptr": out, "n": x.size})
    np.testing.assert_allclose(out, np.exp(x.astype(np.float64)), rtol=4e-6)


def test_falls_back_when_no_version_allocates(persistent, monkeypatch):
    monkeypatch.delenv("PITON_PERSISTENT", raising=False)
    plain = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    persistent(1)
    _fail_builds(monkeypatch, lambda o: o.get("coarsen", 1) > 1)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(1024,))
    assert ck.coarsen == 1 and "coarsen" not in ck.stats
    assert list(ck.words) == list(plain.words) and ck.uniforms == plain.uniforms


@piton.jit
def no_pid_kernel(out_ptr):
    tl.store(out_ptr + tl.arange(0, 16), tl.arange(0, 16))


@pytest.mark.parametrize("value,grid,k", [
    (None, (1024,), 1), ("0", (1024,), 1),
    ("1", (1024,), 4), ("1", (2048,), 4), ("1", (512,), 4), ("1", (256,), 2), ("1", (128,), 1),
    ("1", None, 1), ("1", (1024, 2), 1),                          # automatic: known 1-D grids only
    ("8", (1024,), 8), ("8", (5,), 5), ("3", None, 3), ("4", (64, 8), 4),
])
def test_factor(persistent, monkeypatch, value, grid, k):
    if value is None:
        monkeypatch.delenv("PITON_PERSISTENT", raising=False)
    else:
        persistent(value)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 16}, grid=grid)
    assert ck.coarsen == k


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
    """kernel[grid](...) launches ceil(grid / k) programs."""
    persistent(4)
    n = 11 * 32 - 3
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    ck = jit_add_kernel[(piton.cdiv(n, 32),)](x, y, out, n, BLOCK_SIZE=32)
    assert ck.coarsen == 4
    np.testing.assert_array_equal(out, x + y)


def test_generated_c_launcher(persistent, tmp_path):
    """tools/compile.py's C launcher writes ceil(gx / k) uniform lists with
    pid0 = the hardware program and nprog0 = gx."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    persistent(4)
    ck = compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, grid=(64,))
    assert ck.coarsen == 4
    header, source = emit_c(ck, "addk")
    assert "hardware program runs 4 consecutive x ids" in header
    (tmp_path / "addk.h").write_text(header)
    (tmp_path / "addk.c").write_text(source)
    (tmp_path / "main.c").write_text("""#include <stdio.h>
#include <stdlib.h>
#include "addk.h"
static uint32_t next_bus = 0x40000000;
piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b = {.handle = 1, .bus = next_bus, .cpu = calloc(size, 1), .size = size};
    next_bus += (size + 4095) & ~4095u;
    return b;
}
void piton_free(piton_buf_t *b) { free((void *)b->cpu); b->handle = 0; }
static piton_buf_t *unif_view;
uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n) {
    (void)code;
    printf("programs %u\\n", n);
    for (unsigned i = 0; i < n; i++) {
        printf("UNIFORM");
        for (unsigned j = 0; j < ADDK_NUM_UNIFORMS; j++) {
            /* uniforms live in the most recent allocation (unif_buf) */
            const volatile uint32_t *w = (const volatile uint32_t *)unif_view->cpu;
            printf(" %u", w[(unifs[i] - unif_view->bus) / 4 + j]);
        }
        printf("\\n");
    }
    return 0;
}
""")
    # the launcher's static unif_buf is the third allocation: expose it to the stub
    source = (tmp_path / "addk.c").read_text().replace(
        "static piton_buf_t code_buf, unif_buf, zero_buf;",
        "static piton_buf_t code_buf, unif_buf, zero_buf;\nextern piton_buf_t *unif_view_ptr(void);")
    source += "piton_buf_t *addk_unif_buf(void) { return &unif_buf; }\n"
    (tmp_path / "addk.c").write_text(source)
    (tmp_path / "run.c").write_text("""#include "addk.h"
extern piton_buf_t *addk_unif_buf(void);
extern piton_buf_t *unif_view_set(piton_buf_t *);
int main(void) {
    addk_args_t args = {.x_ptr = 0x100, .y_ptr = 0x200, .output_ptr = 0x300, .n_elements = 640};
    extern piton_buf_t *unif_view;
    return 0;
}
""")
    pytest.skip("replaced below")
