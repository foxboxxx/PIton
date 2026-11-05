"""PITON_MM: the k loop of a tl.dot whose operands its scf.for pipelines
(backend/lower.py dot_mm, MMPlan, LoopStream._mm_row_emitter).

The same multiply-accumulates in the same order as the plain dot, so every
result here is compared bit for bit with the build without the switch (and
with numpy), in the strict emulator: ragged K (masked A and B), every A
layout (row-major, w-column blocks for w = 1, 2, 4, 8), one to four
16-column blocks, a dense A pointer part (rows `% M`), runtime and
constant K, and the kernels the switch leaves alone."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, ROOT)
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton import compiler  # noqa: E402
from piton.backend import lower  # noqa: E402
from piton.backend.regalloc import RegAllocError  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from kernels import matmul_kernel  # noqa: E402
from nn_kernels import dense_layer_kernel  # noqa: E402
from perf.suite import CASES_BY_NAME, compile_case, default_check  # noqa: E402


@piton.jit
def _loop_matmul(a_ptr, b_ptr, out_ptr, k,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(k, BK)):
        a = tl.load(a_ptr + m * k + i * BK + ak, mask=i * BK + ak < k, other=0.0)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n, mask=i * BK + bk < k, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * BN + n, acc)


@piton.jit
def _two_dots(a_ptr, b_ptr, out_ptr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(K // BK):
        a = tl.load(a_ptr + m * K + i * BK + ak)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * BN + n, acc)


@piton.jit
def _a_used_twice(a_ptr, b_ptr, out_ptr, sum_ptr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    s = tl.zeros((BM, BK), tl.float32)
    for i in range(K // BK):
        a = tl.load(a_ptr + m * K + i * BK + ak)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
        s += a
    tl.store(out_ptr + m * BN + n, acc)
    tl.store(sum_ptr + m * BK + ak, s)


@piton.jit
def _a_dense_by_iteration(a_ptr, b_ptr, out_ptr, kmod, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
    # A's columns `% kmod`: a dense pointer part that is lowered again in
    # every environment of the loop (prologue, body, next iteration)
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(K // BK):
        a = tl.load(a_ptr + m * K + (i * BK + ak) % kmod)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * BN + n, acc)


@piton.jit
def _a_mask_carried(a_ptr, b_ptr, out_ptr, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # A's mask carried by the loop: a compare in the prologue, a dense
    # predicate in the body
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    amask = (m >= 0) & (ak < k)
    for i in range(tl.cdiv(k, BK)):
        a = tl.load(a_ptr + m * k + i * BK + ak, mask=amask, other=0.0)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n, mask=i * BK + bk < k, other=0.0)
        acc = tl.dot(a, b, acc)
        amask = (m >= 0) & ((i + 1) * BK + ak < k)
    tl.store(out_ptr + m * BN + n, acc)


@piton.jit
def _a_rows_gathered(a_ptr, idx_ptr, b_ptr, out_ptr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, CARRIED: tl.constexpr):
    # A's rows through an index vector loaded before the loop; the pointer
    # carried (one dense part) or written in the body (lowered again there)
    rows = tl.load(idx_ptr + tl.arange(0, BM))
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    a_ptrs = a_ptr + rows[:, None] * K + ak
    for i in range(K // BK):
        if CARRIED:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptr + rows[:, None] * K + i * BK + ak)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
    tl.store(out_ptr + tl.arange(0, BM)[:, None] * BN + n, acc)


@piton.jit
def _two_dot_loops(a_ptr, b_ptr, out_ptr, M, n1, n2, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
    # both loops' A pointers share one dense part (rows `% M`, hoisted)
    rows = tl.arange(0, BM) % M
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(n1):
        a = tl.load(a_ptr + rows[:, None] * K + i * BK + ak)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
    for i in range(n2):
        a = tl.load(a_ptr + rows[:, None] * K + i * BK + ak)
        b = tl.load(b_ptr + (i * BK + bk) * BN + n)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + tl.arange(0, BM)[:, None] * BN + n, acc)


@piton.jit
def _b_stride_by_iteration(a_ptr, b_ptr, out_ptr, n_iter, NI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr, RUNTIME: tl.constexpr):
    # row k of iteration i's B tile is row k * (i + 1): the row stride depends
    # on the loop variable, the base pointer does not
    m = tl.arange(0, BM)[:, None]
    n = tl.arange(0, BN)[None, :]
    ak = tl.arange(0, BK)[None, :]
    bk = tl.arange(0, BK)[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    if RUNTIME:
        trips = n_iter
    else:
        trips = NI
    for i in range(trips):
        a = tl.load(a_ptr + m * (BK * NI) + i * BK + ak)
        b = tl.load(b_ptr + bk * ((i + 1) * BN) + n)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + m * BN + n, acc)


MATMUL_SIG = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
              "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}


def _build(monkeypatch, mm, kernel, signature, constants, grid):
    if mm is None:
        monkeypatch.delenv("PITON_MM", raising=False)
    else:
        monkeypatch.setenv("PITON_MM", mm)
    return compiler.compile(kernel, signature, constants, grid=grid)


def _run(ck, grid, values):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    run_emulator(ck, grid, values)
    return values


def _off_and_on(monkeypatch, mm, kernel, signature, constants, grid, values, outputs):
    """Outputs of the build without PITON_MM and with PITON_MM=mm, bit for bit
    the same; returns (outputs, compiled kernel with the switch on)."""
    off = _run(_build(monkeypatch, None, kernel, signature, constants, grid), grid, values)
    ck = _build(monkeypatch, mm, kernel, signature, constants, grid)
    on = _run(ck, grid, values)
    for name in outputs:
        np.testing.assert_array_equal(on[name].view(np.uint32), off[name].view(np.uint32))
    return on, ck


def test_switch_values(monkeypatch):
    monkeypatch.delenv("PITON_MM", raising=False)
    assert compiler._mm_options() == {}
    for off in ("0", ""):
        monkeypatch.setenv("PITON_MM", off)
        assert compiler._mm_options() == {}
    monkeypatch.setenv("PITON_MM", "1")
    assert compiler._mm_options() == {"mm": frozenset(compiler.MM_PARTS)}
    monkeypatch.setenv("PITON_MM", "order,rp")
    assert compiler._mm_options() == {"mm": frozenset({"order", "rp"})}
    for bad in ("at", "rp", "order,deep", "2"):
        monkeypatch.setenv("PITON_MM", bad)
        with pytest.raises(ValueError):
            compiler._mm_options()


@pytest.mark.parametrize("bm,bn,bk,w", [(8, 64, 4, 2), (4, 64, 8, 4), (8, 32, 8, 2), (4, 32, 16, 4),
                                        (4, 64, 16, 4), (8, 16, 16, 2), (2, 16, 16, 8), (16, 16, 4, 1),
                                        (16, 16, 8, 1), (16, 32, 4, 1), (2, 16, 8, 0), (1, 32, 16, 0)])
@pytest.mark.parametrize("constant_k", [False, True])
def test_ragged_loop_matmul(monkeypatch, bm, bn, bk, w, constant_k):
    """Masked A and B (K = 2 BK + 5) in every A layout; w = 0: one A register, row-major."""
    k = 2 * bk + 5
    rng = np.random.default_rng(140 + bm + bn + bk)
    a = rng.standard_normal((bm, k)).astype(np.float32)
    b = rng.standard_normal((k, bn)).astype(np.float32)
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"}
    cons = dict(BM=bm, BN=bn, BK=bk)
    if constant_k:
        cons["k"] = k
    else:
        sig["k"] = "i32"
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), k=k)
    out, ck = _off_and_on(monkeypatch, "1", _loop_matmul, sig, cons, (1,), values, ["out_ptr"])
    np.testing.assert_allclose(out["out_ptr"], a @ b, rtol=1e-4, atol=1e-4)
    assert ck.stats["mm_dots"] == 1
    assert ck.stats.get("mm_a_blocks", 0) == int(w > 0)


@pytest.mark.parametrize("bm,bn,bk", [(8, 64, 8), (8, 64, 4), (16, 16, 8), (4, 64, 8)])
@pytest.mark.parametrize("mm", ["order", "order,at", "order,rp", "1"])
def test_unmasked_loop_matmul_parts(monkeypatch, bm, bn, bk, mm):
    """Aligned K: unmasked loads, so B rows come from the accumulator row
    pointer (rp); each part on its own and all of them."""
    k = 4 * bk
    rng = np.random.default_rng(150 + bm + bk)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.zeros((bm, bn), np.float32), k=k)
    out, ck = _off_and_on(monkeypatch, mm, _loop_matmul, {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"},
                          dict(BM=bm, BN=bn, BK=bk, k=k), (1,), values, ["out_ptr"])
    np.testing.assert_array_equal(out["out_ptr"], a @ b)
    blocks = "at" in mm or mm == "1"
    # more than two row-major A registers: only the column blocks take the MM path
    assert ck.stats.get("mm_dots", 0) == int(blocks or bm * bk <= 32)
    assert ck.stats.get("mm_a_blocks", 0) == int(blocks)


@pytest.mark.parametrize("bm,bn,bk,constant_k", [(8, 64, 8, False), (16, 32, 8, False), (16, 32, 8, True)])
def test_fits_where_the_plain_dot_does_not(monkeypatch, bm, bn, bk, constant_k):
    """Two accumulators hold A's column blocks, where the plain dot keeps
    every row-major A register live (and runs out of registers)."""
    k = 2 * bk + 5
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"}
    cons = dict(BM=bm, BN=bn, BK=bk)
    if constant_k:
        cons["k"] = k
    else:
        sig["k"] = "i32"
    with pytest.raises(RegAllocError):
        _build(monkeypatch, None, _loop_matmul, sig, cons, (1,))
    ck = _build(monkeypatch, "1", _loop_matmul, sig, cons, (1,))
    assert ck.stats["mm_a_blocks"] == 1
    rng = np.random.default_rng(170 + bm + bk)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    out = _run(ck, (1,), dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), k=k))
    np.testing.assert_array_equal(out["out_ptr"], a @ b)


@pytest.mark.parametrize("name", ["matmul_8x64x8", "matmul_8x64x4", "dense_relu"])
def test_suite_dot_cases(monkeypatch, name):
    """The suite's dot kernels (rows `% M`: a dense A pointer part permuted
    before the loop; dense_relu: masked rows, constant strides) bit for bit."""
    case = CASES_BY_NAME[name]
    monkeypatch.delenv("PITON_MM", raising=False)
    ck_off, run = compile_case(case, "emu")
    monkeypatch.setenv("PITON_MM", "1")
    ck_on, _ = compile_case(case, "emu", run)
    assert ck_on.stats["mm_dots"] == 1 and ck_on.stats["mm_a_blocks"] == 1
    assert len(ck_on.program.instrs) < len(ck_off.program.instrs)
    off, on = _run(ck_off, run.grid, run.inputs.values), _run(ck_on, run.grid, run.inputs.values)
    for out, expected in run.expected.items():
        np.testing.assert_array_equal(on[out].view(np.uint32), off[out].view(np.uint32))
        assert default_check(run, out, on[out], expected) is None


def _loop_bounds(ck):
    labels = ck.program.labels
    start = next(pos for label, pos in labels.items()
                 if label.startswith("loop") and not label.startswith(("loop_end", "loop_exit", "loop_pre")))
    end = next(pos for label, pos in labels.items() if label.startswith("loop_end"))
    return start, end


def test_bk8_matmul_loop_requests_share_rotations(monkeypatch):
    """matmul 8x64x8 (512^3 build): the k loop gets shorter than the plain
    one's 388 budget, and most B requests share an instruction with a
    rotated broadcast (their lane registers in regfile A, the row pointer
    in an accumulator)."""
    cons = dict(K=512, stride_ak=1, stride_bn=1, stride_cn=1,
                BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8, GROUP_SIZE_M=4)
    ck = _build(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, (512,))
    start, end = _loop_bounds(ck)
    assert end - start <= 360
    body = ck.program.instrs[start:end]
    shared = sum(1 for ins in body if ins.kind == "alu" and ins.mul is not None and ins.mul.rot is not None
                 and ins.add is not None and ins.add.dst in isa.TMU_S)
    assert shared >= 16
    assert ck.stats["mm_dots"] == 1 and not ck.stats.get("mm_fallback")


@pytest.mark.parametrize("bm,bn,bk", [(8, 32, 4), (4, 16, 2), (8, 64, 2)])
@pytest.mark.parametrize("mm", ["order,rp", "1"])
@pytest.mark.parametrize("runtime", [False, True])
def test_b_row_stride_by_iteration(monkeypatch, bm, bn, bk, mm, runtime):
    """B's row pointers (rp) are cached per base register and row stride: the
    prologue's base + k * stride of iteration 0 was reused in every later
    iteration when only the stride depends on the loop variable (wrong B
    rows from the second iteration on)."""
    ni = 3
    rng = np.random.default_rng(210 + bm + bk)
    a = rng.integers(-3, 4, (bm, bk * ni)).astype(np.float32)
    b = rng.integers(-3, 4, (bk * ni, bn)).astype(np.float32)
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n_iter": "i32"}
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), n_iter=ni)
    out, ck = _off_and_on(monkeypatch, mm, _b_stride_by_iteration, sig,
                          dict(NI=ni, BM=bm, BN=bn, BK=bk, RUNTIME=runtime), (1,), values, ["out_ptr"])
    np.testing.assert_array_equal(out["out_ptr"],
                                  sum(a[:, i * bk:(i + 1) * bk] @ b[np.arange(bk) * (i + 1)] for i in range(ni)))
    assert ck.stats["mm_dots"] == 1


def test_build_does_not_depend_on_object_addresses(monkeypatch):
    """The permutation of A's dense pointer part (rows `% M`) in the loop's
    preheader orders its moves by register creation, not by id(): the Pi
    runs the build the emulator checked.  (Sorted by address, the matmul_pi
    512^3 build varied by several instructions from run to run.)"""
    import builtins
    cons = dict(K=512, stride_ak=1, stride_bn=1, stride_cn=1,
                BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8, GROUP_SIZE_M=4)
    words = []
    for order in (1, -1):
        monkeypatch.setattr(lower, "id", lambda o, order=order: order * builtins.id(o), raising=False)
        ck = _build(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, (512,))
        assert ck.stats["mm_a_blocks"] == 1
        words.append(list(ck.words))
    assert words[0] == words[1]


@pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")
@pytest.mark.parametrize("name", ["matmul_8x64x8", "dense_relu"])
def test_qasm_assembles_to_our_bits(monkeypatch, name):
    """TMU requests next to rotations (small-immediate signal): vc4asm -V
    assembles the .qasm to exactly our words, without warnings."""
    monkeypatch.setenv("PITON_MM", "1")
    ck, _ = compile_case(CASES_BY_NAME[name], "pi")
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert list(words) == list(ck.words)
    assert not warnings


def test_runtime_k_stride_keeps_row_major_or_plain(monkeypatch):
    """A runtime stride along k (stride_ak) has no constant column-block
    view: the 8x8 A tile (four registers) gets the plain dot, a 8x4 tile (two)
    the MM dot with row-major A."""
    sig = dict(MATMUL_SIG, stride_ak="i32")
    n, kk = 64, 32
    rng = np.random.default_rng(160)
    a = rng.integers(-3, 4, (n, kk)).astype(np.float32)
    b = rng.integers(-3, 4, (kk, n)).astype(np.float32)
    for bk, mm_dots in ((8, 0), (4, 1)):
        cons = dict(K=kk, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=bk, GROUP_SIZE_M=4)
        values = dict(a_ptr=a, b_ptr=b, c_ptr=np.zeros((n, n), np.float32), M=n, N=n, stride_am=kk,
                      stride_ak=1, stride_bk=n, stride_cm=n)
        grid = ((n // 8) * (n // 64),)
        out, ck = _off_and_on(monkeypatch, "1", matmul_kernel, sig, cons, grid, values, ["c_ptr"])
        np.testing.assert_array_equal(out["c_ptr"], a @ b)
        assert ck.stats.get("mm_dots", 0) == mm_dots
        assert not ck.stats.get("mm_a_blocks")


def test_ragged_matmul_kernel_grid(monkeypatch):
    """The tutorial kernel on a ragged problem (M, N not tile multiples,
    rows and columns wrapping `% M`, `% N`) with several programs."""
    m, n, kk = 21, 80, 24
    rng = np.random.default_rng(161)
    a = rng.integers(-3, 4, (m, kk)).astype(np.float32)
    b = rng.integers(-3, 4, (kk, n)).astype(np.float32)
    cons = dict(K=kk, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=8, GROUP_SIZE_M=2)
    values = dict(a_ptr=a, b_ptr=b, c_ptr=np.full((m, n), np.nan, np.float32), M=m, N=n, stride_am=kk,
                  stride_bk=n, stride_cm=n)
    grid = (((m + 7) // 8) * ((n + 63) // 64),)
    out, ck = _off_and_on(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, grid, values, ["c_ptr"])
    np.testing.assert_array_equal(out["c_ptr"], a @ b)
    assert ck.stats["mm_a_blocks"] == 1


@pytest.mark.parametrize("bk", [4, 8])
def test_coarsened_matmul(monkeypatch, bk):
    """PITON_PERSISTENT coarsening of the tutorial matmul (two ids per
    hardware program, the k loop and its prologue once per id).  8x64x4:
    both builds coarsened, bit for bit.  8x64x8 at K = 128 (as matmul_pi's
    512^3 and 1024^3 builds): no coarsened plain build allocates, so
    compile() keeps one id per program, and so does the MM build, whose
    coarsened version can allocate (compiler._build: the switch does not
    change the ids per hardware program)."""
    n, kk = 64, 128
    rng = np.random.default_rng(165)
    a = rng.integers(-3, 4, (n, kk)).astype(np.float32)
    b = rng.integers(-3, 4, (kk, n)).astype(np.float32)
    cons = dict(K=kk, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=bk, GROUP_SIZE_M=4)
    values = dict(a_ptr=a, b_ptr=b, c_ptr=np.full((n, n), np.nan, np.float32), M=n, N=n, stride_am=kk,
                  stride_bk=n, stride_cm=n)
    grid = ((n // 8) * (n // 64),)
    monkeypatch.setenv("PITON_PERSISTENT", "0")
    plain = _run(_build(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, grid), grid, values)
    monkeypatch.setenv("PITON_PERSISTENT", "2")
    if bk == 4:
        out, ck = _off_and_on(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, grid, values, ["c_ptr"])
        assert ck.coarsen == 2
    else:
        out, ck = _off_and_on(monkeypatch, "1", matmul_kernel, MATMUL_SIG, cons, grid, values, ["c_ptr"])
        assert ck.coarsen == 1
        assert _build(monkeypatch, None, matmul_kernel, MATMUL_SIG, cons, grid).coarsen == 1
    assert ck.stats["mm_dots"] == 1
    np.testing.assert_array_equal(out["c_ptr"].view(np.uint32), plain["c_ptr"].view(np.uint32))
    np.testing.assert_array_equal(out["c_ptr"], a @ b)


@pytest.mark.parametrize("batch", [1, 5, 32])
def test_dense_layers(monkeypatch, batch):
    """The digits dense layers' tiles (examples/nn_kernels.py dense_tiles),
    with the batch rows masked (`rows < M`), bias and ReLU."""
    from nn_kernels import dense_tiles
    rng = np.random.default_rng(162 + batch)
    for k, n, relu in ((64, 128, True), (128, 64, True), (64, 16, False)):
        bm, bn, bk = dense_tiles(n)
        x = rng.random((batch, k), np.float32)
        wt = (rng.random((k, n), np.float32) - 0.5).astype(np.float32)
        bias = (rng.random(n, np.float32) - 0.5).astype(np.float32)
        padded = (batch + bm - 1) // bm * bm
        grid = ((padded // bm) * ((n + bn - 1) // bn),)
        sig = {"x_ptr": "*fp32", "w_ptr": "*fp32", "bias_ptr": "*fp32", "y_ptr": "*fp32", "M": "i32"}
        cons = dict(N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                    ADD_BIAS=True, RELU=relu)
        values = dict(x_ptr=x, w_ptr=wt, bias_ptr=bias, y_ptr=np.full((batch, n), np.nan, np.float32), M=batch)
        out, ck = _off_and_on(monkeypatch, "1", dense_layer_kernel, sig, cons, grid, values, ["y_ptr"])
        ref = x.astype(np.float64) @ wt + bias
        np.testing.assert_allclose(out["y_ptr"], np.maximum(ref, 0) if relu else ref, rtol=1e-4, atol=1e-4)
        assert ck.stats.get("mm_dots", 0) == int(bn % 16 == 0)


@pytest.mark.parametrize("bm,bn", [(16, 32), (8, 64)])
def test_dense_layer_wider_tiles(monkeypatch, bm, bn):
    """Tiles with more A rows per B row than the digits layers' 4x64x4, which
    allocate only with the MM dot (bias and ReLU live through the epilogue),
    give the 4x64x4 plain build's bits: every output sums its k in order."""
    from nn_kernels import dense_tiles
    batch, k, n = 21, 64, 128
    rng = np.random.default_rng(166 + bm)
    x = rng.random((batch, k), np.float32)
    wt = (rng.random((k, n), np.float32) - 0.5).astype(np.float32)
    bias = (rng.random(n, np.float32) - 0.5).astype(np.float32)
    sig = {"x_ptr": "*fp32", "w_ptr": "*fp32", "bias_ptr": "*fp32", "y_ptr": "*fp32", "M": "i32"}
    values = dict(x_ptr=x, w_ptr=wt, bias_ptr=bias, y_ptr=np.full((batch, n), np.nan, np.float32), M=batch)

    def build(mm, tile):
        cons = dict(N=n, K=k, BLOCK_SIZE_M=tile[0], BLOCK_SIZE_N=tile[1], BLOCK_SIZE_K=tile[2], GROUP_SIZE_M=1,
                    ADD_BIAS=True, RELU=True)
        grid = (((batch + tile[0] - 1) // tile[0]) * (n // tile[1]),)
        return _build(monkeypatch, mm, dense_layer_kernel, sig, cons, grid), grid

    plain, grid = build(None, dense_tiles(n))
    ref = _run(plain, grid, values)["y_ptr"]
    with pytest.raises(RegAllocError):
        build(None, (bm, bn, 4))
    ck, grid = build("1", (bm, bn, 4))
    assert ck.stats["mm_a_blocks"] == 1
    np.testing.assert_array_equal(_run(ck, grid, values)["y_ptr"].view(np.uint32), ref.view(np.uint32))
    np.testing.assert_allclose(ref, np.maximum(x.astype(np.float64) @ wt + bias, 0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kernel,extra", [(_two_dots, {}), (_a_used_twice, {"sum_ptr": "*fp32"})])
def test_kernels_left_alone(monkeypatch, kernel, extra):
    """Two dots in one loop, or A used by more than the dot: no MM dot."""
    bm, bn, bk, k = 8, 32, 8, 24
    rng = np.random.default_rng(163)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    sig = dict({"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"}, **extra)
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.zeros((bm, bn), np.float32))
    if extra:
        values["sum_ptr"] = np.zeros((bm, bk), np.float32)
    out, ck = _off_and_on(monkeypatch, "1", kernel, sig, dict(K=k, BM=bm, BN=bn, BK=bk), (1,), values,
                          list(values)[2:])
    assert not ck.stats.get("mm_dots")
    np.testing.assert_array_equal(out["out_ptr"], (2 if kernel is _two_dots else 1) * (a @ b))


def test_no_pipelining_no_mm(monkeypatch):
    """num_stages=1: no LoopStream, so the plain dot."""
    monkeypatch.setenv("PITON_MM", "1")
    ck = compiler.compile(_loop_matmul, {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"},
                          dict(BM=8, BN=64, BK=4, k=16), num_stages=1, grid=(1,))
    assert not ck.stats.get("mm_dots")


def test_registers_that_do_not_fit_fall_back(monkeypatch):
    """A build of the MM dot that runs out of registers is redone without it."""
    def refuse(self, *args, **kwargs):
        raise RegAllocError("test: no registers")
    monkeypatch.setattr(lower.Lowering, "dot_mm", refuse)
    bm, bn, bk, k = 8, 64, 8, 32
    rng = np.random.default_rng(164)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.zeros((bm, bn), np.float32))
    ck = _build(monkeypatch, "1", _loop_matmul, {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"},
                dict(BM=bm, BN=bn, BK=bk, k=k), (1,))
    assert ck.stats["mm_fallback"] == 1 and not ck.stats.get("mm_dots")
    np.testing.assert_array_equal(_run(ck, (1,), values)["out_ptr"], a @ b)


def test_mm_builds_win_over_plain_variants(monkeypatch):
    """PITON_ISSUE's pre-scheduling variants (compiler._compile_tile) whose MM
    dot does not allocate fall back to the plain dot, and those can be
    shorter than every MM build (_b_stride_by_iteration at 8x32x4 with a
    run-time trip count: its flag groups' variants ran out of accumulators,
    and their plain builds won).  An MM build still wins
    (compiler._build_cost), so the switch changes the k loop."""
    real = compiler._build

    def build(tile, ranges, options, interleave, alphas=None, flags=None):
        if options.get("mm_tried") and interleave and flags is None:
            raise RegAllocError("test: no accumulators for the flag groups")
        return real(tile, ranges, options, interleave, alphas, flags)
    monkeypatch.setattr(compiler, "_build", build)
    bm, bn, bk, ni = 8, 32, 4, 3
    rng = np.random.default_rng(220)
    a = rng.integers(-3, 4, (bm, bk * ni)).astype(np.float32)
    b = rng.integers(-3, 4, (bk * ni, bn)).astype(np.float32)
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n_iter": "i32"}
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), n_iter=ni)
    out, ck = _off_and_on(monkeypatch, "1", _b_stride_by_iteration, sig,
                          dict(NI=ni, BM=bm, BN=bn, BK=bk, RUNTIME=True), (1,), values, ["out_ptr"])
    assert ck.stats["mm_dots"] == 1 and not ck.stats.get("mm_fallback")
    np.testing.assert_array_equal(out["out_ptr"],
                                  sum(a[:, i * bk:(i + 1) * bk] @ b[np.arange(bk) * (i + 1)] for i in range(ni)))


def test_plan_order_and_spacing():
    """Every operand register exactly once, B row k + 1 during step k, A's
    column-block register r during step w r - 1, nothing in the last step."""
    plan = lower.MMPlan(None, "A", "B", 8, 8, 4, 2)
    assert plan.top == [("a", 0)] + [("b", cb) for cb in range(4)]
    assert sorted(plan.order) == sorted([("a", r) for r in range(4)] + [("b", j) for j in range(32)])
    assert plan.steps[7] == []
    for k in range(7):
        evs = [ev for _, ev in plan.steps[k]]
        assert evs[-4:] == [("b", (k + 1) * 4 + cb) for cb in range(4)]
        assert evs[:-4] == ([("a", (k + 1) // 2)] if k % 2 else [])
        rows = [m for m, _ in plan.steps[k]]
        assert rows == sorted(rows) and len(set(rows)) == len(rows) and rows[-1] < 8
    row = lower.MMPlan(None, "A", "B", 4, 8, 4, 0)
    assert row.top == [("a", 0), ("a", 1)] + [("b", cb) for cb in range(4)]
    assert row.entries()[:3] == [("A", 0), ("A", 1), ("B", 0)]


@pytest.mark.parametrize("kernel,bm,bn,bk", [(_a_dense_by_iteration, 8, 32, 8), (_a_dense_by_iteration, 16, 16, 4),
                                             (_a_mask_carried, 8, 32, 8), (_a_mask_carried, 16, 16, 4)])
def test_a_view_that_changes_in_the_loop_falls_back(monkeypatch, kernel, bm, bn, bk):
    """The column-block view of A is checked in the prologue; a body (or
    next-iteration) pointer whose dense part is lowered again, or a carried
    mask that is dense there, has no free view: the build falls back to the
    plain dot (it raised KeyError / AttributeError) with the same bits."""
    if kernel is _a_dense_by_iteration:
        k = 4 * bk
        sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "kmod": "i32"}
        cons = dict(K=k, BM=bm, BN=bn, BK=bk)
        extra = dict(kmod=k)
    else:
        k = 2 * bk + 3
        sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "k": "i32"}
        cons = dict(BM=bm, BN=bn, BK=bk)
        extra = dict(k=k)
    rng = np.random.default_rng(180 + bm + bk)
    a = rng.integers(-3, 4, (bm, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), **extra)
    out, ck = _off_and_on(monkeypatch, "1", kernel, sig, cons, (1,), values, ["out_ptr"])
    np.testing.assert_array_equal(out["out_ptr"], a @ b)
    assert ck.stats["mm_fallback"] == 1 and not ck.stats.get("mm_dots")


@pytest.mark.parametrize("bm,bn,bk", [(8, 32, 8), (16, 16, 4)])
@pytest.mark.parametrize("n1,n2", [(0, 2), (1, 2)])
def test_two_loops_share_a_dense_part(monkeypatch, bm, bn, bk, n1, n2):
    """Each loop permutes A's dense pointer part in its own preheader: the
    second loop used the first one's, which a zero-trip first loop never
    computed (wrong rows in the strict emulator, wild addresses on the Pi)."""
    k, mrows = 2 * bk, bm - 3
    rng = np.random.default_rng(190 + bm + n1)
    a = rng.integers(-3, 4, (mrows, k)).astype(np.float32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    rows = np.arange(bm) % mrows
    ref = sum(a[rows, i * bk:(i + 1) * bk] @ b[i * bk:(i + 1) * bk] for i in range(n1)) \
        + sum(a[rows, i * bk:(i + 1) * bk] @ b[i * bk:(i + 1) * bk] for i in range(n2))
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "n1": "i32", "n2": "i32"}
    values = dict(a_ptr=a, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32), M=mrows, n1=n1, n2=n2)
    out, ck = _off_and_on(monkeypatch, "1", _two_dot_loops, sig, dict(K=k, BM=bm, BN=bn, BK=bk), (1,), values,
                          ["out_ptr"])
    np.testing.assert_array_equal(out["out_ptr"], ref)
    assert ck.stats["mm_dots"] == 2 and ck.stats["mm_a_blocks"] == 2


@pytest.mark.parametrize("bm,bn,bk", [(8, 32, 8), (4, 64, 8)])
@pytest.mark.parametrize("carried", [True, False])
def test_gathered_a_rows(monkeypatch, bm, bn, bk, carried):
    """A's rows through loaded indices: a carried pointer keeps its dense
    part (column blocks), a pointer written in the body gets a new one
    there and falls back to the plain dot (it raised KeyError)."""
    k, nrows = 3 * bk, 11
    rng = np.random.default_rng(200 + bm + carried)
    a = rng.integers(-3, 4, (nrows, k)).astype(np.float32)
    idx = rng.integers(0, nrows, bm).astype(np.int32)
    b = rng.integers(-3, 4, (k, bn)).astype(np.float32)
    sig = {"a_ptr": "*fp32", "idx_ptr": "*i32", "b_ptr": "*fp32", "out_ptr": "*fp32"}
    values = dict(a_ptr=a, idx_ptr=idx, b_ptr=b, out_ptr=np.full((bm, bn), np.nan, np.float32))
    out, ck = _off_and_on(monkeypatch, "1", _a_rows_gathered, sig, dict(K=k, BM=bm, BN=bn, BK=bk, CARRIED=carried),
                          (1,), values, ["out_ptr"])
    np.testing.assert_array_equal(out["out_ptr"], a[idx] @ b)
    if carried:
        assert ck.stats["mm_a_blocks"] == 1 and not ck.stats.get("mm_fallback")
    else:
        assert ck.stats["mm_fallback"] == 1 and not ck.stats.get("mm_dots")
