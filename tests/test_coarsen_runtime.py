"""PITON_COARSEN_RUNTIME: the launch chooses a coarsened program's ids per hardware program.

compile() still decides which kernels are coarsened and the most ids per
hardware program (ck.coarsen, PITON_PERSISTENT, compiler._coarsening). With
the switch on, the program reads the launch's factor, 1 ... ck.coarsen, from
uniform coarsen0 (Lowering.begin_programs), and every launch path chooses it
from the launched grid as _coarsening chooses it from a known one
(runtime/launcher.py launch_coarsen): ck.coarsen for at least COARSEN_LAUNCH_MIN_GRID
programs, halved until COARSEN_LAUNCH_MIN_PROGRAMS hardware programs remain, and 1
below, so a small launch runs the plain program's ids, one hardware program
each. A forced factor (PITON_PERSISTENT=k) runs k ids in every launch. Every
run is strict-emulated and compared bit for bit with the plain build.
"""
import dataclasses
import importlib.util
import os
import shutil
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import piton
import piton.language as tl
from piton import compiler
from piton.qpu.emulator import V3D
from piton.qpu.isa import f2u
from piton.runtime import pi
from piton.runtime.launcher import (launch_coarsen, program_ids, run_emulator, semaphore_programs,
                                    slot_semaphores, uniforms_for)
from piton.tools.compile import emit_c

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from kernels import add_kernel, matmul_kernel  # noqa: E402
from nn_kernels import dense_layer_kernel, dense_tiles  # noqa: E402
from primitive_kernels import copy_kernel  # noqa: E402
from piton.backend.regalloc import RegAllocError  # noqa: E402
from test_persistent import (ADD_SIG, COPY_SIG, MATMUL_SIG, STUB_RUNTIME, id_slot_kernel, ids_kernel,  # noqa: E402
                             late_args_kernel)
from test_store_slots import contended  # noqa: E402
from test_store_slots_protocol import protocol_errors  # noqa: E402

rng = np.random.default_rng(31)
LAUNCH = (compiler.COARSEN_LAUNCH_MIN_GRID, compiler.COARSEN_LAUNCH_MIN_PROGRAMS)


@pytest.fixture
def env(monkeypatch):
    """env(persistent, runtime): PITON_PERSISTENT (None = the plain build "0",
    1 = automatic, k = forced) and PITON_COARSEN_RUNTIME "1" or unset."""
    monkeypatch.delenv("PITON_STORE_SLOTS", raising=False)

    def set_env(persistent=1, runtime=True):
        monkeypatch.setenv("PITON_PERSISTENT", "0" if persistent is None else str(persistent))
        if runtime:
            monkeypatch.setenv("PITON_COARSEN_RUNTIME", "1")
        else:
            monkeypatch.delenv("PITON_COARSEN_RUNTIME", raising=False)
    return set_env


def _run(ck, grid, values, coarsen=None, v3d=None):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    stats = run_emulator(ck, grid, values, v3d=v3d, coarsen=coarsen)
    return values, stats["instructions"]


def _same(want, got, what=""):
    for name, v in want.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(got[name].view(np.uint32), v.view(np.uint32), err_msg=f"{name} {what}")


def _add_values(gx, block):
    n = gx * block - 5                       # a masked tail in the last id
    return {"x_ptr": rng.standard_normal(n).astype(np.float32), "y_ptr": rng.standard_normal(n).astype(np.float32),
            "output_ptr": np.full(n + 16, 7.0, dtype=np.float32), "n_elements": n}


def _add(block, grid=(1024,)):
    return compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": block}, grid=grid)


def _only(real, reload):
    """compiler._compile_tile refusing the coarsened builds with the other uniform protocol."""
    def build(fn, signature, constants, name, pid_max, options, *variant, **kw):
        if options.get("coarsen", 1) > 1 and options["coarsen_reload"] != reload:
            raise RegAllocError("forced by the test")
        return real(fn, signature, constants, name, pid_max, options, *variant, **kw)
    return build


def _expected_factor(k, gx):
    """compiler._coarsening's rule for a grid of gx programs and at most k ids."""
    if gx < LAUNCH[0]:
        return 1
    while k > 1 and gx // k < LAUNCH[1]:
        k //= 2
    return k


# -- the switch and the factor ------------------------------------------------------------

def test_switch(env, monkeypatch):
    """Unset, "" or "0": the compile-time factor, no coarsen0 uniform; "1": the
    same factor read from coarsen0 right after nprog0. Builds that are not
    coarsened are the same code either way."""
    env(1, runtime=False)
    off = []
    for value in (None, "", "0"):
        if value is None:
            monkeypatch.delenv("PITON_COARSEN_RUNTIME", raising=False)
        else:
            monkeypatch.setenv("PITON_COARSEN_RUNTIME", value)
        ck = _add(64)
        assert ck.coarsen == 4 and ck.coarsen_launch is None and "coarsen_runtime" not in ck.stats
        assert ("coarsen", "coarsen0") not in ck.uniforms
        off.append((list(ck.words), ck.uniforms))
    assert off[0] == off[1] == off[2]
    plain = _add(64, (32,))
    env(1)
    on = _add(64)
    assert on.coarsen == 4 and on.coarsen_launch == LAUNCH and on.stats["coarsen_runtime"] == 1
    assert on.uniforms[on.uniforms.index(("nprog", "nprog0")) + 1] == ("coarsen", "coarsen0")
    assert "chosen per launch" in on.asm["qasm"]
    again = _add(64, (32,))
    assert again.coarsen == 1 and list(again.words) == list(plain.words) and again.uniforms == plain.uniforms


def test_factor_follows_the_launched_grid(env):
    """One build for 1024 programs: each launch gets the factor a build for
    exactly its grid would have had (k = 4 for >= 128 programs, 2 for 64-127,
    1 below)."""
    env(1)
    on = _add(64)
    grids = (1, 2, 13, 63, 64, 65, 127, 128, 255, 256, 300, 512, 1024)
    got = [launch_coarsen(on, (gx,)) for gx in grids]
    assert got == [_expected_factor(4, gx) for gx in grids] == [1] * 4 + [2] * 3 + [4] * 6
    for gx in (63, 64, 127, 128, 1024):
        assert _add(64, (gx,)).coarsen == launch_coarsen(on, (gx,)), gx
    # the launch decides, not the grid the build is for
    assert launch_coarsen(_add(64, (4096,)), (100,)) == 2
    # the switch off keeps the compile-time thresholds
    env(1, runtime=False)
    assert [_add(64, (gx,)).coarsen for gx in (128, 255, 256, 512)] == [1, 1, 2, 4]


def test_forced_factor_runs_in_every_launch(env):
    env(3)
    ck = _add(64)
    assert ck.coarsen == 3 and ck.coarsen_launch is None and ("coarsen", "coarsen0") in ck.uniforms
    assert [launch_coarsen(ck, (gx,)) for gx in (1, 5, 1024)] == [3, 3, 3]
    values = _add_values(7, 64)
    want, _ = _run(ck, (7,), values, coarsen=1)
    for k in (2, 3):
        _same(want, _run(ck, (7,), values, coarsen=k)[0], f"k={k}")
    with pytest.raises(ValueError, match="program ids per hardware program"):
        _run(ck, (7,), values, coarsen=4)
    env(3, runtime=False)
    with pytest.raises(ValueError, match="program ids per hardware program"):
        _run(_add(64), (7,), values, coarsen=1)            # (the factor is part of the code)


def test_factor_in_the_code_wins(env, fake_pi):
    """A build without coarsen0 runs its compile-time factor in every launch
    path even when coarsen_launch says otherwise (a hand-made CompiledKernel):
    launch_coarsen agrees with emit_c, which keys on the uniform."""
    env(1, runtime=False)
    ck = dataclasses.replace(_add(64), coarsen_launch=LAUNCH)
    assert ck.coarsen == 4 and ("coarsen", "coarsen0") not in ck.uniforms
    assert [launch_coarsen(ck, (gx,)) for gx in (1, 13, 300, 1024)] == [4, 4, 4, 4]
    _, source = emit_c(ck, "kern")
    assert "unsigned hx = (gx + 3) / 4;" in source and "unsigned k" not in source
    values = _add_values(13, 64)
    pi.run_on_pi(ck, (13,), values)
    assert len(fake_pi[-1][1]) == 4
    env(None)
    want, _ = _run(_add(64), (13,), values)
    _same(want, _run(ck, (13,), values)[0])


# -- one build, small and large launches ----------------------------------------------------

@pytest.mark.parametrize("block,grids", [(64, (1, 13, 63, 64, 100, 128, 301, 513)), (128, (5, 100, 257, 513))])
def test_small_and_large_launches(env, block, grids):
    """BLOCK 64: private VPM rows, 128: store slots. Each launch of the one
    build is bit-identical to the plain build: below 64 programs it runs one
    id per hardware program, a few loop instructions more than the plain
    program; at full k it executes at most two instructions per hardware
    program more than the build with k fixed at compile time."""
    env(None)
    plain = _add(block)
    env(1, runtime=False)
    fixed = _add(block)
    env(1)
    ck = _add(block)
    assert ck.coarsen == fixed.coarsen == 4 and bool(ck.stats.get("store_slots")) == (block == 128)
    assert block == 64 or protocol_errors(ck.program) == []
    for gx in grids:
        values = _add_values(gx, block)
        want, n_plain = _run(plain, (gx,), values)
        got, n = _run(ck, (gx,), values)
        _same(want, got, f"grid {gx}")
        k = launch_coarsen(ck, (gx,))
        assert k == _expected_factor(4, gx)
        if k == 1:
            assert n_plain < n <= n_plain + 24 * gx
        if k == 4:
            fixed_got, n_fixed = _run(fixed, (gx,), values)
            _same(want, fixed_got)
            assert n - n_fixed <= 2 * -(-gx // 4)
    values = _add_values(13, block)
    want, _ = _run(plain, (13,), values)
    for k in (1, 2, 3, 4):
        _same(want, _run(ck, (13,), values, coarsen=k)[0], f"k={k}")


@pytest.mark.parametrize("mode", ["registers", "reload"])
def test_matmul_k_loop(env, monkeypatch, mode):
    """A nested runtime loop with loop-carried accumulators and arguments re-read
    after it, with either uniform protocol: every factor 1 ... 3 of the one build
    (3 does not divide the 10 tiles), and a launch of 4 tiles."""
    reload = mode == "reload"
    M, N, K = 40, 128, 16
    cons = dict(K=K, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=4, GROUP_SIZE_M=2)
    grid = ((M // 8) * (N // 64),)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    values = {"a_ptr": a, "b_ptr": b, "c_ptr": np.full(M * N, np.nan, np.float32), "M": M, "N": N,
              "stride_am": K, "stride_bk": N, "stride_cm": N}
    env(None)
    want, _ = _run(compiler.compile(matmul_kernel, MATMUL_SIG, cons, grid=grid), grid, values)
    env(3)
    monkeypatch.setattr(compiler, "_compile_tile", _only(compiler._compile_tile, reload))
    ck = compiler.compile(matmul_kernel, MATMUL_SIG, cons, grid=grid)
    assert ck.coarsen == 3 and ck.stats["coarsen_reload"] == reload and ck.stats["coarsen_runtime"] == 1
    if reload:
        assert ck.uniforms[-4:] == [("pid", "pid0"), ("nprog", "nprog0"), ("coarsen", "coarsen0"),
                                    ("self", "unif_base")]
    for k in (1, 2, 3):
        _same(want, _run(ck, grid, values, coarsen=k)[0], f"k={k}")
    # 4 of the 10 tiles (rows 0-15): a launch smaller than the build's grid
    for k in (1, 3):
        got, _ = _run(ck, (4,), values, coarsen=k)
        np.testing.assert_array_equal(got["c_ptr"].reshape(M, N)[:16].view(np.uint32),
                                      want["c_ptr"].reshape(M, N)[:16].view(np.uint32))
        assert np.isnan(got["c_ptr"].reshape(M, N)[16:]).all()


@pytest.mark.parametrize("grid", [(10, 3), (7, 2), (2, 3)])
def test_two_dimensional_grid(env, grid):
    """Axis 0 coarsened (a forced factor) on 2-D grids, the launch smaller than
    the build's (10, 3): program_id / num_programs stay logical for every factor."""
    env(3)
    ck = compiler.compile(ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": 3}, grid=(10, 3))
    assert ck.coarsen == 3 and ck.coarsen_launch is None
    gx, gy = grid
    values = {"out_ptr": np.full(gx * gy * 16, -1, dtype=np.int32), "flag_ptr": np.zeros(gx * gy, dtype=np.int32)}
    pid, row = np.tile(np.arange(gx), gy), np.repeat(np.arange(gy), gx)
    expected = (3 * pid + 3) * 1000000 + gx * 10000 + gy * 100 + row
    for k in (1, 2, 3):
        got, _ = _run(ck, grid, values, coarsen=k)
        np.testing.assert_array_equal(got["out_ptr"].reshape(-1, 16), np.repeat(expected[:, None], 16, axis=1))
        np.testing.assert_array_equal(got["flag_ptr"], np.where(pid % 3 == 0, pid + 1, 0))


@pytest.mark.parametrize("reload", [False, True])
def test_arguments_reread_after_a_loop(env, monkeypatch, reload):
    """2-D, arguments re-read from the uniform list after a top-level loop, the
    second axis's id read again after it: every factor against the plain build."""
    grid = (7, 2)
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "side_ptr": "*i32", "n_iter": "i32",
           "scale": "fp32"}
    values = {"x_ptr": rng.standard_normal(48).astype(np.float32),
              "y_ptr": rng.standard_normal(14 * 16).astype(np.float32),
              "out_ptr": np.zeros(14 * 16, np.float32), "side_ptr": np.zeros(14, np.int32),
              "n_iter": 3, "scale": 0.75}
    env(None)
    want, _ = _run(compiler.compile(late_args_kernel, sig, {"BLOCK": 16}, grid=grid), grid, values)
    env(3)
    monkeypatch.setattr(compiler, "_compile_tile", _only(compiler._compile_tile, reload))
    ck = compiler.compile(late_args_kernel, sig, {"BLOCK": 16}, grid=grid)
    assert ck.coarsen == 3 and ck.stats["coarsen_reload"] == reload
    for k in (1, 2, 3):
        _same(want, _run(ck, grid, values, coarsen=k)[0], f"k={k}")


def test_store_slots_under_contention(env):
    """copy (BLOCK 256, store slots, k = 4 for a 1024-program build): small and
    large launches and every factor under the contended timing (one VDW engine,
    long jittered DMAs), the semaphores back at 0 after each launch."""
    env(None)
    plain = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,))
    env(1)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,))
    assert ck.coarsen == 4 and ck.coarsen_launch == LAUNCH and ck.stats["store_slots"] == 1
    assert protocol_errors(ck.program) == []
    for gx, ks in ((5, (None, 2, 4)), (9, (3,)), (100, (None,))):
        n = gx * 256 - 9
        values = {"x_ptr": rng.standard_normal(n).astype(np.float32), "out_ptr": np.zeros(n, np.float32), "n": n}
        want, _ = _run(plain, (gx,), values)
        for k in ks:
            v3d = contended(gx + (k or 0))
            got, _ = _run(ck, (gx,), values, coarsen=k, v3d=v3d)
            assert v3d.sems == [0] * 16
            _same(want, got, f"grid {gx} k={k}")
    assert launch_coarsen(ck, (5,)) == 1 and launch_coarsen(ck, (100,)) == 2


def test_digits_dense_layer(env):
    """examples/digits_pi.py's first layer, built once for the full 1800-row
    batch (900 programs, k = 2): a one-image launch (2 programs) runs one id per
    hardware program, the full batch two; both factors match the plain build."""
    k_in, n = 64, 128
    bm, bn, bk = dense_tiles(n)
    constants = dict(N=n, K=k_in, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                     ADD_BIAS=True, RELU=True)
    sig = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    sig["M"] = "i32"
    full = (1800 // bm * (n // bn),)
    env(None)
    plain = compiler.compile(dense_layer_kernel, sig, constants, grid=full)
    env(1)
    ck = compiler.compile(dense_layer_kernel, sig, constants, grid=full)
    assert ck.coarsen == 2 and ck.coarsen_launch == LAUNCH
    assert [launch_coarsen(ck, (g,)) for g in (2, 8, 32, 63, 64, 128, 900)] == [1] * 4 + [2] * 3
    w = rng.standard_normal((k_in, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    for rows in (4, 20):
        x = rng.standard_normal((rows, k_in)).astype(np.float32)
        values = {"x_ptr": x, "w_ptr": w, "bias_ptr": bias, "y_ptr": np.full(rows * n + 32, -3.0, np.float32),
                  "M": rows}
        grid = (-(-rows // bm) * (n // bn),)
        want, n_plain = _run(plain, grid, values)
        got, n_one = _run(ck, grid, values)
        _same(want, got, f"{rows} rows")
        assert n_one <= 1.02 * n_plain         # (the coarsened build's K loop is 2 instructions longer)
        _same(want, _run(ck, grid, values, coarsen=2)[0], f"{rows} rows, k=2")


# -- every id exactly once, the 24-bit multiplier, the uniform list ---------------------------

@piton.jit
def count_kernel(cnt_ptr, BLOCK: tl.constexpr):
    """Each id adds 1 to its own row: an id run twice leaves 2, a skipped one
    0, one past the grid writes into the guard rows."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(cnt_ptr + offs, tl.load(cnt_ptr + offs) + 1)


@pytest.mark.parametrize("persistent", [1, 4])
def test_every_id_runs_once(env, persistent):
    """Every factor 1 ... 4 (and the launch's own) of one build on grids below,
    at and above k, not multiples of it, and around the launch thresholds."""
    env(persistent)
    ck = compiler.compile(count_kernel, {"cnt_ptr": "*i32"}, {"BLOCK": 16}, grid=(1024,))
    assert ck.coarsen == 4 and ("coarsen", "coarsen0") in ck.uniforms
    for gx in (1, 2, 3, 4, 5, 7, 9, 63, 64, 65, 127, 128, 129, 255, 256, 1023):
        for k in (None, 3) if gx > 9 else (None, 1, 2, 3, 4):
            got, _ = _run(ck, (gx,), {"cnt_ptr": np.zeros((gx + 8) * 16, np.int32)}, coarsen=k)
            got = got["cnt_ptr"].reshape(-1, 16)
            assert np.all(got[:gx] == 1) and np.all(got[gx:] == 0), (gx, k)


@pytest.mark.parametrize("G", [1 << 24, (1 << 24) + 3])
def test_first_id_at_the_24_bit_boundary(env, G):
    """pid0 * coarsen0 with pid0 up to G - 1 (k = 1): a mul24 while the grid
    fits 24 bits, the exact product past it; hardware programs at the top of
    the grid and around 2^24 / k for k = 1 ... 4."""
    env(4)
    ck = compiler.compile(id_slot_kernel, {"out_ptr": "*i32"}, {"BLOCK": 16}, grid=(G,))
    assert ck.coarsen == 4 and ("coarsen", "coarsen0") in ck.uniforms
    for k in (1, 2, 3, 4):
        hx = -(-G // k)
        hws, expect = [], {}
        for hw in (0, (1 << 24) // k - 1, (1 << 24) // k, hx - 2, hx - 1):
            ids = range(hw * k, min(hw * k + k, G))
            if hw < hx and not any(p % 1021 in expect for p in ids):
                hws.append(hw)
                expect.update((p % 1021, p) for p in ids)
        v = V3D(num_qpus=4, strict=True)
        out = v.mem.alloc(1021 * 16 * 4, 64)
        v.mem.write_array(out, np.full(1021 * 16, -1, np.int32))
        zero, scratch = v.mem.alloc(64, 64), v.mem.alloc(64, 64)
        unifs = []
        for hw in hws:
            a = v.mem.alloc(4 * len(ck.uniforms), 16)
            v.mem.write_words(a, np.array(uniforms_for(ck, {"out_ptr": np.zeros(1, np.int32)}, {"out_ptr": out},
                                                       (hw, 0, 0), (G, 1, 1), zero, scratch, a, k), dtype=np.uint32))
            unifs.append(a)
        v.run(v.load(ck.words), unifs)
        got = v.mem.read_array(out, (1021, 16), np.int32)
        for slot, p in expect.items():
            assert np.all(got[slot] == p), (k, slot, p)
        assert sum(bool(np.all(r == -1)) for r in got) == 1021 - len(expect)


@pytest.mark.parametrize("reload", [False, True])
def test_uniform_reads_stay_in_the_list(env, monkeypatch, reload):
    """Every uniform a hardware program reads, with either protocol and every
    factor, is one of its own list's words (one past it would be the next
    program's first argument), and it reads all of them."""
    from piton.qpu import emulator
    monkeypatch.setattr(compiler, "_compile_tile", _only(compiler._compile_tile, reload))
    env(1)
    ck = _add(64)
    assert ck.coarsen == 4 and ck.stats["coarsen_reload"] == reload
    base, reads = {}, set()
    start, read = emulator.QPU.start, emulator.QPU.read_regfile

    def start_(self, program, entry, unif_addr):
        start(self, program, entry, unif_addr)
        base[self.num] = unif_addr

    def read_(self, src, file):
        if src.raddr == 32:                      # unif
            off = self.unif_ptr - base[self.num]
            assert 0 <= off < 4 * len(ck.uniforms), off
            reads.add(off)
        return read(self, src, file)
    monkeypatch.setattr(emulator.QPU, "start", start_)
    monkeypatch.setattr(emulator.QPU, "read_regfile", read_)
    for gx in (1, 5, 300):
        for k in (None, 1, 3, 4):
            reads.clear()
            _run(ck, (gx,), _add_values(gx, 64), coarsen=k)
            assert reads == set(range(0, 4 * len(ck.uniforms), 4)), (gx, k)


@piton.jit
def ids3_kernel(out_ptr):
    p0, p1, p2 = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n0, n1, n2 = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    flat = (p2 * n1 + p1) * n0 + p0
    offs = flat * 16 + tl.arange(0, 16)
    tl.store(out_ptr + offs, tl.load(out_ptr + offs) + p0 * 1000000 + p1 * 10000 + p2 * 100 + n0 * 7 + n1 * 3 + n2)


@pytest.mark.parametrize("grid", [(5, 2, 2), (1, 1, 3), (4, 3, 2)])
def test_three_dimensional_grid(env, grid):
    """Axis 0 coarsened (forced 3) on 3-D grids smaller than the build's: every
    factor runs each (x, y, z) once with the logical ids and grid."""
    env(3)
    ck = compiler.compile(ids3_kernel, {"out_ptr": "*i32"}, {}, grid=(9, 3, 3))
    assert ck.coarsen == 3
    gx, gy, gz = grid
    n = gx * gy * gz
    p0 = np.tile(np.arange(gx), gy * gz)
    p1 = np.tile(np.repeat(np.arange(gy), gx), gz)
    p2 = np.repeat(np.arange(gz), gx * gy)
    want = p0 * 1000000 + p1 * 10000 + p2 * 100 + gx * 7 + gy * 3 + gz
    for k in (1, 2, 3):
        got, _ = _run(ck, grid, {"out_ptr": np.zeros((n + 4) * 16, np.int32)}, coarsen=k)
        got = got["out_ptr"].reshape(-1, 16)
        np.testing.assert_array_equal(got[:n], np.repeat(want[:, None], 16, axis=1))
        assert np.all(got[n:] == 0)


# -- the generated C launcher ------------------------------------------------------------------

def _c_launch(tmp_path, ck, args, grids):
    """Build tools/compile.py's launcher against the stub runtime of
    tests/test_persistent.py; returns, per launch, the program count and each
    program's uniform words (its uniform address first)."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    header, source = emit_c(ck, "kern")
    (tmp_path / "kern.h").write_text(header)
    (tmp_path / "kern.c").write_text(source)
    fields = ", ".join(f".{n} = {v}" + ("f" if isinstance(v, float) else "") for n, v in args.items())
    calls = "\n".join(f"    kern_launch(&a, {gx}, {gy}, {gz});" for gx, gy, gz in grids)
    (tmp_path / "main.c").write_text(STUB_RUNTIME + f"int main(void) {{\n    kern_args_t a = {{{fields}}};\n"
                                     f"{calls}\n    return 0;\n}}\n")
    runtime = os.path.abspath(os.path.join(ROOT, "pi", "runtime"))
    subprocess.run([cc, "-std=c99", "-O1", "-Wall", "-Werror", "-I", runtime, "-I", str(tmp_path),
                    str(tmp_path / "main.c"), str(tmp_path / "kern.c"), "-o", str(tmp_path / "run")],
                   check=True, capture_output=True, text=True)
    log = subprocess.run([str(tmp_path / "run")], check=True, capture_output=True, text=True).stdout.splitlines()
    zero = int([ln for ln in log if ln.startswith("ALLOC")][1].split()[1])     # code, zero, uniforms
    launches = []
    for i, ln in enumerate(log):
        if ln.startswith("LAUNCH"):
            count = int(ln.split()[1])
            launches.append([[int(w) for w in p.split()[1:]] for p in log[i + 1:i + 1 + count]])
    assert len(launches) == len(grids)
    return header, source, zero, launches, log


@pytest.mark.parametrize("which", ["add", "copy", "ids", "dense"])
def test_generated_c_launcher(env, monkeypatch, tmp_path, which):
    """The C launcher chooses k at run time as launch_coarsen does (small and
    large gx, one build) and writes the same uniforms as uniforms_for, with
    both uniform protocols; copy goes through piton_launch_slots; ids_kernel's
    forced factor holds for every 2-D launch; dense is examples/digits_pi.py's
    first layer (batches of 1, 16, 64, 256 and 1800 rows)."""
    monkeypatch.setenv("PITON_SLOTS_ONCE", "0")     # the stub runtime logs init and fini
    reloads = (False, True)
    if which == "add":
        kernel, sig, cons, grid = add_kernel, ADD_SIG, {"BLOCK_SIZE": 64}, (1024,)
        args = {"x_ptr": 0x1000, "y_ptr": 0x2000, "output_ptr": 0x3000, "n_elements": 777 * 64}
        grids = [(1, 1, 1), (13, 1, 1), (63, 1, 1), (64, 1, 1), (100, 1, 1), (127, 1, 1), (128, 1, 1),
                 (777, 1, 1), (1024, 1, 1)]
        env(1)
    elif which == "copy":
        kernel, sig, cons, grid = copy_kernel, COPY_SIG, {"BLOCK": 256}, (1024,)
        args = {"x_ptr": 0x1000, "out_ptr": 0x9000, "n": 2000}
        grids = [(5, 1, 1), (300, 1, 1), (600, 1, 1)]
        env(1)
    elif which == "ids":
        kernel, sig, cons, grid = ids_kernel, {"out_ptr": "*i32", "flag_ptr": "*i32"}, {"ROUNDS": 2}, (10, 3)
        args = {"out_ptr": 0x1000, "flag_ptr": 0x2000}
        grids = [(10, 3, 1), (7, 2, 1), (1, 3, 1)]
        env(3)
    else:
        bm, bn, bk = dense_tiles(128)
        kernel, grid = dense_layer_kernel, (1800 // bm * (128 // bn),)
        sig = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
        sig["M"] = "i32"
        cons = dict(N=128, K=64, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                    ADD_BIAS=True, RELU=True)
        args = {"x_ptr": 0x1000, "w_ptr": 0x2000, "bias_ptr": 0x3000, "y_ptr": 0x4000, "M": 4}
        grids = [(2, 1, 1), (8, 1, 1), (32, 1, 1), (128, 1, 1), (900, 1, 1)]
        reloads = (False,)                  # (the build it chooses)
        env(1)
    values = {n: (f2u(v) if isinstance(v, float) else v) for n, v in args.items()}
    real = compiler._compile_tile
    for reload in reloads:
        monkeypatch.setattr(compiler, "_compile_tile", _only(real, reload))
        ck = compiler.compile(kernel, sig, cons, grid=grid)
        assert ck.stats["coarsen_reload"] == reload and ("coarsen", "coarsen0") in ck.uniforms
        d = tmp_path / str(int(reload))
        d.mkdir()
        header, source, zero, launches, log = _c_launch(d, ck, args, grids)
        assert ("runs up to" in header) == (which != "ids")
        assert (f"if (gx < {LAUNCH[0]}) k = 1;" in source) == (which != "ids")
        at = ck.uniforms.index(("coarsen", "coarsen0"))
        for g, programs in zip(grids, launches):
            k = launch_coarsen(ck, g)
            ids = list(program_ids(g, k))
            assert len(programs) == len(ids) == -(-g[0] // k) * g[1] * g[2]
            for (pid, gg), words in zip(ids, programs):
                assert words[1:] == uniforms_for(ck, values, {}, pid, gg, zero, zero + 2048, words[0])
                assert words[1 + at] == k
        sems = slot_semaphores(ck.words)
        assert bool(sems) == (which == "copy")
        if sems:
            init, _ = semaphore_programs(sems)
            assert [ln for ln in log if ln.startswith("SLOTS")] == \
                [f"SLOTS {8 * len(ck.words)} {8 * (len(ck.words) + len(init))}"] * len(grids)
    if which == "add":
        assert [launch_coarsen(ck, g) for g in grids] == [1, 1, 1, 2, 2, 2, 4, 4, 4]
    if which == "dense":
        assert [launch_coarsen(ck, g) for g in grids] == [1, 1, 1, 2, 2]


def test_generated_c_without_the_switch(env):
    """Switch off: the launcher keeps the compile-time factor and has no k."""
    env(1, runtime=False)
    _, source = emit_c(_add(64), "kern")
    assert "unsigned hx = (gx + 3) / 4;" in source and "unsigned k" not in source


COUNT_STUB = r"""
#include <stdio.h>
#include <stdlib.h>
#include "kern.h"
static piton_buf_t bufs[4096];
static unsigned nbufs;
static uint32_t next_bus = 0x40000000u;
piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b = {.handle = nbufs + 1, .bus = next_bus, .cpu = calloc(size, 1), .size = size};
    next_bus += (size + 4095u) & ~4095u;
    bufs[nbufs++] = b;
    return b;
}
void piton_free(piton_buf_t *b) { b->handle = 0; }
static const volatile uint32_t *at(uint32_t bus) {
    for (unsigned i = nbufs; i-- > 0;)
        if (bus >= bufs[i].bus && bus < bufs[i].bus + bufs[i].size)
            return (const volatile uint32_t *)bufs[i].cpu + (bus - bufs[i].bus) / 4;
    abort();
}
/* per launch: programs, coarsen0 of each (one value or -1), ids covered, bad first ids */
uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n) {
    unsigned k = n ? at(unifs[0])[AT_COARSEN] : 0, covered = 0, bad = 0;
    (void)code;
    for (unsigned i = 0; i < n; i++) {
        const volatile uint32_t *u = at(unifs[i]);
        unsigned lo = u[AT_PID] * u[AT_COARSEN], hi = lo + u[AT_COARSEN];
        if (u[AT_COARSEN] != k) k = ~0u;
        if (lo >= u[AT_NPROG] || u[AT_PID] != i) bad++;
        covered += (hi < u[AT_NPROG] ? hi : u[AT_NPROG]) - lo;
    }
    printf("%u %d %u %u\n", n, (int)k, covered, bad);
    return 0;
}
uint32_t piton_launch_slots(uint32_t code, const uint32_t *unifs, uint32_t n, uint32_t init, uint32_t fini) {
    (void)init; (void)fini;
    return piton_launch(code, unifs, n);
}
"""


def test_generated_c_factor_for_every_gx(env, tmp_path):
    """The C launcher's k and hardware program count against launch_coarsen
    and program_ids for every gx up to 1100 (and a few large ones): the
    automatic factor 4, a forced 3, and a factor of 6 chosen per launch
    (6 -> 3 -> 1, coarsen_launch set by hand); every id runs exactly once."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    env(1)
    auto = _add(64)
    env(3)
    forced = _add(64)
    env(6)
    six = dataclasses.replace(_add(64), coarsen_launch=(16, 4))
    assert (auto.coarsen, forced.coarsen, six.coarsen) == (4, 3, 6) and forced.coarsen_launch is None
    gxs = list(range(1101)) + [4095, 4096, 65537, 1 << 20]
    for name, ck in (("auto", auto), ("forced", forced), ("six", six)):
        d = tmp_path / name
        d.mkdir()
        header, source = emit_c(ck, "kern")
        (d / "kern.h").write_text(header)
        (d / "kern.c").write_text(source)
        calls = "\n".join(f"    kern_launch(&a, {gx}, 1, 1);" for gx in gxs)
        (d / "main.c").write_text(COUNT_STUB + f"int main(void) {{\n    kern_args_t a = {{0}};\n{calls}\n"
                                  "    return 0;\n}\n")
        at = [f"-DAT_{n.upper()}={ck.uniforms.index((n, n + '0'))}" for n in ("pid", "nprog", "coarsen")]
        built = subprocess.run([cc, "-std=c99", "-O0", "-Wall", "-Werror", *at, "-I",
                                os.path.join(ROOT, "pi", "runtime"), "-I", str(d), str(d / "main.c"),
                                str(d / "kern.c"), "-o", str(d / "run")], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        log = subprocess.run([str(d / "run")], check=True, capture_output=True, text=True).stdout.splitlines()
        assert len(log) == len(gxs)
        for gx, line in zip(gxs, log):
            k = launch_coarsen(ck, (gx,))
            programs = len(list(program_ids((gx,), k)))
            assert [int(w) for w in line.split()] == [programs, k if gx else 0, gx, 0], (name, gx)
    assert [launch_coarsen(six, (gx,)) for gx in (15, 16, 23, 24, 1100)] == [1, 3, 3, 6, 6]


def test_generated_c_does_not_divide_by_k(env):
    """The ARM1176 has no divide instruction and the bare-metal programs (libpi)
    link no __aeabi_uidiv: the launcher divides only by constants, at any
    optimization level (gcc -O0 / -Og kept `gx / k` as a library call)."""
    env(1)
    auto = _add(64)
    env(3)
    forced = _add(64)
    env(6)
    six = dataclasses.replace(_add(64), coarsen_launch=(16, 4))
    for ck in (auto, forced, six):
        header, source = emit_c(ck, "kern")
        code = re.sub(r"//.*", "", source)
        assert all(c.isdigit() for c in re.findall(r"/=?\s*(\S)", code)), ck.coarsen
        arm = shutil.which("arm-none-eabi-gcc")
        nm = shutil.which("arm-none-eabi-nm")
        if arm is None or nm is None:
            continue
        with tempfile.TemporaryDirectory() as d:
            for name, text in (("kern.h", header), ("kern.c", source)):
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)
            for opt in ("-O0", "-Og"):
                obj = os.path.join(d, "kern.o")
                subprocess.run([arm, opt, "-std=gnu99", "-Wall", "-Werror", "-ffreestanding", "-nostdlib",
                                "-mcpu=arm1176jzf-s", "-mhard-float", "-mfpu=vfp",
                                "-I", os.path.join(ROOT, "pi", "runtime"), "-I", d, "-c", os.path.join(d, "kern.c"),
                                "-o", obj], check=True, capture_output=True)
                undefined = subprocess.run([nm, "-u", obj], check=True, capture_output=True, text=True).stdout
                assert "div" not in undefined, (ck.coarsen, opt, undefined)


# -- the Python launch paths to the Pi (a fake runner: nothing reaches the board) ---------------

@pytest.fixture
def fake_pi(monkeypatch):
    """PiRunner records the programs a launch path queues instead of running
    them: its outputs are zeros (sampled as asked), so a path that checks them
    afterwards fails its check, after the launch."""
    seen = []

    def init(self, *a, **kw):
        self.verbose = False

    def run(self, code_words, buffers, programs, outputs, repeats=1, label="piton", sample=None,
            perf_events=None, setup_c="", semaphores=None):
        object.__new__(pi.PiRunner)._generate(code_words, buffers, programs, outputs, repeats, label,
                                              sample or {}, perf_events or [], setup_c,
                                              slot_semaphores(code_words) if semaphores is None else semaphores)
        seen.append((list(code_words), programs))
        sample = sample or {}
        res = {b.name: np.zeros(b.nbytes // 4, np.uint32)[::sample.get(b.name, 1)] for b in buffers
               if b.name in outputs}
        return pi.PiResult(res, [100] * repeats, "PITON_PREP b 1\n", [[1] * len(perf_events or [])] * repeats)

    def refuse(*a, **kw):
        raise AssertionError("the tests never look for the Pi")
    monkeypatch.setattr(pi.PiRunner, "__init__", init)
    monkeypatch.setattr(pi.PiRunner, "run", run)
    monkeypatch.setattr(pi, "find_tty", refuse)
    monkeypatch.setattr(pi, "find_bootloader", refuse)
    return seen


def _ids(ck, programs):
    """(pid0, nprog0, coarsen0) of every queued program."""
    at = [ck.uniforms.index(u) for u in (("pid", "pid0"), ("nprog", "nprog0"), ("coarsen", "coarsen0"))]
    return [tuple(p[i] for i in at) for p in programs]


def _want(ck, gx):
    k = launch_coarsen(ck, (gx,))
    return [(x, gx, k) for x in range(-(-gx // k))]


def test_run_on_pi(env, fake_pi):
    env(1)
    ck = _add(64)
    for gx in (13, 100, 600):
        values = _add_values(gx, 64)
        pi.run_on_pi(ck, (gx,), values)
        words, programs = fake_pi[-1]
        assert words == list(ck.words) and _ids(ck, programs) == _want(ck, gx)
    assert [launch_coarsen(ck, (gx,)) for gx in (13, 100, 600)] == [1, 2, 4]


def test_profile_run_case(env, fake_pi):
    """perf/profile_pi.py's launch (imported, with the fake runner)."""
    from perf.profile_pi import run_case
    from perf.suite import CASES_BY_NAME, compile_case
    env(1)
    case = CASES_BY_NAME["add"]
    ck, run = compile_case(case, "pi")
    assert ck.coarsen == 4 and ck.coarsen_launch == LAUNCH
    for gx in (1024, 100):
        run.grid = (gx,)
        result = run_case(case, 1, pi.PiRunner(), (ck, run))
        assert result["programs"] == len(_want(ck, gx)) and _ids(ck, fake_pi[-1][1]) == _want(ck, gx)


def test_bench_matmul_on_pi(env, fake_pi):
    """A build that reads its factor (forced 2), launched as one that chooses
    it with small thresholds: 1 for 1 tile, 2 for 16."""
    env(2)
    cons = dict(K=16, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=4, GROUP_SIZE_M=1)
    ck = compiler.compile(matmul_kernel, MATMUL_SIG, cons, grid=(16,))
    assert ck.coarsen == 2 and ck.coarsen_launch is None
    ck = dataclasses.replace(ck, coarsen_launch=(8, 2))
    for m, n in ((8, 64), (64, 128)):
        gx = (m // 8) * (n // 64)
        scal = {"M": m, "N": n, "stride_am": 16, "stride_bk": n, "stride_cm": n}
        pi.bench_matmul_on_pi(ck, m, n, 16, (gx,), scal, repeats=1, sample_stride=7)
        assert _ids(ck, fake_pi[-1][1]) == _want(ck, gx)
    assert [p[2] for p in _ids(ck, fake_pi[0][1])] == [1] and {p[2] for p in _ids(ck, fake_pi[1][1])} == {2}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_launch_paths(env, fake_pi, monkeypatch):
    """benchmarks/packed_matmul_pi.py and compute_peak_pi.py (imported, with the
    fake runner; both check the zero outputs after the launch and fail): each
    hardware program gets the forced factor as coarsen0."""
    built = []
    real = compiler.compile

    def compile_(*a, **kw):
        built.append(real(*a, **kw))
        return built[-1]
    monkeypatch.setattr(compiler, "compile", compile_)
    env(2)
    packed = _load("benchmarks/packed_matmul_pi.py", "packed_matmul_pi")
    with pytest.raises(AssertionError, match="sampled correctness"):
        packed.bench(64, bk=4, repeats=1)
    peak = _load("benchmarks/compute_peak_pi.py", "compute_peak_pi")
    with pytest.raises(AssertionError, match="verification failed"):
        peak.run(iterations=2, unroll=4, repeats=1)
    assert len(built) == len(fake_pi) == 2
    for ck, (words, programs), gx in zip(built, fake_pi, (8, 12)):
        assert ck.coarsen == 2 and words == list(ck.words) and _ids(ck, programs) == _want(ck, gx)
