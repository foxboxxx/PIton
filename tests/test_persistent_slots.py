"""PITON_PERSISTENT with store slots (PITON_STORE_SLOTS, on by default).

A coarsened program runs every store's acquire ... release once per program
id: no slot is held from one id into the next or into the thread end, a launch
leaves the semaphores as it found them, and no id waits for a DMA of the one
before (a store through a slot waited for its DMAs before releasing the slot;
the private rows' overlap is in tests/test_persistent.py). Every coarsened
build passes the all-paths protocol check of tests/test_store_slots_protocol.py
and runs under the contended timing of tests/test_store_slots.py (one VDW
engine, long jittered DMAs and starts, so QPUs sharing a slot wait for each
other), bit for bit against the plain build.
"""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from piton import compiler  # noqa: E402
from piton.backend.regalloc import RegAllocError  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.emulator import EmulatorError  # noqa: E402
from piton.runtime.launcher import run_emulator, semaphore_programs, slot_semaphores  # noqa: E402
from piton.runtime.pi import PiResult, PiRunner, run_on_pi  # noqa: E402
from primitive_kernels import copy_kernel  # noqa: E402
from test_persistent import (COPY_SIG, early_exit_kernel, inner_store_kernel, mixed_store_kernel,  # noqa: E402
                             store_first_kernel, tile_store_kernel)
from test_store_slots import contended, launch  # noqa: E402
from test_store_slots_protocol import protocol_errors  # noqa: E402

rng = np.random.default_rng(77)


@pytest.fixture
def env(monkeypatch):
    """env(PITON_PERSISTENT value or None); store slots and port-aware allocation at their defaults."""
    monkeypatch.delenv("PITON_STORE_SLOTS", raising=False)
    monkeypatch.delenv("PITON_PORT_ALLOC", raising=False)

    def set_k(value):
        if value is None:
            monkeypatch.delenv("PITON_PERSISTENT", raising=False)
        else:
            monkeypatch.setenv("PITON_PERSISTENT", str(value))
    return set_k


def _bound(grid):
    return tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)


def _waits(ck) -> int:
    return sum(isa.VW_WAIT in [s for op in (i.add, i.mul) if op for s in op.srcs()]
               for i in ck.program.instrs if i.kind == "alu")


def _check_coarsened(ck, k):
    assert ck.coarsen == k and ck.stats.get("store_slots") == 1
    assert any(lbl.startswith("programs") for lbl in ck.program.labels)
    assert protocol_errors(ck.program) == []


def differential(env, k, kernel, sig, cons, grid, values, seed, num_qpus=12):
    """Plain (default timing) and coarsened by k (contended): both with store
    slots, the same outputs bit for bit, semaphores back at 0 after the launch."""
    env(None)
    plain = compiler.compile(kernel, sig, cons, grid=_bound(grid))
    assert plain.coarsen == 1 and plain.stats.get("store_slots") == 1
    env(k)
    ck = compiler.compile(kernel, sig, cons, grid=_bound(grid))
    _check_coarsened(ck, k)
    want, got = copy.deepcopy(values), copy.deepcopy(values)
    run_emulator(plain, grid, want)
    launch(ck, grid, got, contended(seed, num_qpus))
    for name, v in want.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(got[name].view(np.uint32), v.view(np.uint32), err_msg=name)
    return ck, got


# -- suite kernels --------------------------------------------------------------------------

SUITE = [("copy", "emu", 4), ("add", "emu", 3), ("exp", "emu", 4), ("sqrt", "emu", 2), ("transpose", "emu", 2),
         ("matmul_8x64x4", "emu", 3), ("softmax", "pi", 2)]


@pytest.mark.parametrize("name,size,k", SUITE)
def test_suite_case_coarsened_with_slots(env, name, size, k):
    """Coarsened by k: every path follows the slot protocol, and a contended
    run is bit-identical to the plain slot build."""
    from perf.suite import CASES_BY_NAME, compile_case, default_check
    case = CASES_BY_NAME[name]
    run = case.make(size)
    if name == "softmax":
        run.grid = (24,)                    # 24 of its 256 rows (256 wide only at Pi size)
        run.expected = {o: v[:24 * 256] for o, v in run.expected.items()}
    env(None)
    plain, _ = compile_case(case, size, run)
    env(k)
    ck, _ = compile_case(case, size, run)
    _check_coarsened(ck, k)
    assert bool(ck.stats["rerolled"]) == bool(plain.stats["rerolled"])
    values = {n: (v.copy() if isinstance(v, np.ndarray) else v) for n, v in run.inputs.values.items()}
    want = copy.deepcopy(values)
    run_emulator(plain, run.grid, want)
    launch(ck, run.grid, values, contended(k + len(name)))
    for out, expected in run.expected.items():
        assert default_check(run, out, values[out][:expected.size], expected) is None
        np.testing.assert_array_equal(values[out].view(np.uint32), want[out].view(np.uint32))


@pytest.mark.parametrize("name", ["copy", "add", "exp", "log", "recip", "transpose", "softmax"])
def test_automatic_factor_keeps_slots(env, name):
    """PITON_PERSISTENT=1 at Pi sizes: the coarsened build keeps the plain
    build's 1 KiB slot stores and passes the protocol check."""
    from perf.suite import CASES_BY_NAME, compile_case
    env(1)
    ck, run = compile_case(CASES_BY_NAME[name], "pi")
    _check_coarsened(ck, 2 if name in ("transpose", "softmax") else 4)


# -- the id loop and the slot protocol ----------------------------------------------------

def test_no_wait_for_a_previous_id(env):
    """Coarsened, a slot kernel waits exactly where its plain build does (before
    each release, and between a store's DMAs): not at the start of an id, not
    before the thread end."""
    env(None)
    plain = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(16,))
    env(4)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(16,))
    _check_coarsened(ck, 4)
    assert _waits(ck) == _waits(plain)
    instrs, labels = ck.program.instrs, ck.program.labels
    start = labels[next(lbl for lbl in labels if lbl.startswith("programs"))]
    end = next(pc for pc, i in enumerate(instrs) if i.sig == "thrend")
    acquire = next(pc for pc in range(start, end) if instrs[pc].kind == "sem")
    text = [isa.format_instr(instrs[pc]) for pc in range(len(instrs))]
    assert not any("vw_wait" in t for t in text[start:acquire])
    release = max(pc for pc in range(end) if instrs[pc].kind == "sem")
    assert not instrs[release].acquire and not any("vw_wait" in t for t in text[release + 1:end + 3])


def test_holding_a_slot_into_the_next_id_deadlocks(env):
    """Without its releases a coarsened program's second id waits for the slot
    its first id still holds: the protocol check flags it, and one QPU running
    the launch deadlocks in the emulator."""
    env(4)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(8,))
    _check_coarsened(ck, 4)
    for pc, ins in enumerate(ck.program.instrs):
        if ins.kind == "sem" and not ins.acquire:
            ck.program.instrs[pc] = isa.Instr.nop()
    assert any("acquire while holding a slot" in e for e in protocol_errors(ck.program))
    n = 8 * 256
    with pytest.raises(EmulatorError, match="deadlock: QPU0 waits for the semaphore 0"):
        run_emulator(ck, (8,), {"x_ptr": np.zeros(n, np.float32), "out_ptr": np.zeros(n, np.float32), "n": n},
                     num_qpus=1)


@pytest.mark.parametrize("blocks,k,num_qpus", [(10, 4, 12), (7, 3, 1), (9, 2, 5), (24, 4, 12), (3, 4, 2)])
def test_ragged_copy(env, blocks, k, num_qpus):
    """Grids that are not a multiple of k, a masked last block (its rows are
    filled once for both store versions), few and many QPUs per slot."""
    n = blocks * 256 - 37
    x = rng.standard_normal(n).astype(np.float32)
    _, got = differential(env, k, copy_kernel, COPY_SIG, {"BLOCK": 256}, (blocks,),
                          {"x_ptr": x, "out_ptr": np.full(n + 64, -2.0, np.float32), "n": n}, blocks + k, num_qpus)
    np.testing.assert_array_equal(got["out_ptr"][:n], x)
    assert np.all(got["out_ptr"][n:] == -2.0)


def test_slot_store_scatter_and_scalar(env):
    """Three stores per id, each through the slot: contiguous, reversed
    (element by element) and scalar."""
    n, block, blocks = 7 * 256 - 9, 256, 7
    x = rng.standard_normal(n).astype(np.float32)
    differential(env, 3, mixed_store_kernel,
                 {"x_ptr": "*fp32", "out_ptr": "*fp32", "rev_ptr": "*fp32", "last_ptr": "*fp32", "n": "i32"},
                 {"BLOCK": block}, (blocks,),
                 {"x_ptr": x, "out_ptr": np.zeros(n, np.float32), "rev_ptr": np.zeros(blocks * block, np.float32),
                  "last_ptr": np.zeros(blocks, np.float32), "n": n}, 11)


def test_early_return(env):
    """`return` before the store skips to the next id holding no slot."""
    x = rng.standard_normal(10 * 256).astype(np.float32)
    _, got = differential(env, 4, early_exit_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_blocks": "i32"},
                          {"BLOCK": 256}, (10,), {"x_ptr": x, "out_ptr": np.zeros(10 * 256, np.float32),
                                                  "n_blocks": 7}, 12, num_qpus=3)
    np.testing.assert_array_equal(got["out_ptr"][:7 * 256], x[:7 * 256] * 2)


@pytest.mark.parametrize("k,blocks", [(3, 7), (4, 9)])
def test_stores_before_loads(env, k, blocks):
    """A store before any load, then two masked stores into the same rows."""
    n = blocks * 256 - 7
    differential(env, k, store_first_kernel, {"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32", "n": "i32"},
                 {"BLOCK": 256}, (blocks,),
                 {"x_ptr": rng.standard_normal(n).astype(np.float32), "a_ptr": np.full(n + 256, -9.0, np.float32),
                  "b_ptr": np.full(n + 256, -9.0, np.float32), "n": n}, k * blocks)


@pytest.mark.parametrize("k,blocks", [(3, 7), (4, 8)])
def test_store_loop_with_zero_trips(env, k, blocks):
    """A run-time loop storing 0, 1 or 2 times per id, then a store after it."""
    differential(env, k, inner_store_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "sum_ptr": "*fp32"},
                 {"BLOCK": 256}, (blocks,),
                 {"x_ptr": rng.standard_normal(blocks * 3 * 256).astype(np.float32),
                  "out_ptr": np.zeros(blocks * 3 * 256, np.float32),
                  "sum_ptr": np.zeros(blocks * 256, np.float32)}, k + blocks, num_qpus=4)


# 0x4000 + 40: the row gaps do not fit the VDW stride field (one DMA per row)
@pytest.mark.parametrize("stride_o", [40, 0x4000 + 40])
def test_tile_store_with_runtime_row_stride(env, stride_o):
    """16-row tiles (one slot) with ragged edges and a run-time row stride."""
    M, N = 37, 37
    grid = (3 * 3,)
    differential(env, 3, tile_store_kernel,
                 {"x_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32", "stride_x": "i32", "stride_o": "i32"},
                 {"BM": 16, "BN": 16}, grid,
                 {"x_ptr": rng.standard_normal(M * N).astype(np.float32),
                  "out_ptr": np.full((M - 1) * stride_o + N + 16, -3.0, np.float32), "M": M, "N": N,
                  "stride_x": N, "stride_o": stride_o}, stride_o)


@pytest.mark.parametrize("reload", [False, True])
def test_uniform_protocols(env, monkeypatch, reload):
    """Both ways of keeping the kernel's uniforms (in registers, or re-read
    through unif_addr for every id) with slot stores."""
    real = compiler._compile_tile

    def only(fn, signature, constants, name, pid_max, options, chunk=0):
        if options.get("coarsen", 1) > 1 and options["coarsen_reload"] != reload:
            raise RegAllocError("forced by the test")
        return real(fn, signature, constants, name, pid_max, options, chunk)
    monkeypatch.setattr(compiler, "_compile_tile", only)
    n = 9 * 256 - 100
    ck, _ = differential(env, 4, copy_kernel, COPY_SIG, {"BLOCK": 256}, (9,),
                         {"x_ptr": rng.standard_normal(n).astype(np.float32), "out_ptr": np.zeros(n, np.float32),
                          "n": n}, 5)
    assert ck.stats["coarsen_reload"] == reload


def test_coarsened_builds_keep_store_slots(env, monkeypatch):
    """A coarsened version that allocates only with the private VPM rows is not
    taken over a plain build with slots (its 1 KiB DMAs gain more); a version
    that keeps the slots is, even if it is not the first one tried."""
    env(None)
    plain = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,))
    assert plain.stats.get("store_slots") == 1
    real = compiler._build

    def failing(fail):
        def build(tile, ranges, options, interleave):
            if options.get("coarsen", 1) > 1 and options.get("store_slots") and fail(options):
                raise RegAllocError("forced by the test")   # _compile_tile retries with the private rows
            return real(tile, ranges, options, interleave)
        return build
    env(1)
    monkeypatch.setattr(compiler, "_build", failing(lambda o: True))
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,))
    assert ck.coarsen == 1 and "coarsen" not in ck.stats
    assert list(ck.words) == list(plain.words) and ck.uniforms == plain.uniforms
    monkeypatch.setattr(compiler, "_build", failing(lambda o: o["coarsen_hoist"]))
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,))
    assert ck.coarsen == 4 and ck.stats["store_slots"] == 1 and ck.stats["coarsen_hoist"] == 0
    assert protocol_errors(ck.program) == []


def test_pi_launch_of_a_coarsened_slot_kernel(env):
    """run_on_pi queues ceil(gx / k) programs through piton_launch_slots, with
    the kernel's slot semaphores set up around the launch."""
    env(4)
    ck = compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(16,))
    _check_coarsened(ck, 4)

    class Runner:
        def run(self, code_words, buffers, programs, outputs, repeats=1, label="piton", **kw):
            sems = slot_semaphores(code_words)
            self.programs = programs
            self.source = object.__new__(PiRunner)._generate(code_words, buffers, programs, outputs, repeats,
                                                             label, {}, semaphores=sems)
            self.code = code_words
            return PiResult({o: np.zeros(b.nbytes // 4, np.uint32) for b in buffers for o in outputs
                             if b.name == o}, [1], "", [])
    runner = Runner()
    n = 13 * 256
    run_on_pi(ck, (13,), {"x_ptr": np.zeros(n, np.float32), "out_ptr": np.zeros(n, np.float32), "n": n},
              runner=runner)
    assert len(runner.programs) == 4 and list(runner.code) == list(ck.words)
    pid0 = ck.uniforms.index(("pid", "pid0"))
    nprog0 = ck.uniforms.index(("nprog", "nprog0"))
    assert [p[pid0] for p in runner.programs] == [0, 1, 2, 3] and {p[nprog0] for p in runner.programs} == {13}
    init, _ = semaphore_programs(slot_semaphores(ck.words))
    words = len(ck.words)
    assert (f"piton_launch_slots(m.bus, unif_ptrs, 4, m.bus + {8 * words}u, m.bus + {8 * (words + len(init))}u)"
            in runner.source)
