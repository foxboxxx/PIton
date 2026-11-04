"""Store slots (PITON_STORE_SLOTS=1): a static check of the slot protocol on
every path of the compiled code, and slot kernels against the same kernels
compiled without slots (bit-identical outputs).

The emulator only sees the paths a test happens to take, and with a given
interleaving of the QPUs.  `protocol_errors` walks every path of the program
(branch conditions ignored, delay slots included) and checks what the
hardware needs and the emulator cannot fully show: every VPM write, VDW setup
and DMA start happens while this QPU holds its slot; a DMA is waited for
before the next one starts, before the slot is released and before the
program ends; no semaphore instruction sits under the mutex or in a branch
delay slot; acquire and release alternate; the thread end and its delay
slots touch neither the VPM nor a semaphore.
"""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from test_store_slots import build, contended, launch  # noqa: E402

rng = np.random.default_rng(41)


def _ops(ins):
    return [op for op in (ins.add, ins.mul) if op is not None]


def _reads(ins, reg):
    return ins.kind == "alu" and any(op.op != "nop" and any(
        isinstance(s, isa.Reg) and s.name == reg.name and s.raddr == reg.raddr for s in op.srcs())
        for op in _ops(ins))


def _writes(ins, reg):
    return ins.kind in ("alu", "ldi") and any(
        isinstance(op.dst, isa.Reg) and op.dst.name == reg.name and (op.op != "nop" or ins.kind == "ldi")
        for op in _ops(ins))


def protocol_errors(prog, slots: bool = True) -> list:
    """All-paths check of a compiled program; [] when it follows the protocol."""
    instrs, labels = prog.instrs, prog.labels
    errors, seen = set(), set()
    work = [(0, (False, False, False), None)]     # pc, (slot held, DMA in flight, mutex held), delay slots
    while work:
        pc, st, br = work.pop()
        if (pc, st, br) in seen or pc >= len(instrs):
            continue
        seen.add((pc, st, br))
        held, dma, mtx = st
        ins = instrs[pc]
        if ins.kind == "sem":
            if mtx:
                errors.add(f"pc {pc}: semaphore under the mutex")
            if br is not None:
                errors.add(f"pc {pc}: semaphore in a branch delay slot")
            if ins.acquire:
                if held or dma:
                    errors.add(f"pc {pc}: acquire while holding a slot or with a DMA in flight")
                held = True
            else:
                if not held:
                    errors.add(f"pc {pc}: release without a slot")
                if dma:
                    errors.add(f"pc {pc}: release while the DMA may still read the slot")
                held = False
        if _reads(ins, isa.VW_WAIT):
            dma = False
        if _reads(ins, isa.MUTEX):
            if mtx:
                errors.add(f"pc {pc}: mutex acquired twice")
            mtx = True
        if _writes(ins, isa.VPM):
            if slots and not held:
                errors.add(f"pc {pc}: VPM write without the slot")
            if dma:
                errors.add(f"pc {pc}: VPM write while this QPU's DMA runs")
        if _writes(ins, isa.VW_SETUP) and slots and not held:
            errors.add(f"pc {pc}: VPM / VDW setup without the slot")
        if _writes(ins, isa.VW_ADDR):
            if slots and not held:
                errors.add(f"pc {pc}: DMA start without the slot")
            if dma:
                errors.add(f"pc {pc}: DMA start while the previous DMA runs")
            if not mtx:
                errors.add(f"pc {pc}: DMA start outside the mutex")
            dma = True
        if _writes(ins, isa.MUTEX):
            mtx = False
        st = (held, dma, mtx)
        if ins.kind == "alu" and ins.sig == "thrend":
            if held or dma or mtx:
                errors.add(f"pc {pc}: thread end holding a slot, the mutex or an unwaited DMA")
            for q in range(pc, min(pc + 3, len(instrs))):
                x = instrs[q]
                if x.kind == "sem" or any(_writes(x, r) for r in (isa.VPM, isa.VW_SETUP, isa.VW_ADDR)) or \
                        any(_reads(x, r) for r in (isa.VPM, isa.VW_WAIT)):
                    errors.add(f"pc {q}: VPM, VDW or semaphore access in the thread end's delay slots")
            continue
        if ins.kind == "branch":
            if br is not None:
                errors.add(f"pc {pc}: branch in a delay slot")
            work.append((pc + 1, st, (3, labels[ins.target], ins.bcond)))
        elif br is not None:
            n, target, cond = br
            if n > 1:
                work.append((pc + 1, st, (n - 1, target, cond)))
            else:
                work.append((target, st, None))
                if cond != "always":
                    work.append((pc + 1, st, None))
        else:
            work.append((pc + 1, st, None))
    return sorted(errors)


@pytest.fixture
def slots(monkeypatch):
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")


SLOT_CASES = [("copy", "emu"), ("add", "emu"), ("scaled_relu", "emu"), ("exp", "emu"), ("log", "emu"),
              ("sqrt", "emu"), ("transpose", "emu"), ("transpose", "pi"), ("softmax", "pi"),
              ("matmul_8x64x4", "emu")]


@pytest.mark.parametrize("name,size", SLOT_CASES)
def test_protocol_on_every_path(slots, name, size):
    from perf.suite import CASES_BY_NAME, compile_case
    ck, _ = compile_case(CASES_BY_NAME[name], size)
    assert ck.stats.get("store_slots") == 1
    assert protocol_errors(ck.program) == []


def test_protocol_check_catches_mutations(slots):
    """The checker itself: drop a release's vw_wait, or an acquire."""
    from primitive_kernels import copy_kernel
    ck = build(copy_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}, (8,))
    prog = ck.program
    release = [pc for pc, i in enumerate(prog.instrs) if i.kind == "sem" and not i.acquire][0]
    acquire = [pc for pc, i in enumerate(prog.instrs) if i.kind == "sem" and i.acquire][0]
    no_wait = copy.deepcopy(prog)
    no_wait.instrs[release - 1] = isa.Instr.nop()
    assert any("release while the DMA" in e for e in protocol_errors(no_wait))
    no_acquire = copy.deepcopy(prog)
    no_acquire.instrs[acquire] = isa.Instr.nop()
    assert any("VPM write without the slot" in e for e in protocol_errors(no_acquire))


def differential(kernel, sig, cons, grid, values, monkeypatch, seed=0):
    """Slots off (default timing) and on (contended): bit-identical outputs, and
    the slot build passes the protocol check."""
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")
    off = build(kernel, sig, cons, grid)
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")
    on = build(kernel, sig, cons, grid)
    assert on.stats.get("store_slots") == 1
    assert protocol_errors(on.program) == []
    a, b = copy.deepcopy(values), copy.deepcopy(values)
    run_emulator(off, grid, a)
    launch(on, grid, b, contended(seed))
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(b[k].view(np.uint32), v.view(np.uint32), err_msg=k)
    return b


@piton.jit
def xorshift_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0)
    x = x ^ (x << 13)
    x = x ^ (x >> 17)
    x = x ^ (x << 5)
    x = x + 0x1234567
    x = x ^ (x << 13)
    x = x ^ (x >> 17)
    x = x ^ (x << 5)
    x = x + 0x7654321
    x = x ^ (x << 13)
    x = x ^ (x >> 17)
    x = x ^ (x << 5)
    x = x - 99
    x = x ^ (x << 7)
    x = x ^ (x >> 9)
    x = x ^ (x << 3)
    x = x ^ (x >> 11)
    tl.store(out_ptr + offs, x, mask=m)


def test_integer_store_queue(monkeypatch):
    """An int32 kernel rerolled with the store queue (int zeros shifted out)."""
    n = 256 * 5 - 77
    x = rng.integers(-2 ** 31, 2 ** 31 - 1, n, dtype=np.int64).astype(np.int32)
    got = differential(xorshift_k, {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32"}, {"BLOCK": 256}, (8,),
                       {"x_ptr": x, "out_ptr": np.full(n + 32, 77, np.int32), "n": n}, monkeypatch, 3)["out_ptr"]
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")
    ck = build(xorshift_k, {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32"}, {"BLOCK": 256}, (8,))
    assert ck.stats["rerolled"] and "piton.shift_in" in ck.asm["ttir_rerolled"]
    want = x.copy()
    with np.errstate(over="ignore"):
        for k in (0x1234567, 0x7654321, -99):
            want = want ^ (want << 13)
            want = want ^ (want >> 17)
            want = want ^ (want << 5)
            want = (want + np.int32(k)).astype(np.int32)
        for s, left in ((7, True), (9, False), (3, True), (11, False)):
            want = want ^ ((want << s) if left else (want >> s))
    np.testing.assert_array_equal(got[:n], want)
    np.testing.assert_array_equal(got[n:], 77)


@piton.jit
def scatter_k(x_ptr, out_ptr, k, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x * 2.0)                                   # one slot DMA
    i = tl.arange(0, 32)
    h = tl.load(x_ptr + pid * BLOCK + i)
    tl.store(out_ptr + 65536 + pid * 64 + 2 * i, h, mask=i < k)          # masked scatter
    j = pid * 32 + i
    tl.store(out_ptr + 131072 + j, h + 1.0, mask=(j % 3) == 0)          # mask without bounds


@pytest.mark.parametrize("k", [0, 5, 32])
def test_masked_scatter_stores(monkeypatch, k):
    programs, block = 9, 256
    x = rng.standard_normal(programs * block).astype(np.float32)
    differential(scatter_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "k": "i32"}, {"BLOCK": block}, (programs,),
                 {"x_ptr": x, "out_ptr": np.zeros(131072 + programs * 32 + 64, np.float32), "k": k}, monkeypatch, k)


@piton.jit
def if_in_loop_k(x_ptr, out_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for i in range(iters):
        offs = (pid * iters + i) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        if i % 2 == 1:
            tl.store(out_ptr + offs, x * 2.0)
        else:
            tl.store(out_ptr + offs, x + 1.0, mask=tl.arange(0, BLOCK) < 200)


@pytest.mark.parametrize("iters", [0, 1, 4])
def test_stores_in_branches_in_loops(monkeypatch, iters):
    programs, block = 8, 256
    n = programs * max(iters, 1) * block
    x = rng.standard_normal(n).astype(np.float32)
    differential(if_in_loop_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "iters": "i32"}, {"BLOCK": block},
                 (programs,), {"x_ptr": x, "out_ptr": np.full(n, -5.0, np.float32), "iters": iters}, monkeypatch,
                 iters)


@piton.jit
def tile2d_k(x_ptr, out_ptr, n_rows, n_cols, stride, R: tl.constexpr, L: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * L + tl.arange(0, L)
    x = tl.load(x_ptr + rows[:, None] * 64 + cols[None, :])
    m = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    tl.store(out_ptr + rows[:, None] * stride + cols[None, :], x + 1.0, mask=m)


@pytest.mark.parametrize("R,L,stride", [(32, 16, 64), (32, 16, 30000), (16, 16, 100), (8, 64, 64), (8, 32, 30000)])
def test_2d_grid_masked_tiles(monkeypatch, R, L, stride):
    """Two DMAs per store (32 rows), a 2-D grid, row and column masks, and row
    gaps that do or do not fit the VDW stride field."""
    gx, gy = 3, 64 // L
    x = rng.standard_normal((gx * R, 64)).astype(np.float32)
    differential(tile2d_k, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_rows": "i32", "n_cols": "i32",
                            "stride": "i32"}, {"R": R, "L": L}, (gx, gy),
                 {"x_ptr": x, "out_ptr": np.full(stride * gx * R + 64, -4.0, np.float32), "n_rows": gx * R - 3,
                  "n_cols": 59, "stride": stride}, monkeypatch, R + stride)
