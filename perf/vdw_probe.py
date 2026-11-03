"""How fast can the QPUs write memory? VDW DMA throughput on the Pi.

Elementwise kernels are limited by stores (perf/store_experiment.py: copy is
~650 MB/s, store-only takes as long as copy). PIton stores through 4 VPM
rows per QPU, so a 256-float store is four 256-byte DMAs, each waited for
before the rows are refilled. Each configuration here writes 1 MiB in total
from hand-written QPU code, varying:

  qpus      programs storing at once (one per QPU)
  rows      16-word VPM rows per DMA (bytes per DMA = 64 * rows)
  wait      vw_wait before every DMA (as compiled code must, to reuse VPM rows)
            or only once at the end (DMAs queue back to back; data not reused)

Output: MB/s for each, so we can tell a slow shared engine (throughput flat
across qpus) from per-QPU DMA latency (throughput grows with qpus and rows).

    .venv/bin/python perf/vdw_probe.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import Program  # noqa: E402
from piton.runtime.pi import Buffer, PiRunner  # noqa: E402

TOTAL = 1 << 20

def vdw_setup(units: int, depth: int, y: int) -> int:
    """Horizontal 32-bit VDW setup word (VPM row y)."""
    return 0x80000000 | ((units & 127) << 23) | ((depth & 127) << 16) | 0x4000 | (y << 7)


def program(rows: int, rows_per_qpu: int, dma_bytes: int, wait: bool) -> list:
    """Store `count` (uniform 2) DMAs of `rows` VPM rows from this QPU's rows,
    to consecutive memory starting at uniform 1."""
    A, B, R = isa.RA, isa.RB, isa.ACC

    def alu(add=None, mul=None, setf=False, sig="none"):
        return isa.Instr(add=add, mul=mul, setf=setf, sig=sig)

    def ldi(dst, value):
        return isa.Instr(kind="ldi", add=isa.AluOp("mov", dst, isa.NOP), imm=value)

    mov = lambda d, s: isa.AluOp("mov", d, s)
    code = [
        alu(mov(A[0], isa.UNIF)),                                   # output address
        alu(mov(A[1], isa.UNIF)),                                   # number of DMAs
        alu(mov(R[0], isa.QPU_NUM)),
        ldi(R[1], rows_per_qpu),
        alu(mul=isa.AluOp("mul24", R[0], R[0], R[1])),              # first VPM row of this QPU
        alu(isa.AluOp("shl", R[0], R[0], isa.Imm(7))),              # VDW setup y field
        ldi(R[1], vdw_setup(rows, 16, 0)),
        alu(isa.AluOp("add", A[2], R[0], R[1])),
        ldi(B[0], dma_bytes),
        ldi(isa.VW_SETUP, 0xC0000000),                              # stride 0: rows back to back
    ]
    loop = len(code)
    code += [
        alu(mov(isa.NOP, isa.VW_WAIT)) if wait else isa.Instr.nop(),
        alu(mov(isa.VW_SETUP, A[2])),
        alu(mov(isa.VW_ADDR, A[0])),
        alu(isa.AluOp("add", A[0], A[0], B[0])),
        alu(isa.AluOp("sub", A[1], A[1], isa.Imm(1)), setf=True),
        isa.Instr(kind="branch", bcond="anynz", target="loop", rel=True,
                  add=isa.AluOp("mov", isa.NOP, isa.NOP), mul=isa.AluOp("mov", isa.NOP, isa.NOP)),
        isa.Instr.nop(), isa.Instr.nop(), isa.Instr.nop(),
        alu(mov(isa.NOP, isa.VW_WAIT)),
        alu(sig="thrend"), isa.Instr.nop(), isa.Instr.nop(),
    ]
    return Program("vdw_probe", code, {"loop": loop}).words()


def measure(runner, qpus: int, rows: int, wait: bool) -> float:
    dma_bytes = 64 * rows
    per_prog = TOTAL // qpus
    count = per_prog // dma_bytes
    words = program(rows, max(rows, 4) if qpus * max(rows, 4) <= 64 else rows, dma_bytes, wait)
    programs = [[("buf", "out", p * per_prog), count] for p in range(qpus)]
    res = runner.run(words, [Buffer("out", TOTAL)], programs, [], repeats=5, label=f"vdw q{qpus} r{rows}")
    return TOTAL / min(res.times_us)                                # bytes/us = MB/s


def main():
    runner = PiRunner(timeout=120)
    configs = [(12, 1, True), (12, 2, True), (12, 4, True), (1, 4, True), (4, 4, True),
               (4, 8, True), (4, 16, True), (12, 4, False), (4, 16, False), (1, 16, True)]
    print(f"{'qpus':>4} {'rows':>4} {'bytes/DMA':>9} {'wait':>5}  {'MB/s':>7}")
    for qpus, rows, wait in configs:
        try:
            mbs = measure(runner, qpus, rows, wait)
            print(f"{qpus:4} {rows:4} {64 * rows:9} {str(wait):>5}  {mbs:7.0f}", flush=True)
        except Exception as ex:
            print(f"{qpus:4} {rows:4} {64 * rows:9} {str(wait):>5}  FAILED {str(ex).splitlines()[-1][:60]}",
                  flush=True)


if __name__ == "__main__":
    main()


def capacity_program(rows: int) -> list:
    """Write row r of the VPM with r * 16 + lane, then DMA rows 0..rows-1 out in
    16-row blocks. Rows that alias (VPM smaller than `rows`) come back wrong."""
    A, R = isa.RA, isa.ACC

    def alu(add=None, mul=None, sig="none"):
        return isa.Instr(add=add, mul=mul, sig=sig)

    def ldi(dst, value):
        return isa.Instr(kind="ldi", add=isa.AluOp("mov", dst, isa.NOP), imm=value)

    code = [alu(isa.AluOp("mov", A[0], isa.UNIF)),
            ldi(isa.VW_SETUP, (1 << 12) | 0xA00),                    # h32 writes from row 0, stride 1
            alu(isa.AluOp("mov", R[0], isa.ELEM_NUM))]
    for r in range(rows):
        code.append(alu(isa.AluOp("mov", isa.VPM, R[0])))
        code.append(alu(isa.AluOp("add", R[0], R[0], isa.Imm(15))))
        code.append(alu(isa.AluOp("add", R[0], R[0], isa.Imm(1))))
    code.append(ldi(isa.VW_SETUP, 0xC0000000))
    for block in range(rows // 16):
        code += [alu(isa.AluOp("mov", isa.NOP, isa.VW_WAIT)),
                 ldi(isa.VW_SETUP, vdw_setup(16, 16, 16 * block)),
                 ldi(R[1], 1024 * block),
                 alu(isa.AluOp("add", isa.VW_ADDR, A[0], R[1]))]
    code += [alu(isa.AluOp("mov", isa.NOP, isa.VW_WAIT)),
             alu(sig="thrend"), isa.Instr.nop(), isa.Instr.nop()]
    return Program("vpm_capacity", code, {}).words()


def capacity(runner, rows: int = 128) -> int:
    """Number of leading VPM rows that store and come back correctly."""
    import numpy as np
    res = runner.run(capacity_program(rows), [Buffer("out", 64 * rows)], [[("buf", "out", 0)]], ["out"],
                     label="vpm capacity")
    got = res.outputs["out"].reshape(rows, 16)
    good = np.all(got == np.arange(rows * 16, dtype=np.uint32).reshape(rows, 16), axis=1)
    return int(np.argmin(good)) if not good.all() else rows
