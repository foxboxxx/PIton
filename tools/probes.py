"""Hardware probes: tiny hand-written QPU programs whose results pin down
behaviour the compiler relies on.  Each probe runs on the Pi (if attached)
and in the emulator, and prints both.

    python3 tools/probes.py [probe-name ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piton.qpu.asm import vc4asm_assemble          # noqa: E402
from piton.qpu.emulator import V3D                  # noqa: E402
from piton.runtime.pi import Buffer, PiRunner, pi_available   # noqa: E402

PROLOGUE = """
.set vpm_setup_h32(num, stride, y), ((num & 15) << 20) | ((stride & 63) << 12) | 0xa00 | y
.set vdw_h32(units, depth, y, x), 0x80000000 | ((units & 127) << 23) | ((depth & 127) << 16) | 0x4000 | (y << 7) | (x << 3)
.set vdw_stride(blockmode, stride), 0xc0000000 | (blockmode << 16) | stride
.set vdr_h32(mpitch, rowlen, nrows, vpitch, y, x), 0x80000000 | (mpitch << 24) | ((rowlen & 15) << 20) | ((nrows & 15) << 16) | ((vpitch & 15) << 12) | (y << 4) | x
"""

EPILOGUE = """
    nop; thrend
    nop
    nop
"""


def store_rows(n, out_reg="ra0"):
    """DMA VPM rows 0..n-1 (16 words each) to the address in `out_reg`."""
    return f"""
    mov -, vw_wait
    ldi vw_setup, vdw_h32({n}, 16, 0, 0)
    mov vw_addr, {out_reg}
    mov -, vw_wait
"""


PROBES = {}


def probe(fn):
    PROBES[fn.__name__] = fn
    return fn


@probe
def broadcast():
    """r5rep broadcast read back immediately; rotation direction; r5quad."""
    src = PROLOGUE + """
    mov ra0, unif
    mov r0, elem_num
    itof r1, elem_num
    nop
    ldi vw_setup, vpm_setup_h32(4, 1, 0)
    nop; mov r5rep, r0 >> 3          # upward 3: lane0 <- lane13
    mov vpm, r5                      # row0: r5 read in the very next instruction
    nop; mov r2, r0 >> 1             # row1: upward rotation by 1
    mov r5quad, r0
    mov vpm, r2
    mov vpm, r5                      # row2: r5quad -> per-quad element 0
    nop; mov r5rep, r1 << 5          # vc4asm '<< 5' = upward 11: lane0 <- lane5
    nop; fmul vpm, r5, r1            # row3: 5.0 * i, r5 used immediately by fmul
""" + store_rows(4) + EPILOGUE
    expect = np.array([[13] * 16, [(i - 1) % 16 for i in range(16)],
                       [4 * (i // 4) for i in range(16)],
                       np.float32(5.0 * np.arange(16)).view(np.uint32)], dtype=np.uint32)
    return src, [Buffer("out", 256)], [[("buf", "out", 0)]], ["out"], expect.ravel()


@probe
def vdw_long_rows():
    """Can one horizontal VDW memory row be longer than a 16-word VPM row?"""
    src = PROLOGUE + """
    mov ra0, unif
    mov r0, elem_num
    ldi vw_setup, vpm_setup_h32(4, 1, 0)
    mov vpm, r0
    add r1, r0, 16
    mov vpm, r1
    add r1, r1, 15
    add r1, r1, 1
    mov vpm, r1
    add r1, r1, 15
    add r1, r1, 1
    mov vpm, r1
    mov -, vw_wait
    ldi vw_setup, vdw_h32(1, 64, 0, 0)      # 1 row of 64 words
    mov vw_addr, ra0
    mov -, vw_wait
    ldi r2, 256
    add ra1, ra0, r2
    ldi vw_setup, vdw_h32(2, 32, 0, 0)      # 2 rows of 32 words, blockmode 1
    ldi vw_setup, vdw_stride(1, 0)
    nop
    mov vw_addr, ra1
    mov -, vw_wait
""" + EPILOGUE
    expect = np.concatenate([np.arange(64), np.arange(64)]).astype(np.uint32)
    return src, [Buffer("out", 512)], [[("buf", "out", 0)]], ["out"], expect


@probe
def vdw_strided():
    """4 rows x 16 words with a 64-byte gap between memory rows."""
    src = PROLOGUE + """
    mov ra0, unif
    mov r0, elem_num
    ldi vw_setup, vpm_setup_h32(4, 1, 0)
    mov vpm, r0
    add r0, r0, 15
    add r0, r0, 1
    mov vpm, r0
    add r0, r0, 15
    add r0, r0, 1
    mov vpm, r0
    add r0, r0, 15
    add r0, r0, 1
    mov vpm, r0
    mov -, vw_wait
    ldi vw_setup, vdw_h32(4, 16, 0, 0)
    ldi vw_setup, vdw_stride(0, 64)
    mov vw_addr, ra0
    mov -, vw_wait
""" + EPILOGUE
    exp = np.zeros((4, 32), dtype=np.uint32)
    for r in range(4):
        exp[r, :16] = np.arange(16) + 16 * r
    out = np.zeros(128, dtype=np.uint32)
    out_init = out.copy()
    return src, [Buffer("out", 512, init=out_init)], [[("buf", "out", 0)]], ["out"], exp.ravel()


@probe
def tmu_fifo():
    """4 outstanding requests on each TMU, then drain (8 per TMU overflows on hardware)."""
    lines = ["    mov ra0, unif", "    mov ra1, unif", "    shl r0, elem_num, 2", "    add r1, ra1, r0",
             "    ldi r2, 64"]
    for j in range(4):
        lines.append("    mov t0s, r1")
        lines.append("    add r1, r1, r2")
        lines.append("    mov t1s, r1")
        lines.append("    add r1, r1, r2")
    lines.append("    ldi vw_setup, vpm_setup_h32(8, 1, 0)")
    for j in range(4):
        lines.append("    nop; nop; ldtmu0")
        lines.append("    mov vpm, r4; nop; ldtmu1")
        lines.append("    mov vpm, r4")
    src = PROLOGUE + "\n".join(lines) + "\n" + store_rows(8) + EPILOGUE
    data = (np.arange(256, dtype=np.uint32) * 7 + 3).astype(np.uint32)
    return src, [Buffer("out", 1024), Buffer("inp", 1024, init=data)], \
        [[("buf", "out", 0), ("buf", "inp", 0)]], ["out"], data[:128]


@probe
def multi_qpu():
    """24 programs: each writes (program id, qpu_num) to its own row via VPM row qpu_num*4."""
    src = PROLOGUE + """
    mov ra0, unif            # out
    mov ra1, unif            # program id
    mov r0, qpu_num
    shl r1, r0, 2            # vpm row = 4*qpu_num
    ldi r2, vpm_setup_h32(1, 1, 0)
    add vw_setup, r1, r2
    mov r3, ra1
    nop
    mov vpm, r3; nop
    mov vpm, r0
    ldi vw_setup, vpm_setup_h32(1, 1, 1)
    shl r1, r1, 7
    ldi r2, vdw_h32(1, 1, 0, 0)
    add vw_setup, r1, r2
    shl r1, ra1, 2
    add vw_addr, ra0, r1
    mov -, vw_wait
    ldi r2, 0x800000        # units field +1? no: second DMA for qpu num
""" + EPILOGUE
    return None


def run(name):
    spec = PROBES[name]()
    if spec is None:
        return
    src, buffers, programs, outputs, expect = spec
    words, warnings = vc4asm_assemble(src, verify=True)
    for w in warnings:
        print("  vc4asm:", w)
    # emulator
    v = V3D(strict=False)
    addrs = {}
    for b in buffers:
        addrs[b.name] = v.mem.alloc(b.nbytes)
        if b.init is not None:
            v.mem.write_words(addrs[b.name], b.init.view(np.uint32).ravel())
    unif_addrs = []
    for p in programs:
        a = v.mem.alloc(4 * len(p), 16)
        vals = [a if u == ("self",) else (addrs[u[1]] + u[2] if isinstance(u, tuple) else u) for u in p]
        v.mem.write_words(a, np.array(vals, dtype=np.uint32))
        unif_addrs.append(a)
    try:
        v.run(v.load(words), unif_addrs)
        emu = v.mem.read_array(addrs[outputs[0]], (len(expect),), np.uint32)
        print(f"[{name}] emulator matches expectation: {np.array_equal(emu, expect)}")
    except Exception as e:
        emu = None
        print(f"[{name}] emulator error: {e}")
    if pi_available():
        res = PiRunner().run(words, buffers, programs, outputs, label=name)
        hw = res.outputs[outputs[0]][:len(expect)]
        ok = np.array_equal(hw, expect)
        print(f"[{name}] hardware matches expectation: {ok}  ({res.times_us} us)")
        if not ok:
            print("  expect:", expect.reshape(-1, 16) if len(expect) % 16 == 0 else expect)
            print("  hw    :", hw.reshape(-1, 16) if len(hw) % 16 == 0 else hw)
        if emu is not None and not np.array_equal(emu, hw):
            print("  emulator differs from hardware!")
            print("  emu   :", emu.reshape(-1, 16) if len(emu) % 16 == 0 else emu)




def timing_probe(name, body, iters, per_iter_instrs, nprog=1):
    """Measure wall time of `iters` loop iterations on the Pi."""
    src = PROLOGUE + f"""
    mov ra0, unif
    mov ra1, unif
    ldi ra2, {iters}
    shl r0, elem_num, 2
    add ra3, ra1, r0
    nop
:loop
{body}
    sub.setf ra2, ra2, 1
    brr.anynz -, :loop
    nop
    nop
    nop
""" + EPILOGUE
    words, _ = vc4asm_assemble(src, verify=False)
    data = np.arange(4096, dtype=np.uint32)
    bufs = [Buffer("out", 64), Buffer("inp", 16384, init=data)]
    progs = [[("buf", "out", 0), ("buf", "inp", 0)] for _ in range(nprog)]
    res = PiRunner().run(words, bufs, progs, ["out"], repeats=3, label=name)
    t = min(res.times_us)
    instrs = iters * per_iter_instrs * nprog
    print(f"[{name}] {t} us for {iters} iters x {nprog} programs -> "
          f"{instrs / t:.1f} M instr/s total, {t * 1000.0 / iters:.1f} ns/iter per program")


def timings():
    n = 200000
    timing_probe("nop_loop", "    nop\n    nop\n    nop\n    nop\n    nop", n, 9)
    timing_probe("nop_loop_12qpu", "    nop\n    nop\n    nop\n    nop\n    nop", n, 9, nprog=12)
    timing_probe("tmu_latency", "    mov t0s, ra3\n    nop; nop; ldtmu0\n    mov r0, r4", n, 7)
    timing_probe("tmu_4_inflight", "\n".join(["    mov t0s, ra3"] * 4 + ["    nop; nop; ldtmu0"] * 4), n, 12)
    timing_probe("tmu_8_inflight_2tmu", "\n".join(["    mov t0s, ra3", "    mov t1s, ra3"] * 4 +
                                             ["    nop; nop; ldtmu0", "    nop; nop; ldtmu1"] * 4), n, 20)
    timing_probe("tmu_8_inflight_12qpu", "\n".join(["    mov t0s, ra3", "    mov t1s, ra3"] * 4 +
                                              ["    nop; nop; ldtmu0", "    nop; nop; ldtmu1"] * 4),
                 n // 4, 20, nprog=12)


@probe
def unif_reset():
    """Re-reading uniforms by rewriting unif_addr from the add ALU and from the mul ALU."""
    src = PROLOGUE + """
    mov ra0, unif            # out
    mov ra1, unif            # address of this uniform list
    mov ra2, unif            # magic 0x1234
    mov ra3, unif            # magic 0x5678
    ldi vw_setup, vpm_setup_h32(4, 1, 0)
    add r0, ra1, 8
    mov unif_addr, r0        # add ALU write
    nop
    nop
    mov vpm, unif            # expect 0x1234
    add r0, ra1, 12
    nop; mov unif_addr, r0   # mul ALU write
    nop
    nop
    mov vpm, unif            # expect 0x5678
    add r0, ra1, 8
    nop; mov unif_addr, r0   # mul ALU write
    nop
    nop
    nop
    nop
    mov vpm, unif            # expect 0x1234 (longer wait)
    add r0, ra1, 12
    mov unif_addr, r0
    nop
    nop
    mov vpm, unif            # expect 0x5678
""" + store_rows(4) + EPILOGUE
    expect = np.repeat(np.array([0x1234, 0x5678, 0x1234, 0x5678], dtype=np.uint32), 16)
    return src, [Buffer("out", 256)], [[("buf", "out", 0), ("self",), 0x1234, 0x5678]], ["out"], expect




@probe
def unif_reset_paired():
    """unif_addr written by the mul ALU while the add ALU writes a register."""
    src = PROLOGUE + """
    mov ra0, unif            # out
    mov ra1, unif            # address of this uniform list
    mov ra2, unif
    mov ra3, unif
    ldi vw_setup, vpm_setup_h32(5, 1, 0)
    mov r0, 1
    sub.setf -, r0, 0        # N clear
    add r0, ra1, 8
    add ra5, ra2, 1; mov unif_addr, r0       # V1: unconditional add + mul unif_addr
    nop
    nop
    mov vpm, unif
    add r0, ra1, 12
    add.ifnn ra5, ra2, 1; mov unif_addr, r0  # V2: conditional add + mul unif_addr
    nop
    nop
    mov vpm, unif
    add r0, ra1, 8
    add.ifnn ra5, ra2, 1; mov unif_addr, r0  # V3: same, 3 instructions gap
    nop
    nop
    nop
    mov vpm, unif
    add r0, ra1, 12
    mov unif_addr, r0; fmul ra5, r1, r1      # V4: add ALU unif_addr + mul regfile write
    nop
    nop
    mov vpm, unif
    add r0, ra1, 8
    nop; mov unif_addr, r0                   # V5: control
    nop
    nop
    mov vpm, unif
""" + store_rows(5) + EPILOGUE
    expect = np.repeat(np.array([0x1234, 0x5678, 0x1234, 0x5678, 0x1234], dtype=np.uint32), 16)
    return src, [Buffer("out", 512)], [[("buf", "out", 0), ("self",), 0x1234, 0x5678]], ["out"], expect



def vdw_concurrency(use_mutex: bool, nprog: int = 48, chunks: int = 8):
    """Many QPUs storing 4-row chunks at once, with or without the mutex.
    Returns the number of corrupted 64-byte rows."""
    lock = "    mov -, mutex" if use_mutex else "    nop"
    unlock = "    mov mutex, 0" if use_mutex else "    nop"
    body = []
    for c in range(chunks):
        body.append(f"""
{lock}
    mov r0, qpu_num
    nop
    shl r1, r0, 2
    ldi r2, vpm_setup_h32(4, 1, 0)
    add vw_setup, r1, r2
    add r3, ra1, {c}
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    shl r1, r1, 7
    ldi r2, vdw_h32(4, 16, 0, 0)
    add vw_setup, r1, r2
    ldi r2, {c * 256}
    add vw_addr, ra2, r2
    mov -, vw_wait
{unlock}""")
    src = PROLOGUE + """
    mov ra0, unif            # out base
    mov ra1, unif            # pattern base = pid * 1000
    mov ra2, unif            # this program's output address
    nop
""" + "".join(body) + EPILOGUE
    words, _ = vc4asm_assemble(src, verify=False)
    row_bytes = 64
    per_prog = chunks * 4 * row_bytes
    progs = [[("buf", "out", 0), p * 1000, ("buf", "out", p * per_prog)] for p in range(nprog)]
    res = PiRunner(timeout=60).run(words, [Buffer("out", per_prog * nprog)], progs, ["out"])
    got = res.outputs["out"].reshape(nprog, chunks, 4, 16)
    exp = np.zeros_like(got)
    for p in range(nprog):
        for c in range(chunks):
            for r in range(4):
                exp[p, c, r, :] = p * 1000 + c + r
    bad = int(np.sum(np.any(got != exp, axis=-1)))
    return bad, nprog * chunks * 4


def vdw_concurrency_probe():
    for m in (False, True):
        bad, total = vdw_concurrency(m)
        print(f"[vdw_concurrency] mutex={m}: {bad}/{total} rows corrupted")



def vdw_stride_concurrency(use_mutex: bool, nprog: int = 48, chunks: int = 8, stride_once: bool = False,
                           early_unlock: bool = False):
    """Like vdw_concurrency, but every program uses its own VDW row stride
    (0, 64 or 128 bytes): tests whether the stride setup is per QPU."""
    lock = "    mov -, mutex" if use_mutex else "    nop"
    unlock = "    mov mutex, 0" if use_mutex else "    nop"
    body = []
    for c in range(chunks):
        body.append(f"""
{lock}
    mov r0, qpu_num
    nop
    shl r1, r0, 2
    ldi r2, vpm_setup_h32(4, 1, 0)
    add vw_setup, r1, r2
    add r3, ra1, {c}
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    add r3, r3, 1
    mov vpm, r3
    shl r1, r1, 7
    ldi r2, vdw_h32(4, 16, 0, 0)
    add vw_setup, r1, r2
{"    nop" if (stride_once and c > 0) else "    mov vw_setup, ra3        # stride setup (per program)"}
    mov r2, ra4              # chunk pitch = 4 * (64 + stride)
    nop
    mul24 r2, r2, {c}
    add vw_addr, ra2, r2
{unlock if early_unlock else "    nop"}
    mov -, vw_wait
{"    nop" if early_unlock else unlock}""")
    src = PROLOGUE + """
    mov ra0, unif            # out base
    mov ra1, unif            # pattern base = pid * 1000
    mov ra2, unif            # this program's output address
    mov ra3, unif            # vdw stride setup word
    mov ra4, unif            # bytes per chunk
    nop
""" + "".join(body) + EPILOGUE
    words, _ = vc4asm_assemble(src, verify=False)
    strides = [((p % 3) * 64) for p in range(nprog)]
    per_prog = chunks * 4 * (64 + 128)
    progs = [[("buf", "out", 0), p * 1000, ("buf", "out", p * per_prog), 0xC0000000 | strides[p],
              4 * (64 + strides[p])] for p in range(nprog)]
    res = PiRunner(timeout=60).run(words, [Buffer("out", per_prog * nprog)], progs, ["out"])
    words_out = res.outputs["out"]
    bad = 0
    total = 0
    for p in range(nprog):
        base = p * per_prog // 4
        pitch = (64 + strides[p]) // 4
        for c in range(chunks):
            for r in range(4):
                off = base + (c * 4 + r) * pitch
                total += 1
                if not np.all(words_out[off:off + 16] == p * 1000 + c + r):
                    bad += 1
    return bad, total


def vdw_stride_probe():
    for once in (False, True):
        for m in (False, True):
            bad, total = vdw_stride_concurrency(m, stride_once=once)
            print(f"[vdw_stride_concurrency] stride written {'once' if once else 'every chunk'}, "
                  f"mutex={m}: {bad}/{total} rows wrong")

if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in PROBES if n != "multi_qpu"]
    for n in names:
        if n == "vdw_concurrency":
            vdw_concurrency_probe()
        elif n == "vdw_stride":
            vdw_stride_probe()
        elif n == "timings":
            timings()
        else:
            run(n)
