"""A functional + timing-approximate emulator for the VideoCore IV V3D.

It executes real 64-bit QPU machine code (from our encoder or from vc4asm)
and models the parts of the chip a general-purpose kernel touches:

  * 16-lane registers: regfile A/B, accumulators r0-r5, per-lane N/Z/C flags
  * add/mul ALUs, small immediates, mul-output vector rotation
  * uniforms stream (+ unif_addr rewrites)
  * TMU direct-address gathers (per-QPU request FIFO, results in r4)
  * SFU recip / recipsqrt / exp2 / log2 (results in r4 three instrs later)
  * VPM generic block reads/writes, VDR DMA loads and VDW DMA stores
  * semaphores, the mutex, thread end, branches with three delay slots
  * the host-side user-program request queue (V3D_SRQPC / V3D_SRQUA)

In `strict` mode the emulator raises `HazardError` for code that the
reference guide says is undefined (p37 "Summary of Instruction
Restrictions"), e.g. reading a regfile location written by the previous
instruction, touching r4 too early after an SFU write, rotating an
accumulator written by the previous instruction, overflowing the TMU FIFO,
or starting a DMA while the previous one is still running.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import isa
from .isa import Imm, Reg

LANES = 16
MASK32 = 0xFFFFFFFF


class HazardError(Exception):
    pass


class EmulatorError(Exception):
    pass


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory:
    """Flat little-endian memory addressed with VideoCore bus addresses."""

    def __init__(self, size: int = 32 << 20, base: int = 0x40100000):
        self.base = base
        self.size = size
        self.words = np.zeros(size // 4, dtype=np.uint32)
        self._brk = base

    def alloc(self, nbytes: int, align: int = 4096) -> int:
        addr = (self._brk + align - 1) & ~(align - 1)
        if addr + nbytes > self.base + self.size:
            raise MemoryError("emulator memory exhausted")
        self._brk = addr + nbytes
        return addr

    def _index(self, addr):
        idx = (np.asarray(addr, dtype=np.int64) - self.base) >> 2
        if np.any(idx < 0) or np.any(idx >= len(self.words)):
            bad = np.asarray(addr).ravel()[0] if np.ndim(addr) else addr
            raise EmulatorError(f"memory access out of range: 0x{int(bad) & MASK32:08x}")
        return idx

    def read_words(self, addrs) -> np.ndarray:
        return self.words[self._index(addrs)]

    def write_words(self, addr: int, values) -> None:
        idx = self._index(addr)
        self.words[idx:idx + len(values)] = values

    def write_array(self, addr: int, arr: np.ndarray) -> None:
        flat = np.ascontiguousarray(arr).view(np.uint32).ravel()
        self.write_words(addr, flat)

    def read_array(self, addr: int, shape, dtype) -> np.ndarray:
        n = int(np.prod(shape))
        idx = self._index(addr)
        return self.words[idx:idx + n].copy().view(dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Decoded instruction cache
# ---------------------------------------------------------------------------


def _f(x: np.ndarray) -> np.ndarray:
    return x.view(np.float32)


def _u(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _sat_u8(x):
    return np.clip(x, 0, 255).astype(np.uint32)


def _v8(a, b, fn):
    out = np.zeros(LANES, dtype=np.uint32)
    for byte in range(4):
        sa = (a >> (8 * byte)) & 0xFF
        sb = (b >> (8 * byte)) & 0xFF
        out |= (fn(sa.astype(np.int64), sb.astype(np.int64)).astype(np.uint32) & 0xFF) << (8 * byte)
    return out


# Float ALU semantics measured on a Pi Zero (tools/float_specials_probe.py,
# tools/probes.py): results are the exact value truncated toward zero; a
# result of 2**128 or more is inf, one below 2**-126 is +0, and zero results
# are always +0; denormal operands read as +0 and NaN operands as infinities
# of their sign; 0 * inf is +0 and inf + -inf is the second operand; fmin /
# fmax order operands by their sign-magnitude bits (so +NaN is the largest
# value, -0 equals +0, denormals count as positive) and return the operand
# itself.  Not modelled: ftoi of values >= 2**31 in magnitude (undefined in
# Triton; the hardware mostly wraps modulo 2**32, but not always).

_FLT_MIN = 2.0 ** -126
_TWO128 = 2.0 ** 128


def _rtz(exact: np.ndarray, err: Optional[np.ndarray] = None) -> np.ndarray:
    """float64 -> float32 as the QPU rounds.  `err` is the part of the exact
    value lost in `exact` (from TwoSum), so sums are truncated exactly."""
    with np.errstate(all="ignore"):
        rn = exact.astype(np.float32)
        bigger = np.abs(rn.astype(np.float64)) > np.abs(exact)
        out = np.where(bigger & np.isfinite(exact), np.nextafter(rn, np.float32(0)), rn)
        if err is not None:
            # exact == out but the true value is a little closer to zero
            below = (out.astype(np.float64) == exact) & (err != 0) & (np.sign(err) != np.sign(exact))
            out = np.where(below, np.nextafter(out, np.float32(0)), out)
        out = np.where(np.abs(exact) >= _TWO128, np.copysign(np.float32(np.inf), exact), out)
        out = np.where(np.abs(out) < np.float32(_FLT_MIN), np.float32(0), out)   # also -0 -> +0
    return out.astype(np.float32)


def _fp(x):
    return _f(x).astype(np.float64)


def _qpu_in(x: np.ndarray) -> np.ndarray:
    """Operand value as the float ALUs read it (float64)."""
    e = (x >> 23) & 0xFF
    f = _fp(x)
    f = np.where((e == 0) & ((x & 0x7FFFFF) != 0), 0.0, f)
    return np.where(e == 255, np.where(x >> 31 != 0, -np.inf, np.inf), f)


def _qpu_fadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa, fb = _qpu_in(a), _qpu_in(b)
    with np.errstate(all="ignore"):
        s = fa + fb
        bb = s - fa
        err = (fa - (s - bb)) + (fb - bb)
        out = _rtz(s, err)
        out = np.where(np.isinf(fb), fb.astype(np.float32), out)
        out = np.where(np.isinf(fa) & ~np.isinf(fb), fa.astype(np.float32), out)
    return out


def _qpu_fmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa, fb = _qpu_in(a), _qpu_in(b)
    with np.errstate(all="ignore"):
        out = _rtz(fa * fb)
        zero_inf = ((fa == 0) & np.isinf(fb)) | (np.isinf(fa) & (fb == 0))
        return np.where(zero_inf, np.float32(0), out)


def _fkey(x: np.ndarray) -> np.ndarray:
    """fmin / fmax ordering key: sign-magnitude bits; denormals (and -0) count
    as positive."""
    m = (x & 0x7FFFFFFF).astype(np.int64)
    return np.where((x >> 31 != 0) & ((x >> 23) & 0xFF != 0), -m, m)


def _qpu_ftoi(a: np.ndarray) -> np.ndarray:
    """Truncate, then keep the low 32 bits (so inf, NaN and huge values give 0)."""
    e = (a >> 23) & 0xFF
    f = np.where((e == 0) | (e >= 127 + 55), 0.0, _fp(a))
    return (np.trunc(f).astype(np.int64) & MASK32).astype(np.uint32)


def alu_add(op: str, a: np.ndarray, b: np.ndarray):
    """Returns (result uint32[16], is_float, carry bool[16] or None)."""
    with np.errstate(all="ignore"):
        if op == "fadd":
            return _u(_qpu_fadd(a, b)), True, None
        if op == "fsub":
            return _u(_qpu_fadd(a, b ^ np.uint32(0x80000000))), True, None
        if op == "fmin":
            return np.where(_fkey(a) <= _fkey(b), a, b).astype(np.uint32), True, None
        if op == "fmax":
            return np.where(_fkey(a) > _fkey(b), a, b).astype(np.uint32), True, None
        if op == "fminabs":
            ma, mb = a & np.uint32(0x7FFFFFFF), b & np.uint32(0x7FFFFFFF)
            return np.where(ma <= mb, ma, mb).astype(np.uint32), True, None
        if op == "fmaxabs":
            ma, mb = a & np.uint32(0x7FFFFFFF), b & np.uint32(0x7FFFFFFF)
            return np.where(ma > mb, ma, mb).astype(np.uint32), True, None
        if op == "ftoi":
            return _qpu_ftoi(a), False, None
        if op == "itof":
            return _u(_rtz(a.view(np.int32).astype(np.float64))), True, None
        ai, bi = a.astype(np.int64), b.astype(np.int64)
        sa, sb = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
        sh = (b & 31).astype(np.uint32)
        if op == "add":
            s = ai + bi
            return (s & MASK32).astype(np.uint32), False, s > MASK32
        if op == "sub":
            s = ai - bi
            return (s & MASK32).astype(np.uint32), False, ai < bi
        if op == "shr":
            return (a >> sh).astype(np.uint32), False, None
        if op == "asr":
            return (a.view(np.int32) >> sh.astype(np.int32)).view(np.uint32), False, None
        if op == "ror":
            return ((a >> sh) | (a << ((32 - sh) & 31))).astype(np.uint32) & MASK32, False, None
        if op == "shl":
            return (a << sh).astype(np.uint32), False, None
        if op == "min":
            return (np.minimum(sa, sb) & MASK32).astype(np.uint32), False, None
        if op == "max":
            return (np.maximum(sa, sb) & MASK32).astype(np.uint32), False, None
        if op in ("and", "mov"):
            return (a & b) if op == "and" else a.copy(), False, None
        if op == "or":
            return a | b, False, None
        if op == "xor":
            return a ^ b, False, None
        if op == "not":
            return ~a, False, None
        if op == "clz":
            out = np.array([32 - int(x).bit_length() for x in a], dtype=np.uint32)
            return out, False, None
        if op == "v8adds":
            return _v8(a, b, lambda x, y: np.clip(x + y, 0, 255)), False, None
        if op == "v8subs":
            return _v8(a, b, lambda x, y: np.clip(x - y, 0, 255)), False, None
    raise EmulatorError(f"unsupported add op {op}")


def alu_mul(op: str, a: np.ndarray, b: np.ndarray):
    with np.errstate(all="ignore"):
        if op == "fmul":
            return _u(_qpu_fmul(a, b)), True
        if op == "mul24":
            prod = (a & 0xFFFFFF).astype(np.uint64) * (b & 0xFFFFFF).astype(np.uint64)
            return (prod & MASK32).astype(np.uint32), False
        if op in ("v8min", "mov"):
            return _v8(a, b, np.minimum), False
        if op == "v8max":
            return _v8(a, b, np.maximum), False
        if op == "v8adds":
            return _v8(a, b, lambda x, y: np.clip(x + y, 0, 255)), False
        if op == "v8subs":
            return _v8(a, b, lambda x, y: np.clip(x - y, 0, 255)), False
        if op == "v8muld":
            return _v8(a, b, lambda x, y: (x * y + 127) // 255), False
    raise EmulatorError(f"unsupported mul op {op}")


def _cond_mask(cond: str, flags) -> np.ndarray:
    n, z, c = flags
    if cond == "always":
        return np.ones(LANES, bool)
    if cond == "never":
        return np.zeros(LANES, bool)
    return {"ifz": z, "ifnz": ~z, "ifn": n, "ifnn": ~n, "ifc": c, "ifnc": ~c}[cond]


def _branch_taken(cond: str, flags) -> bool:
    n, z, c = flags
    if cond == "always":
        return True
    quant, rest = ("all", cond[3:]) if cond.startswith("all") else ("any", cond[3:])
    vec = {"z": z, "nz": ~z, "n": n, "nn": ~n, "c": c, "nc": ~c}[rest]
    return bool(vec.all()) if quant == "all" else bool(vec.any())


# ---------------------------------------------------------------------------
# VPM / VCD state (shared by all QPUs)
# ---------------------------------------------------------------------------


@dataclass
class VPMRead:
    num: int
    stride: int
    horiz: bool
    addr: int


class VPMState:
    def __init__(self, rows: int = 64):
        self.mem = np.zeros((rows, 16), dtype=np.uint32)
        self.rows = rows

    def read_vec(self, horiz: bool, addr: int) -> np.ndarray:
        if horiz:
            return self.mem[addr & 63].copy()
        y0, x = (addr >> 4) & 3, addr & 15
        return self.mem[16 * y0:16 * y0 + 16, x].copy()

    def write_vec(self, horiz: bool, addr: int, vec: np.ndarray) -> None:
        if horiz:
            self.mem[addr & 63] = vec
        else:
            y0, x = (addr >> 4) & 3, addr & 15
            self.mem[16 * y0:16 * y0 + 16, x] = vec


# ---------------------------------------------------------------------------
# One QPU
# ---------------------------------------------------------------------------


@dataclass
class TimingModel:
    """Very rough latencies in QPU instruction cycles (1 cycle = 4 clocks)."""
    tmu_latency: int = 24          # cycles from t*s write until ldtmu can complete
    vdw_cycles_per_row: int = 6    # DMA store cost per memory row
    vdw_base: int = 12
    vdr_cycles_per_row: int = 6
    vdr_base: int = 12


class QPU:
    def __init__(self, num: int, v3d: "V3D"):
        self.num = num
        self.v3d = v3d
        self.reset()

    def reset(self):
        z = lambda: np.zeros(LANES, dtype=np.uint32)
        self.ra = [z() for _ in range(32)]
        self.rb = [z() for _ in range(32)]
        self.acc = [z() for _ in range(6)]
        self.flags = (np.zeros(LANES, bool), np.zeros(LANES, bool), np.zeros(LANES, bool))
        self.pc = 0
        self.running = False
        self.end_countdown = None
        self.branch_target = None
        self.branch_countdown = None
        self.unif_ptr = 0
        self.unif_block_until = -1
        self.tmu = [deque(), deque()]
        self.r4_pending = None          # (ready_cycle, value) for SFU
        self.sfu_busy_until = -1
        self.last_written = set()       # ('A'|'B', addr) written by previous instruction
        self.last_acc_written = set()
        self.last_r5_written = False
        self.vpm_reads = deque()
        self.vpm_read_ready = -1
        self.vpm_write = (True, 0, 1)   # horiz, addr, stride
        self.vdw = {"units": 1, "depth": 16, "horiz": True, "vpmbase": 0, "stride": 0, "blockmode": 0}
        self.vdr = {"mpitch": 0, "rowlen": 16, "nrows": 1, "vpitch": 1, "vert": False, "addr": 0,
                    "pitch_ext": 0}
        self.vdw_busy_until = None
        self.vdr_busy_until = None
        self.cycle = 0
        self.instructions = 0
        self.stall_cycles = 0
        self.blocked = None
        self.stats = {"tmu_loads": 0, "vdw_bytes": 0, "vdr_bytes": 0, "fmul": 0, "fadd": 0}
        self.program = None

    # -- helpers -----------------------------------------------------------

    def hazard(self, msg: str):
        if self.v3d.strict:
            raise HazardError(f"QPU{self.num} pc=0x{self.pc * 8:x} "
                              f"({isa.format_instr(self.program[self.pc][0])}): {msg}")

    def start(self, program, entry: int, unif_addr: int):
        self.reset()
        self.program = program
        self.pc = entry
        self.unif_ptr = unif_addr
        self.running = True

    # -- operand access ----------------------------------------------------

    def read(self, src, written_now) -> np.ndarray:
        if isinstance(src, Imm):
            return np.full(LANES, src.value, dtype=np.uint32)
        if src.acc is not None:
            if src.acc == 4:
                if self.r4_pending is not None and self.cycle < self.r4_pending[0]:
                    self.hazard("r4 read before SFU result is ready")
                return self.acc[4]
            return self.acc[src.acc]
        return self.read_regfile(src, written_now)

    def read_regfile(self, src: Reg, file: str) -> np.ndarray:
        addr = src.raddr
        if addr < 32:
            if (file, addr) in self.last_written:
                self.hazard(f"read of {src.name} written by previous instruction")
            return (self.ra if file == "A" else self.rb)[addr]
        if addr == 32:
            if self.cycle < self.unif_block_until:
                self.hazard("uniform read too soon after unif_addr write")
            v = self.v3d.mem.read_words(np.array([self.unif_ptr]))[0]
            self.unif_ptr += 4
            return np.full(LANES, v, dtype=np.uint32)
        if addr == 38:
            if file == "A":
                return np.arange(LANES, dtype=np.uint32)
            return np.full(LANES, self.num, dtype=np.uint32)
        if addr == 39:
            return np.zeros(LANES, dtype=np.uint32)
        if addr == 48:
            return self.vpm_read()
        if addr == 49:
            busy = self.vdr_busy_until if file == "A" else self.vdw_busy_until
            return np.full(LANES, int(busy is not None and busy > self.cycle), dtype=np.uint32)
        if addr == 50:
            if file == "A":
                self.wait_vdr()
            else:
                self.wait_vdw()
            return np.zeros(LANES, dtype=np.uint32)
        if addr == 51:
            if not self.v3d.mutex_acquire(self):
                raise _Blocked("mutex")
            return np.arange(LANES, dtype=np.uint32)
        raise EmulatorError(f"unsupported read address {file}{addr}")

    # -- VPM ---------------------------------------------------------------

    def vpm_read(self) -> np.ndarray:
        if not self.vpm_reads:
            self.hazard("VPM read without a pending read setup")
            return np.zeros(LANES, dtype=np.uint32)
        if self.cycle < self.vpm_read_ready:
            self.hazard("VPM read less than 3 instructions after read setup")
        r = self.vpm_reads[0]
        vec = self.v3d.vpm.read_vec(r.horiz, r.addr)
        r.addr = (r.addr + r.stride) & 0xFF
        r.num -= 1
        if r.num == 0:
            self.vpm_reads.popleft()
        return vec

    def check_shared_vpm(self, what):
        """The VDW stride register is shared by all QPUs (tools/probes.py
        vdw_stride), so with more than one QPU running, writing it and
        starting a DMA must happen under the mutex."""
        if self.v3d.require_vpm_mutex and self.v3d.active_qpus() > 1 and self.v3d.mutex_owner != self.num:
            self.hazard(f"{what} without holding the mutex while other QPUs run")

    def wait_vdw(self):
        if self.vdw_busy_until is not None:
            if self.vdw_busy_until > self.cycle:
                self.stall_cycles += self.vdw_busy_until - self.cycle
                self.cycle = self.vdw_busy_until
            self.vdw_busy_until = None

    def wait_vdr(self):
        if self.vdr_busy_until is not None:
            if self.vdr_busy_until > self.cycle:
                self.stall_cycles += self.vdr_busy_until - self.cycle
                self.cycle = self.vdr_busy_until
            self.vdr_busy_until = None

    def write_vr_setup(self, v: int):
        if v >> 31:                          # VDR DMA load setup
            modew = (v >> 28) & 7
            if modew == 1:                   # extended memory pitch
                self.vdr["pitch_ext"] = v & 0x1FFF
                return
            if modew != 0:
                raise EmulatorError("only 32-bit VDR DMA is modelled")
            if self.vdr_busy_until is not None:
                self.hazard("VDR setup while a DMA load is still running")
            self.vdr.update(mpitch=(v >> 24) & 15, rowlen=((v >> 20) & 15) or 16,
                            nrows=((v >> 16) & 15) or 16, vpitch=((v >> 12) & 15) or 16,
                            vert=bool((v >> 11) & 1), addr=v & 0x7FF)
            return
        if (v >> 30) != 0:
            raise EmulatorError(f"unknown vr_setup value 0x{v:08x}")
        if ((v >> 8) & 3) != 2:
            raise EmulatorError("only 32-bit VPM reads are modelled")
        if len(self.vpm_reads) >= 2:
            self.hazard("more than two VPM read setups queued (setup ignored)")
            return
        self.vpm_reads.append(VPMRead(num=((v >> 20) & 15) or 16, stride=((v >> 12) & 63) or 64,
                                      horiz=bool((v >> 11) & 1), addr=v & 0xFF))
        self.vpm_read_ready = self.cycle + 3

    def write_vw_setup(self, v: int):
        top = v >> 30
        if top == 0:                         # VPM generic block write
            if ((v >> 8) & 3) != 2:
                raise EmulatorError("only 32-bit VPM writes are modelled")
            self.vpm_write = (bool((v >> 11) & 1), v & 0xFF, ((v >> 12) & 63) or 64)
        elif top == 2:                       # VDW basic setup
            if self.vdw_busy_until is not None:
                self.hazard("VDW setup while a DMA store is still running")
            if v & 7:
                raise EmulatorError("only 32-bit VDW DMA is modelled")
            self.vdw.update(units=((v >> 23) & 127) or 128, depth=((v >> 16) & 127) or 128,
                            horiz=bool((v >> 14) & 1), vpmbase=(v >> 3) & 0x7FF)
        elif top == 3:                       # VDW stride setup: one register for all QPUs
            self.v3d.vdw_stride = {"stride": v & 0xFFFF, "blockmode": (v >> 16) & 1}
        else:
            raise EmulatorError(f"unsupported vw_setup value 0x{v:08x}")

    def write_vpm(self, vec: np.ndarray):
        if self.vdw_busy_until is not None and self.cycle < self.vdw_busy_until:
            # the DMA reads VPM while it runs (here it copied at start)
            self.hazard("VPM written while this QPU's VDW DMA may still be reading it")
        horiz, addr, stride = self.vpm_write
        self.v3d.vpm.write_vec(horiz, addr, vec)
        self.vpm_write = (horiz, (addr + stride) & 0xFF, stride)

    def start_vdw(self, addr: int):
        if self.vdw_busy_until is not None:
            self.hazard("VDW started while the previous DMA store is still running")
        d = dict(self.vdw, **self.v3d.vdw_stride)
        y, x = (d["vpmbase"] >> 4) & 127, d["vpmbase"] & 15
        vpm = self.v3d.vpm.mem
        for u in range(d["units"]):
            if d["horiz"]:
                # Measured on hardware (tools/probes.py vdw_long_rows): a
                # horizontal memory row always comes from one VPM row and the
                # X address wraps inside it, whatever BLOCKMODE says.
                if x + d["depth"] > 16:
                    self.hazard("VDW horizontal row longer than a VPM row (wraps on hardware)")
                row = vpm[(y + u) % self.v3d.vpm.rows, (x + np.arange(d["depth"])) % 16]
            else:
                rows = [(y + i) % self.v3d.vpm.rows for i in range(d["depth"])]
                row = vpm[rows, (x + u) & 15]
            self.v3d.mem.write_words(addr, row.astype(np.uint32))
            addr += 4 * d["depth"] + d["stride"]
        nbytes = 4 * d["units"] * d["depth"]
        self.stats["vdw_bytes"] += nbytes
        t = self.v3d.timing
        self.vdw_busy_until = self.cycle + t.vdw_base + t.vdw_cycles_per_row * d["units"]

    def start_vdr(self, addr: int):
        if self.vdr_busy_until is not None:
            self.hazard("VDR started while the previous DMA load is still running")
        d = self.vdr
        pitch = d["pitch_ext"] if d["mpitch"] == 0 else 8 << d["mpitch"]
        y, x = (d["addr"] >> 4) & 63, d["addr"] & 15
        vpm = self.v3d.vpm.mem
        for r in range(d["nrows"]):
            words = self.v3d.mem.read_words(addr + 4 * np.arange(d["rowlen"]))
            if d["vert"]:
                vpm[y:y + d["rowlen"], (x + r * d["vpitch"]) & 15] = words
            else:
                yy = (y + r * d["vpitch"]) % self.v3d.vpm.rows
                if x + d["rowlen"] > 16:
                    raise EmulatorError("VDR horizontal row crosses a VPM row")
                vpm[yy, x:x + d["rowlen"]] = words
            addr += pitch
        self.stats["vdr_bytes"] += 4 * d["nrows"] * d["rowlen"]
        t = self.v3d.timing
        self.vdr_busy_until = self.cycle + t.vdr_base + t.vdr_cycles_per_row * d["nrows"]

    # -- writes ------------------------------------------------------------

    def write(self, dst: Reg, file: str, value: np.ndarray, mask: np.ndarray, writes: list):
        """Queue a write; applied at the end of the instruction."""
        writes.append((dst, file, value, mask))

    def commit(self, dst: Reg, file: str, value: np.ndarray, mask: np.ndarray):
        if dst is _R4_SINK:
            self.acc[4] = value
            return
        w = dst.waddr
        if w < 32:
            regs = self.ra if file == "A" else self.rb
            regs[w] = np.where(mask, value, regs[w]).astype(np.uint32)
            self.last_written_now.add((file, w))
            return
        if 32 <= w <= 35:
            self.acc[w - 32] = np.where(mask, value, self.acc[w - 32]).astype(np.uint32)
            self.last_acc_now.add(w - 32)
            return
        if w == 39:
            return
        if not mask.any():
            return
        lane0 = int(value[np.argmax(mask)]) if not mask[0] else int(value[0])
        if w == 37:
            if file == "A":   # r5quad: replicate element 0 of each quad
                rep = np.repeat(value[0::4], 4)
            else:             # r5rep: replicate element 0
                rep = np.full(LANES, value[0], dtype=np.uint32)
            self.acc[5] = np.where(mask, rep, self.acc[5]).astype(np.uint32)
            self.last_r5_now = True
            return
        if w == 36:           # tmu_noswap
            return
        if w == 38:           # host interrupt
            self.v3d.host_interrupts += 1
            return
        if w == 40:
            if file == "B":
                raise EmulatorError("write to unif_addr_rel (B-space address 40) is not modelled")
            self.unif_ptr = int(value[0])
            self.unif_block_until = self.cycle + 3
            return
        if w == 48:
            self.write_vpm(value)
            return
        if w == 49:
            if file == "B" and (int(value[0]) >> 30) == 3:
                self.check_shared_vpm("VDW stride setup")
            (self.write_vr_setup if file == "A" else self.write_vw_setup)(int(value[0]))
            return
        if w == 50:
            if file == "B":
                self.check_shared_vpm("vw_addr")
            (self.start_vdr if file == "A" else self.start_vdw)(int(value[0]))
            return
        if w == 51:
            self.v3d.mutex_release(self)
            return
        if 52 <= w <= 55:
            x = _f(value).astype(np.float64)
            with np.errstate(all="ignore"):
                if w == 52:
                    r = 1.0 / x
                elif w == 53:
                    r = 1.0 / np.sqrt(x)
                elif w == 54:
                    r = np.exp2(x)
                else:
                    r = np.log2(x)
            self.r4_pending = (self.cycle + 3, _u(r.astype(np.float32)))
            return
        if w in (56, 60):
            unit = 0 if w == 56 else 1
            if len(self.tmu[unit]) >= self.v3d.tmu_fifo_depth:
                self.hazard(f"TMU{unit} request FIFO overflow (> {self.v3d.tmu_fifo_depth})")
            addrs = (value & ~np.uint32(3)).astype(np.int64)
            data = self.v3d.mem.read_words(addrs)
            self.tmu[unit].append((self.cycle + self.v3d.timing.tmu_latency, data))
            self.stats["tmu_loads"] += 1
            return
        raise EmulatorError(f"unsupported write to {dst.name}")

    # -- execution ---------------------------------------------------------

    def step(self):
        program = self.program
        if not 0 <= self.pc < len(program):
            raise EmulatorError(f"QPU{self.num} jumped outside the program (pc={self.pc})")
        ins, extra = program[self.pc]
        self.last_written_now = set()
        self.last_acc_now = set()
        self.last_r5_now = False

        if self.r4_pending is not None and self.cycle >= self.r4_pending[0]:
            self.acc[4] = self.r4_pending[1]
            self.r4_pending = None

        in_delay_slot = self.branch_countdown is not None
        in_end_slot = self.end_countdown is not None
        writes = []
        taken_target = None
        if ins.kind == "alu":
            self.exec_alu(ins, writes)
        elif ins.kind == "ldi":
            self.exec_ldi(ins, writes)
        elif ins.kind == "sem":
            if ins.acquire:
                if not self.v3d.sem_acquire(ins.sema):
                    raise _Blocked("semaphore")
            else:
                self.v3d.sem_release(ins.sema)
            self.exec_ldi(ins, writes)
        elif ins.kind == "branch":
            if in_delay_slot:
                self.hazard("branch in a branch delay slot")
            if _branch_taken(ins.bcond, self.flags):
                target = extra
                if ins.breg is not None:
                    if ("A", ins.breg.raddr) in self.last_written:
                        self.hazard("branch register written by previous instruction")
                    byte = (ins.imm + (8 * (self.pc + 4) if ins.rel else 0)
                            + int(self.ra[ins.breg.raddr][0])) & MASK32
                    target = byte // 8
                taken_target = target
                link = np.full(LANES, 8 * (self.pc + 4), dtype=np.uint32)
                for op, natural in ((ins.add, "A"), (ins.mul, "B")):
                    if op is not None and op.dst != isa.NOP:
                        self.write(op.dst, self._dst_file(ins, op, natural), link,
                                   np.ones(LANES, bool), writes)

        for dst, file, value, mask in writes:
            self.commit(dst, file, value, mask)

        self.last_written = self.last_written_now
        self.last_acc_written = self.last_acc_now
        self.last_r5_written = self.last_r5_now
        self.instructions += 1
        self.cycle += 1

        if in_end_slot:
            self.end_countdown -= 1
            if self.end_countdown == 0:
                self.finish()
                return
        if ins.kind == "alu" and ins.sig == "thrend":
            if in_end_slot:
                self.hazard("thrend in a thread-end delay slot")
            self.end_countdown = 2
        next_pc = self.pc + 1
        if in_delay_slot:
            self.branch_countdown -= 1
            if self.branch_countdown == 0:
                next_pc = self.branch_target
                self.branch_countdown = self.branch_target = None
        if taken_target is not None:
            self.branch_target = taken_target
            self.branch_countdown = 3
        self.pc = next_pc

    def finish(self):
        if any(self.tmu):
            self.hazard("program ended with TMU results still queued")
        if self.vpm_reads:
            self.hazard("program ended with VPM reads still pending")
        self.running = False
        self.v3d.on_finish(self)

    def exec_ldi(self, ins, writes):
        imm = ins.imm & MASK32
        if ins.ldi_mode == 0 or ins.kind == "sem":
            value = np.full(LANES, imm, dtype=np.uint32)
        else:
            ms = np.array([(imm >> (16 + i)) & 1 for i in range(LANES)])
            ls = np.array([(imm >> i) & 1 for i in range(LANES)])
            v = (ms << 1) | ls
            if ins.ldi_mode == 1:
                v = np.where(ms == 1, v - 4, v)
            value = (v.astype(np.int64) & MASK32).astype(np.uint32)
        for op, natural in ((ins.add, "A"), (ins.mul, "B")):
            if op is None:
                continue
            self.write(op.dst, self._dst_file(ins, op, natural), value,
                       _cond_mask(op.cond, self.flags), writes)
        if ins.setf:
            self._set_flags(value, False, None)

    @staticmethod
    def _ws(ins) -> int:
        if ins.add is not None and ins.add.dst.wfile in ("A", "B"):
            return 0 if ins.add.dst.wfile == "A" else 1
        if ins.mul is not None and ins.mul.dst.wfile in ("A", "B"):
            return 0 if ins.mul.dst.wfile == "B" else 1
        return 0

    def _dst_file(self, ins, op, natural):
        """Write space ('A'/'B') that `op` (the add or mul op) writes into."""
        ws = self._ws(ins)
        if natural == "A":
            return "A" if ws == 0 else "B"
        return "B" if ws == 0 else "A"

    def _set_flags(self, value, is_float, carry):
        z = (_f(value) == 0) if is_float else (value == 0)
        n = (value >> 31).astype(bool)
        c = carry if carry is not None else np.zeros(LANES, bool)
        self.flags = (n, z, np.asarray(c, bool))

    def exec_alu(self, ins, writes):
        cache = {}

        def self_cancel(op):
            # vc4asm's `mov x, 0` is `xor x, s, s` / `v8subs x, s, s`: s is not really read
            return op is not None and op.op in ("xor", "v8subs", "sub") and op.a == op.b

        # registers read *only* by self-cancelling ops: their value is irrelevant,
        # so a read-after-write hazard on them is harmless
        quiet = {op.a.name for op in (ins.add, ins.mul) if self_cancel(op) and isinstance(op.a, Reg)}
        for op in (ins.add, ins.mul):
            if op is not None and not self_cancel(op):
                quiet -= {x.name for x in op.srcs() if isinstance(x, Reg)}

        def get(src):
            if isinstance(src, Imm):
                return np.full(LANES, src.value, dtype=np.uint32)
            if isinstance(src, Reg) and src.name in quiet and src.acc is None and src.raddr is not None \
                    and src.raddr < 32:
                return np.zeros(LANES, dtype=np.uint32)
            key = src.name
            if key not in cache:
                if src.acc is not None:
                    cache[key] = self.read(src, None)
                else:
                    cache[key] = self.read_regfile(src, "B" if src.rfile == "B" else "A")
            return cache[key]

        add, mul = ins.add, ins.mul
        add_res = mul_res = None
        if add is not None:
            a = get(add.a)
            b = a if add.op in isa.UNARY_OPS else get(add.b)
            if add.op == "nop":
                add_res = (np.zeros(LANES, np.uint32), False, None)
            else:
                add_res = alu_add("or" if add.op == "mov" else add.op, a, b)
                if add.op in ("fadd", "fsub"):
                    self.stats["fadd"] += 1
        if mul is not None:
            if mul.rot is not None:
                for s in mul.srcs():
                    if not (isinstance(s, Reg) and s.acc is not None and s.acc < 4):
                        self.hazard("vector rotation with a non r0-r3 source")
                    elif s.acc in self.last_acc_written:
                        self.hazard(f"rotation of {s.name} written by the previous instruction")
                if mul.rot == "r5" and self.last_r5_written:
                    self.hazard("rotation by r5 immediately after an r5 write")
            a = get(mul.a)
            b = a if mul.op in isa.UNARY_OPS else get(mul.b)
            if mul.op == "nop":
                mul_res = (np.zeros(LANES, np.uint32), False)
            else:
                mul_res = alu_mul(mul.op, a, b)
                if mul.op == "fmul":
                    self.stats["fmul"] += 1
            if mul.rot is not None:
                amount = (int(self.acc[5][0]) & 15) if mul.rot == "r5" else mul.rot
                mul_res = (np.roll(mul_res[0], amount), mul_res[1])

        tmu_data = None
        if ins.sig in ("ldtmu0", "ldtmu1"):
            unit = 0 if ins.sig == "ldtmu0" else 1
            if not self.tmu[unit]:
                raise EmulatorError(f"QPU{self.num}: ldtmu{unit} with an empty TMU FIFO")
            if self.r4_pending is not None:
                self.hazard("TMU load while an SFU result is pending")
            ready, tmu_data = self.tmu[unit].popleft()
            if ready > self.cycle:
                self.stall_cycles += ready - self.cycle
                self.cycle = ready
        elif ins.sig not in ("none", "thrend"):
            raise EmulatorError(f"unsupported signal {ins.sig}")

        for op in (add, mul):
            if op is not None and op.dst in isa.SFU_REGS and self.r4_pending is not None:
                self.hazard("SFU write while another SFU result is pending")

        if add is not None:
            self.write(add.dst, self._dst_file(ins, add, "A"), add_res[0],
                       _cond_mask(add.cond, self.flags), writes)
        if mul is not None:
            self.write(mul.dst, self._dst_file(ins, mul, "B"), mul_res[0],
                       _cond_mask(mul.cond, self.flags), writes)
        if ins.setf:
            if add is not None and add.op != "nop" and add.cond != "never":
                self._set_flags(*add_res)
            elif mul is not None:
                self._set_flags(mul_res[0], mul_res[1], None)
        if tmu_data is not None:
            writes.append((_R4_SINK, "A", tmu_data, None))


class _R4Sink:
    waddr = -4
    name = "r4"


_R4_SINK = _R4Sink()


class _Blocked(Exception):
    pass


# ---------------------------------------------------------------------------
# The V3D: QPUs, shared units and the user-program request queue
# ---------------------------------------------------------------------------


class V3D:
    # tmu_fifo_depth: the guide says 8 direct lookups fit per TMU, but on a
    # Pi Zero the 5th outstanding request overwrites the oldest result
    # (tools/probes.py tmu_fifo). 4 per TMU is the real limit.
    def __init__(self, num_qpus: int = 12, mem: Optional[Memory] = None, strict: bool = True,
                 tmu_fifo_depth: int = 4, timing: Optional[TimingModel] = None):
        self.mem = mem or Memory()
        self.vpm = VPMState()
        self.strict = strict
        self.tmu_fifo_depth = tmu_fifo_depth
        self.timing = timing or TimingModel()
        self.qpus = [QPU(i, self) for i in range(num_qpus)]
        self.require_vpm_mutex = True
        self.vdw_stride = {"stride": 0, "blockmode": 0}     # shared by all QPUs (measured)
        self.sems = [0] * 16
        self.mutex_owner = None
        self.host_interrupts = 0
        self.queue = deque()
        self.completed = 0
        self.programs = {}
        self.finished_qpu_cycles = []

    # semaphores / mutex
    def sem_acquire(self, n):
        if self.sems[n] == 0:
            return False
        self.sems[n] -= 1
        return True

    def sem_release(self, n):
        if self.sems[n] >= 15:
            raise EmulatorError("semaphore overflow")
        self.sems[n] += 1

    def mutex_acquire(self, qpu):
        if self.mutex_owner in (None, qpu.num):
            self.mutex_owner = qpu.num
            return True
        return False

    def mutex_release(self, qpu):
        if self.mutex_owner not in (None, qpu.num):
            raise EmulatorError(f"QPU{qpu.num} released a mutex held by QPU{self.mutex_owner}")
        self.mutex_owner = None

    def active_qpus(self) -> int:
        return sum(1 for q in self.qpus if q.running)

    def on_finish(self, qpu):
        self.completed += 1
        self.finished_qpu_cycles.append(qpu.cycle)

    # program loading
    def load(self, words) -> tuple:
        """Decode a program once; returns a handle usable with `launch`."""
        key = id(words)
        decoded = []
        for pc, w in enumerate(words):
            ins = isa.decode(w)
            extra = None
            if ins.kind == "branch":
                if ins.rel:
                    extra = (8 * (pc + 4) + ins.imm) // 8
                else:
                    extra = ins.imm // 8 if ins.breg is None else ins.imm
            decoded.append((ins, extra))
        return decoded

    def run(self, program, unif_addrs, max_cycles: int = 200_000_000) -> dict:
        """Run one program instance per uniforms address, like the host queue.

        All QPUs share the queue; a QPU picks the next request as soon as it
        becomes idle. Returns timing statistics.
        """
        pending = deque(unif_addrs)
        global_cycle = 0
        for q in self.qpus:
            q.running = False
            q.cycle = 0
        total_instr = 0
        stalls = 0
        stats = {"tmu_loads": 0, "vdw_bytes": 0, "vdr_bytes": 0, "fmul": 0, "fadd": 0}
        busy_cycles = [0] * len(self.qpus)
        start = {}
        while True:
            for q in self.qpus:
                if not q.running and pending:
                    u = pending.popleft()
                    t0 = max(q.cycle, 0)
                    q.start(program, 0, u)
                    q.cycle = t0
                    start[q.num] = t0
            running = [q for q in self.qpus if q.running]
            if not running:
                break
            # advance the QPU that is furthest behind in time (keeps shared-unit
            # interactions roughly ordered)
            q = min(running, key=lambda x: x.cycle)
            try:
                q.step()
            except _Blocked:
                q.cycle += 1
                q.stall_cycles += 1
            if q.cycle > max_cycles:
                raise EmulatorError("cycle limit exceeded (infinite loop?)")
            if not q.running:
                total_instr += q.instructions
                stalls += q.stall_cycles
                busy_cycles[q.num] = q.cycle
                for k in stats:
                    stats[k] += q.stats[k]
        makespan = max(busy_cycles) if busy_cycles else 0
        stats.update(instructions=total_instr, stall_cycles=stalls, cycles=makespan)
        return stats
