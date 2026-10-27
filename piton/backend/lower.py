"""Tile IR -> machine IR for the VideoCore IV QPU.

Layout
------
A block tensor of N elements lives in ceil(N/16) 16-lane registers in
row-major order: element f is lane f % 16 of register f // 16.  Values are
kept in the cheapest representation that is still exact:

  S        scalar: a python constant or a uniform register
  Splat    every element equal to one scalar
  Affine   integer/pointer tensor  base + sum_d coeff_d * index_d
           (tl.arange / program-id index math; materialised per register as
           one add of a lane-pattern register and a per-register offset)
  Dense    list of registers (or pending TMU tokens)
  Scal     one uniform register per element (reduction results, broadcasts)
  Pred     lazy predicate tree (masks), turned into condition flags only
           where a load / store / select needs them

Memory
------
Loads are TMU gathers (16 independent addresses per request, at most 4
requests outstanding per TMU - measured, see tools/probes.py).  Loads of a
loop body whose addresses only depend on loop-carried pointers are software
pipelined: requests for the next iteration are issued as soon as FIFO slots
free up.  Stores go through this QPU's private VPM rows and the VDW DMA.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .. import ir
from ..ir import Value
from ..language.core import CompilationError
from ..passes import RangeAnalysis
from ..qpu import isa
from ..qpu.isa import Imm, f2u, is_small_imm
from . import mir
from .mir import MInst, MOp, VReg, alu

LANES = 16
TMU_DEPTH = 4                  # per TMU, measured on hardware
VPM_ROWS_PER_QPU = 4


def _f2u(x: float) -> int:
    return f2u(float(np.float32(x)))


# ---------------------------------------------------------------------------
# Value representations
# ---------------------------------------------------------------------------


@dataclass
class S:
    """Scalar: constant python value or a (uniform) virtual register."""
    const: Union[int, float, None] = None
    reg: Optional[VReg] = None
    is_float: bool = False

    @property
    def is_const(self):
        return self.reg is None

    def __repr__(self):
        return repr(self.const) if self.reg is None else repr(self.reg)


def C(v, is_float=False) -> S:
    return S(const=v, is_float=is_float)


@dataclass
class ArgReload:
    """A kernel argument that is re-read from the uniform list when needed."""
    index: int
    is_float: bool = False


@dataclass
class Splat:
    shape: tuple
    dtype: ir.DType
    s: S


@dataclass
class Affine:
    """value = dense + base + sum_d coeffs[d] * index_d   (dense is optional)."""
    shape: tuple
    dtype: ir.DType
    base: S
    coeffs: list            # list[S]
    dense: Optional["Dense"] = None


@dataclass
class Dense:
    shape: tuple
    dtype: ir.DType
    regs: list              # VReg | Token


@dataclass
class Scal:
    shape: tuple
    dtype: ir.DType
    elems: list             # list[S], row-major


@dataclass
class Pred:
    kind: str               # 'const' | 'cmp' | 'and' | 'or' | 'not' | 'dense'
    shape: tuple
    value: object = None    # const bool / predicate name / Dense
    args: list = field(default_factory=list)
    wide: bool = True       # cmp: x - y may overflow, so compare with min/max


def numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def nregs(shape) -> int:
    return max(1, (numel(shape) + LANES - 1) // LANES)


def strides(shape):
    out = []
    acc = 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return list(reversed(out))


def index_patterns(shape, j):
    """Per-dimension index of each lane of register j (numpy int arrays)."""
    flat = j * LANES + np.arange(LANES)
    st = strides(shape)
    return [(flat // st[d]) % shape[d] for d in range(len(shape))]


# ---------------------------------------------------------------------------
# TMU queue
# ---------------------------------------------------------------------------


class Token:
    __slots__ = ("unit", "vreg", "post", "want_acc", "stream", "entry", "issued", "emit_addr")

    def __init__(self, unit):
        self.unit = unit
        self.vreg = None
        self.post = None
        self.want_acc = False
        self.stream = None
        self.entry = None
        self.issued = False
        self.emit_addr = None


class TMUQueue:
    def __init__(self, lw: "Lowering"):
        self.lw = lw
        self.q = [deque(), deque()]
        self.next_unit = 0
        self.stream = None          # active LoopStream

    def outstanding(self):
        return len(self.q[0]) + len(self.q[1])

    def issue(self, emit_addr, unit=None, tok=None) -> Token:
        u = (self.next_unit if unit is None else unit) & 1
        if unit is None:
            self.next_unit ^= 1
        while len(self.q[u]) >= TMU_DEPTH:
            self._recv_one(u)
        tok = tok or Token(u)
        tok.unit = u
        emit_addr(isa.TMU_S[u])
        tok.issued = True
        self.q[u].append(tok)
        return tok

    def receive(self, tok: Token, acc=False) -> VReg:
        if tok.vreg is not None:
            return tok.vreg
        if not tok.issued:
            tok.stream.issue_until(tok)
        tok.want_acc = tok.want_acc or acc
        while tok.vreg is None:
            if not self.q[tok.unit]:
                raise CompilationError("internal: TMU token lost")
            self._recv_one(tok.unit)
        return tok.vreg

    def _recv_one(self, u):
        t = self.q[u].popleft()
        lw = self.lw
        v4 = VReg("tmu", cls="r4")
        lw.emit(MInst(ops=[MOp("r4def", v4, [], unit="none")], sig=f"ldtmu{u}"))
        dst = VReg("ld", cls="acc" if t.want_acc else "any")
        lw.emit(alu("mov", dst, v4))
        t.vreg = t.post(dst) if t.post else dst
        if self.stream is not None:
            self.stream.on_receive()

    def drain(self):
        while self.q[0] or self.q[1]:
            for u in (0, 1):
                if self.q[u]:
                    self._recv_one(u)


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


class Lowering:
    def __init__(self, fn: ir.Function, ranges: RangeAnalysis, options: dict):
        self.fn = fn
        self.ranges = ranges
        self.opts = options
        self.mf = mir.Function(fn.name)
        self.cold_blocks = []
        self.block_sink = self.mf.blocks
        self.blocks = 0
        self.cur = self.new_block("entry")
        self.env = {}
        self.uniforms = []           # host ABI: list of ("arg", name) / ("pid", ax) / ...
        self.tmu = TMUQueue(self)
        self.consts = {}
        self.vcache = {}
        # uniforms and hoisted constants live in a preamble that falls through
        self.const_block = self.cur
        self.cur = self.new_block("main")
        self.lanevecs = {}
        self.vpm_setup_base = None
        self.vdw_base = None
        self.vpm_block = None
        self.vdw_pending = False
        self.vdw_stride = None
        self.loop_depth = 0
        self.stats = {"tmu_requests": 0, "dma_stores": 0}
        self.zero_ptr = None
        self.scratch_ptr = None
        self.unif_base = None
        self.region_depth = 0
        self.loop_cmp_wide = False
        self.row_gap_fits = None
        self.divcache = {}
        self.in_pipelined_body = False
        self.use_counts = {}
        for o in fn.walk():
            for v in o.operands:
                if v is not None:
                    self.use_counts[v.id] = self.use_counts.get(v.id, 0) + 1

    # -- infrastructure --------------------------------------------------------

    def new_block(self, hint="bb") -> mir.Block:
        b = mir.Block(f"{hint}{self.blocks}" if hint != "entry" else "entry",
                      loop_depth=getattr(self, "loop_depth", 0))
        self.blocks += 1
        self.add_block(b)
        return b

    def add_block(self, b: mir.Block):
        self.block_sink.append(b)

    def begin_cold(self, label):
        """Start out-of-line code (placed after the program's end): returns the
        state end_cold() restores."""
        saved = (self.block_sink, self.cur)
        self.block_sink = []
        self.cur = mir.Block(label, loop_depth=self.loop_depth)
        self.block_sink.append(self.cur)
        return saved

    def end_cold(self, saved, join_label):
        self.emit(MInst(kind="branch", bcond="always", target=join_label))
        for b in self.block_sink:
            b.cold = True
        self.cold_blocks.extend(self.block_sink)
        self.block_sink, self.cur = saved

    def emit(self, inst: MInst) -> MInst:
        self.cur.insts.append(inst)
        return inst

    def op(self, name, dst, *srcs, cond="always", setf=False, rot=None, unit="") -> VReg:
        # one small-immediate field per instruction (a rotation also uses it)
        imms = [x for x in srcs if isinstance(x, Imm)]
        if imms and (rot is not None or len({i.value for i in imms}) > 1):
            keep = None if rot is not None else imms[-1].value
            fixed = []
            for x in srcs:
                if isinstance(x, Imm) and x.value != keep:
                    t = VReg("im")
                    self.emit(alu("mov", t, x))
                    x = t
                fixed.append(x)
            srcs = fixed
        # a B-file special register cannot share the B port with a small immediate
        if any(isinstance(x, Imm) for x in srcs) or rot is not None:
            fixed = []
            for x in srcs:
                if isinstance(x, isa.Reg) and x.acc is None and x.rfile == "B":
                    t = VReg("sp")
                    self.emit(alu("mov", t, x))
                    x = t
                fixed.append(x)
            srcs = fixed
        self.emit(alu(name, dst, *srcs, cond=cond, setf=setf, rot=rot, unit=unit))
        return dst

    def new(self, name="t", is_float=False, cls="any") -> VReg:
        return VReg(name, cls=cls, is_float=is_float)

    def error(self, msg):
        return CompilationError(f"{self.fn.name}: {msg}")

    # -- constants / operands --------------------------------------------------

    def const_reg(self, bits: int) -> VReg:
        """Register holding a 32-bit constant (one ldi per block; machine-IR
        LICM moves the loop-invariant ones out of loops)."""
        bits &= 0xFFFFFFFF
        key = (self.cur.label, bits)
        if key in self.consts:
            return self.consts[key]
        v = VReg("k")
        self.emit(MInst(kind="ldi", ops=[MOp("mov", v, [])], imm=bits, comment=f"const 0x{bits:x}"))
        self.consts[key] = v
        return v

    def operand(self, s: S):
        """Machine operand for scalar s (small immediate, constant reg or reg)."""
        if s.reg is not None:
            return s.reg
        bits = _f2u(s.const) if s.is_float else int(s.const) & 0xFFFFFFFF
        if is_small_imm(bits):
            return Imm(bits)
        return self.const_reg(bits)

    def sreg(self, s: S) -> VReg:
        """Scalar as a register."""
        if s.reg is not None:
            return s.reg
        o = self.operand(s)
        if isinstance(o, VReg):
            return o
        v = self.new("c", s.is_float)
        self.op("mov", v, o)
        return v

    def uniform(self, kind, name) -> VReg:
        v = VReg(name)
        self.entry_uniform_insts.append(alu("mov", v, isa.UNIF))
        self.uniforms.append((kind, name))
        return v

    # -- scalar arithmetic -----------------------------------------------------

    def s_bin(self, opname: str, a: S, b: S, is_float: bool, signed=True, ra=None, rb=None) -> S:
        if a.is_const and b.is_const:
            return C(_const_op(opname, a.const, b.const, is_float), is_float)
        key = (self.cur.label, "s", opname, repr(a), repr(b), is_float)
        if key in self.vcache:
            return self.vcache[key]
        if opname in ("divs", "rems", "divu", "remu"):
            # valid while no block was started since (the division ends in its own block)
            dkey = (opname[-1], repr(a), repr(b))
            ent = self.divcache.get(dkey)
            if ent is None or ent[2] is not self.cur:
                q, r = self.idiv(self.sreg(a), self.operand(b), signed=opname.endswith("s"), ra=ra, rb=rb, cb=b)
                ent = self.divcache[dkey] = (q, r, self.cur)
            return S(reg=ent[0] if opname.startswith("div") else ent[1])
        out = self.new("s", is_float)
        self.elementwise(opname, out, self.operand(a), self.operand(b), is_float, signed,
                         ra, rb, a, b)
        res = S(reg=out, is_float=is_float)
        self.vcache[key] = res
        return res

    def elementwise(self, opname, dst, a, b, is_float, signed=True, ra=None, rb=None, sa=None, sb=None):
        """Emit one register worth of `opname`."""
        if is_float:
            simple = {"add": "fadd", "sub": "fsub", "mul": "fmul", "min": "fmin", "max": "fmax"}
            if opname in simple:
                self.op(simple[opname], dst, a, b)
                return
            if opname == "div":
                self.fdiv(dst, a, b)
                return
            raise self.error(f"float op {opname} not supported")
        simple = {"add": "add", "sub": "sub", "and": "and", "or": "or", "xor": "xor", "shl": "shl",
                  "min": "min", "max": "max", "shrs": "asr", "shru": "shr"}
        if opname in ("add", "sub") and sb is not None and sb.is_const and isinstance(b, VReg):
            neg = (-_wrap(sb.const)) & 0xFFFFFFFF
            if is_small_imm(neg):        # x + 16 == x - (-16): no constant register
                self.op("sub" if opname == "add" else "add", dst, a, Imm(neg))
                return
        if opname in simple:
            self.op(simple[opname], dst, a, b)
            return
        if opname == "mul":
            self.imul(dst, a, b, ra, rb, sa, sb)
            return
        if opname in ("maxu", "minu"):
            # unsigned compare = signed compare with the sign bit flipped
            k = self.const_reg(0x80000000)
            x, y, m = self.new("u"), self.new("u"), self.new("u")
            self.op("xor", x, a, k)
            self.op("xor", y, b, k)
            self.op("max" if opname == "maxu" else "min", m, x, y)
            self.op("xor", dst, m, k)
            return
        if opname in ("divs", "rems", "divu", "remu"):
            q, r = self.idiv(a, b, signed=opname.endswith("s"), ra=ra, rb=rb, cb=sb)
            self.op("mov", dst, q if opname.startswith("div") else r)
            return
        raise self.error(f"int op {opname} not supported")

    def imul(self, dst, a, b, ra=None, rb=None, sa=None, sb=None):
        ca = sa.const if sa is not None and sa.is_const else _imm_const(a)
        cb = sb.const if sb is not None and sb.is_const else _imm_const(b)
        if ca is not None and cb is None:
            a, b, ca, cb, ra, rb = b, a, cb, ca, rb, ra
        if isinstance(a, Imm):
            t = self.new("m")
            self.op("mov", t, a)
            a = t
        if cb is not None:
            cb = _wrap(cb)
            if cb < 0 and cb != -(1 << 31):
                # x * -c == -(x * c)
                t = self.new("m")
                self.imul(t, a, self.operand(C(-cb)), ra, None, None, C(-cb))
                self.op("sub", dst, Imm(0), t)
                return
            if cb == 0:
                self.op("mov", dst, Imm(0))
                return
            if cb == 1:
                self.op("mov", dst, a)
                return
            if cb > 0 and cb & (cb - 1) == 0:
                self.op("shl", dst, a, Imm(_shift_imm(cb.bit_length() - 1)))
                return
            if 0 < cb < 1024 and bin(cb).count("1") == 2:
                hi, lo = cb.bit_length() - 1, (cb & -cb).bit_length() - 1
                t1, t2 = self.new("m"), self.new("m")
                self.op("shl", t1, a, Imm(_shift_imm(hi)))
                if lo:
                    self.op("shl", t2, a, Imm(_shift_imm(lo)))
                else:
                    t2 = a
                self.op("add", dst, t1, t2)
                return
            if 0 < cb < 1024 and (cb + 1) & cb == 0:          # 2^k - 1
                t1 = self.new("m")
                self.op("shl", t1, a, Imm(_shift_imm(cb.bit_length())))
                self.op("sub", dst, t1, a)
                return
            if 0 < cb < (1 << 24) and _fits24(ra):
                self.op("mul24", dst, a, self.operand(C(cb)))
                return
            b = self.operand(C(cb))
            rb = (cb, cb) if cb >= 0 else None
        if _fits24(ra) and _fits24(rb):
            self.op("mul24", dst, a, b)
            return
        if isinstance(b, Imm):
            t = self.new("m")
            self.op("mov", t, b)
            b = t
        # exact 32-bit product from three 24-bit products
        p0, ah, bh, p1, p2, s = (self.new("m") for _ in range(6))
        self.op("mul24", p0, a, b)
        self.op("shr", ah, a, Imm(_shift_imm(24)))
        self.op("shr", bh, b, self.const_reg(24))      # (b is often long-lived: no small imm)
        self.op("mul24", p1, ah, b)
        self.op("mul24", p2, a, bh)
        self.op("add", s, p1, p2)
        s2 = self.new("m")
        self.op("shl", s2, s, Imm(_shift_imm(24)))
        self.op("add", dst, p0, s2)

    def recip(self, b) -> VReg:
        """1/b on the ALUs: integer seed (12% error) and three Newton steps.
        The SFU's results are not reproducible bit for bit, these are."""
        bc = _imm_const_f(b)
        if bc is not None and bc != 0:
            return self.sreg(C(float(np.float32(1.0) / np.float32(bc)), True))
        if isinstance(b, Imm) or (isinstance(b, VReg) and b.cls == "r4"):
            bb = self.new("d", True)
            self.op("mov", bb, b)
            b = bb
        r = self.recip_core(b)
        eb, sg = self.new("ex"), self.new("ex")
        self.op("and", eb, b, self.const_reg(0x7F800000))
        self.op("and", sg, b, self.const_reg(0x80000000))
        # |b| >= 2**126 (and inf): 1/b is a denormal or 0, flushed to +-0
        self.op("sub", isa.NOP, eb, self.const_reg(0x7E800000), setf=True, unit="add")
        self.op("mov", r, sg, cond="ifnn")
        # +-0 (and denormals, flushed): +-inf
        si = self.new("ex")
        self.op("or", si, sg, self.const_reg(0x7F800000))
        self.op("mov", isa.NOP, eb, setf=True, unit="add")
        self.op("mov", r, si, cond="ifz")
        self.keep_nan(r, b)
        return r

    def recip_core(self, b) -> VReg:
        """1/b for normal b (no special values)."""
        r = self.new("nr", True)
        self.op("sub", r, self.const_reg(0x7EF311C3), b)
        for _ in range(3):
            t, e, r2 = self.new("nr", True), self.new("nr", True), self.new("nr", True)
            self.op("fmul", t, b, r)
            self.op("fsub", e, Imm(_f2u(2.0)), t)
            self.op("fmul", r2, r, e)
            r = r2
        return r

    def fdiv(self, dst, a, b):
        """a / b = a * (1 / b).  The QPU multiplier reads NaN as inf and gives
        0 * inf = +0, so NaN operands, 0 / 0 and inf / inf are set to NaN."""
        key = ("recip", id(b), self.cur.label)
        if key not in self.consts or not isinstance(b, VReg):
            self.consts[key] = self.recip(b)
        self.op("fmul", dst, a, self.consts[key])
        if _imm_const_f(a) is not None and _imm_const_f(b) is not None:
            return
        ma, mb = self.new("dv"), self.new("dv")
        for m, x in ((ma, a), (mb, b)):
            if isinstance(x, Imm):
                self.op("mov", m, x)
                self.op("and", m, m, self.const_reg(0x7FFFFFFF))
            else:
                self.op("and", m, x, self.const_reg(0x7FFFFFFF))
        nan = self.const_reg(0x7FC00000)
        t = self.new("dv")
        self.op("max", t, ma, mb)
        self.op("sub", isa.NOP, t, self.const_reg(0x7F800001), setf=True, unit="add")
        self.op("mov", dst, nan, cond="ifnn")                      # a or b is NaN
        t = self.new("dv")
        self.op("or", t, ma, mb)
        self.op("sub", isa.NOP, t, self.const_reg(0x00800000), setf=True, unit="add")
        self.op("mov", dst, nan, cond="ifn")                       # 0 / 0 (denormals are 0)
        t = self.new("dv")
        self.op("and", t, ma, mb)
        self.op("sub", isa.NOP, t, self.const_reg(0x7F800000), setf=True, unit="add")
        self.op("mov", dst, nan, cond="ifnn")                      # inf / inf

    def rsqrt_core(self, x) -> VReg:
        """1/sqrt(x) for x > 0 normal: integer seed (3.4% error), three Newton steps."""
        h = self.new("rs")
        self.op("shr", h, x, Imm(1))
        r = self.new("rs", True)
        self.op("sub", r, self.const_reg(0x5F375A86), h)
        for _ in range(3):
            # r * (1.5 - 0.5 * (x * r) * r): x * r ~ sqrt(x) and (x * r) * r ~ 1 stay
            # normal for every normal x (r * r or 0.5 * x would underflow at the ends)
            t1, t2, t3, t4, r2 = (self.new("rs", True) for _ in range(5))
            self.op("fmul", t1, x, r)
            self.op("fmul", t2, t1, r)
            self.op("fmul", t3, t2, Imm(_f2u(0.5)))
            self.op("fsub", t4, self.fk(1.5), t3)
            self.op("fmul", r2, r, t4)
            r = r2
        return r

    def idiv(self, a, b, signed=True, ra=None, rb=None, cb=None):
        """Integer quotient and remainder (C semantics, exact for all 32-bit values)."""
        bconst = cb.const if cb is not None and cb.is_const else _imm_const(b)
        a_nonneg = ra is not None and ra[0] >= 0
        if bconst is not None:
            bconst = _wrap(bconst)
            if bconst == 0:
                raise self.error("division by constant zero")
            if bconst == 1:
                q, r = self.new("q"), self.new("r")
                self.op("mov", q, a)
                self.op("mov", r, Imm(0))
                return q, r
            if bconst > 0 and bconst & (bconst - 1) == 0:
                k = bconst.bit_length() - 1
                q, r = self.new("q"), self.new("r")
                if a_nonneg or not signed:
                    self.op("shr" if not signed else "asr", q, a, Imm(_shift_imm(k)))
                    self.op("and", r, a, self.operand(C(bconst - 1)))
                    return q, r
                # signed: bias negative values towards zero
                sgn, bias, t = self.new("s"), self.new("s"), self.new("s")
                self.op("asr", sgn, a, Imm(_shift_imm(31)))
                self.op("shr", bias, sgn, Imm(_shift_imm(32 - k)))
                self.op("add", t, a, bias)
                self.op("asr", q, t, Imm(_shift_imm(k)))
                qs = self.new("s")
                self.op("shl", qs, q, Imm(_shift_imm(k)))
                self.op("sub", r, a, qs)
                return q, r
        b_pos = (rb is not None and rb[0] > 0) or (bconst is not None and bconst > 0)
        signed_a = signed and not a_nonneg
        signed_b = signed and not b_pos
        ua, ub = a, b
        sa = sb = None
        if signed_a:
            x, ua = self.new("x"), self.new("ua")
            sa = self.sign_mask(a)
            self.op("xor", x, a, sa)
            self.op("sub", ua, x, sa)
        if signed_b:
            # |b| (and its sign) are shared by consecutive divisions by the same b
            ent = self.divcache.get(("absb", id(b)))
            if ent is not None and ent[2] is self.cur:
                sb, ub = ent[0], ent[1]
            else:
                x, ub = self.new("x"), self.new("ub")
                sb = self.sign_mask(b)
                self.op("xor", x, b, sb)
                self.op("sub", ub, x, sb)
        if isinstance(ub, Imm):
            u2 = self.new("ub")
            self.op("mov", u2, ub)
            ub = u2
        ubc = abs(bconst) if bconst is not None else None
        if not signed and bconst is not None:
            ubc = bconst & 0xFFFFFFFF
        ub_max = ubc
        if ub_max is None and rb is not None and (signed or rb[0] >= 0):
            ub_max = max(abs(rb[0]), abs(rb[1]))
        ua_max = max(abs(ra[0]), abs(ra[1])) if ra is not None and (signed or ra[0] >= 0) else None
        q, r = self.udivmod(ua, ub, ubc, ub_max, ua_max)
        if signed_b and not isinstance(b, Imm):
            self.divcache[("absb", id(b))] = (sb, ub, self.cur)
        if signed_a or signed_b:
            sq = self.new("sq")
            if signed_a and signed_b:
                self.op("xor", sq, sa, sb)
            else:
                self.op("mov", sq, sa if signed_a else sb)
            x, q2 = self.new("x"), self.new("q")
            self.op("xor", x, q, sq)
            self.op("sub", q2, x, sq)
            q = q2
            if signed_a:
                x2, r2 = self.new("x"), self.new("r")
                self.op("xor", x2, r, sa)
                self.op("sub", r2, x2, sa)
                r = r2
        return q, r

    def u2f(self, x) -> VReg:
        """uint32 register -> float, to within 2 ulp (for estimates): values >= 2**31
        read as negative by itof get 2**32 added.  x is read with no small
        immediate, so a long-lived x is not tied to regfile A."""
        d = self.new("u", True)
        self.op("itof", d, x)
        self.op("mov", isa.NOP, x, setf=True, unit="add")
        self.op("fadd", d, d, self.const_reg(0x4F800000), cond="ifn")
        return d

    def sign_mask(self, x) -> VReg:
        """-1 where x < 0, else 0 (like asr x, 31 without a small immediate on x)."""
        m = self.new("sg")
        self.op("mov", m, Imm(0))
        self.op("mov", isa.NOP, x, setf=True, unit="add")
        self.op("mov", m, Imm(-1 & 0xFFFFFFFF), cond="ifn")
        return m

    def f2u(self, x) -> VReg:
        """float in [0, 2**32) -> uint32: ftoi keeps the low 32 bits of the truncated
        value (measured, tools/float_specials_probe.py), so this is one ftoi."""
        d = self.new("u")
        self.op("ftoi", d, x)
        return d

    def udivmod(self, ua, ub, ubc=None, ub_max=None, ua_max=None):
        """Exact uint32 quotient and remainder (ub != 0; ubc: ub's value if constant;
        ua_max / ub_max: bounds if known).  Operands below 2**23 (the usual case)
        take a short sequence, chosen at compile time from the ranges or else by a
        uniform branch on the values."""
        small = 1 << 23
        b_small = ub_max is not None and ub_max < small
        if ubc is not None:
            rcp = self.sreg(C(float(np.float32(1.0) / np.float32(ubc)), True))
        else:
            # 1 / ub, shared by consecutive divisions by the same register (e.g. the
            # registers of offs % N); computed before the branch so both paths have it
            ent = self.divcache.get(("rcp", id(ub)))
            if ent is not None and ent[1] is self.cur:
                rcp = ent[0]
            else:
                fb = self.new("fb", True)
                if b_small:
                    self.op("itof", fb, ub)
                else:
                    fb = self.u2f(ub)
                rcp = self.recip_core(fb)
        if ua_max is not None and ua_max < small and b_small:
            q, r = self._udivmod_small(ua, ub, rcp)
        elif ubc is not None and ubc >= small:
            q, r = self._udivmod_full(ua, ub, rcp, ubc, ub_max)
        else:
            q, r = self._udivmod_branch(ua, ub, rcp, ubc, ub_max, b_small)
        if ubc is None:
            self.divcache[("rcp", id(ub))] = (rcp, self.cur)
        return q, r

    def _udivmod_branch(self, ua, ub, rcp, ubc, ub_max, b_small):
        t = self.new("t")
        self.op("or", t, ua, ub)
        self.op("shr", isa.NOP, t, Imm(_shift_imm(23)), setf=True, unit="add")
        full_label, done_label = f"div_full{self.blocks}", f"div_done{self.blocks + 1}"
        self.blocks += 2
        qo, ro = self.new("q"), self.new("r")
        self.emit(MInst(kind="branch", bcond="anynz", target=full_label))
        self.cur = self.new_block("div_small")
        q, r = self._udivmod_small(ua, ub, rcp)
        self.op("mov", qo, q)
        self.op("mov", ro, r)
        done = mir.Block(done_label, loop_depth=self.loop_depth)
        self.add_block(done)
        saved = self.begin_cold(full_label)
        q, r = self._udivmod_full(ua, ub, rcp, ubc, ub_max)
        self.op("mov", qo, q)
        self.op("mov", ro, r)
        self.end_cold(saved, done_label)
        self.cur = done
        return qo, ro

    def _udivmod_small(self, ua, ub, rcp):
        """ua, ub < 2**23: the float estimate is within a few units, one refinement
        brings it within one, and the products fit mul24."""
        fa = self.new("fa", True)
        self.op("itof", fa, ua)
        fq, q = self.new("fq", True), self.new("q")
        self.op("fmul", fq, fa, rcp)
        self.op("ftoi", q, fq)
        qb, r1 = self.new("qb"), self.new("r")
        self.op("mul24", qb, q, ub)
        self.op("sub", r1, ua, qb)
        fr, fd, dq, q2 = self.new("fr", True), self.new("fd", True), self.new("dq"), self.new("q")
        self.op("itof", fr, r1)
        self.op("fmul", fd, fr, rcp)
        self.op("ftoi", dq, fd)
        self.op("add", q2, q, dq)
        qb, r = self.new("qb"), self.new("r")
        self.op("mul24", qb, q2, ub)
        self.op("sub", r, ua, qb)
        return self._div_correct(q2, r, ub)

    def _div_correct(self, q, r, ub):
        """r in (-ub, 2 ub): one correction each way."""
        t = self.new("t")
        self.op("sub", t, r, ub, setf=True)          # r - ub >= 0 -> too small
        self.op("add", q, q, Imm(1), cond="ifnn")
        self.op("mov", r, t, cond="ifnn")
        self.op("mov", isa.NOP, r, setf=True, unit="add")  # r < 0 -> too big
        self.op("sub", q, q, Imm(1), cond="ifn")
        self.op("add", r, r, ub, cond="ifn")
        return q, r

    def _udivmod_full(self, ua, ub, rcp, ubc=None, ub_max=None):
        """Any uint32 operands.

        A float estimate q1 of ua / ub is off by at most ~ua * 2**-21 + 2, so
        R1 = ua - q1 * ub (exact, mod 2**32) is a small signed number while
        ub < 2**29; q2 = q1 + R1 / ub is then within 1 and a last remainder fixes
        it.  Divisors >= 2**29 have quotients <= 7: three steps of long division."""
        fa = self.u2f(ua)
        sb = C(ubc) if ubc is not None else None
        fq = self.new("fq", True)
        self.op("fmul", fq, fa, rcp)
        q = self.f2u(fq)
        qb, r1 = self.new("qb"), self.new("r")
        self.imul(qb, q, ub, None, None, None, sb)
        self.op("sub", r1, ua, qb)
        fr, fd, dq, q2 = self.new("fr", True), self.new("fd", True), self.new("dq"), self.new("q")
        self.op("itof", fr, r1)
        self.op("fmul", fd, fr, rcp)
        self.op("ftoi", dq, fd)
        self.op("add", q2, q, dq)
        q = q2
        qb, r = self.new("qb"), self.new("r")
        self.imul(qb, q, ub, None, None, None, sb)
        self.op("sub", r, ua, qb)
        q, r = self._div_correct(q, r, ub)
        if ub_max is not None and ub_max < 1 << 29:
            return q, r
        if ubc is not None and ubc < 1 << 29:
            return q, r
        # ub >= 2**29 (quotient <= 7): lanes restart from q = 0, r = ua and run three
        # steps of long division in place; bit k is set when (r >> k) >= ub, compared
        # unsigned (no overflow).  Other lanes compare against 0xFFFFFFFF, which no
        # remainder reaches, so they keep the result above.
        k31 = self.const_reg(0x80000000)
        self.op("shr", isa.NOP, ub, self.const_reg(29), setf=True, unit="add")   # no small imm on ub
        ubm = self.new("bm")
        self.op("mov", ubm, self.const_reg(0xFFFFFFFF))
        self.op("mov", ubm, ub, cond="ifnz")
        self.op("mov", q, Imm(0), cond="ifnz")
        self.op("mov", r, ua, cond="ifnz")
        ubf = self.new("bf")
        self.op("xor", ubf, ubm, k31)
        for k in (2, 1, 0):
            sh = r
            if k:
                sh = self.new("t")
                self.op("shr", sh, r, Imm(k))
            x, m = self.new("t"), self.new("t")
            self.op("xor", x, sh, k31)
            self.op("min", m, x, ubf)
            bk = ubm
            if k:
                bk = self.new("t")
                self.op("shl", bk, ubm, Imm(k))
            self.op("sub", isa.NOP, m, ubf, setf=True, unit="add")   # min == ub  <=>  (r >> k) >= ub
            self.op("sub", r, r, bk, cond="ifz")
            self.op("add", q, q, Imm(1 << k), cond="ifz")
        return q, r

    # -- value access ------------------------------------------------------------

    def get(self, v: Value):
        if v.id in self.env:
            x = self.env[v.id]
            if isinstance(x, ArgReload):
                r = self.reload_arg(x)
                if self.region_depth == 0:
                    self.env[v.id] = r       # inside a branch / loop the reload may not dominate
                return r
            return x
        op = v.op
        if op is not None and op.name not in ("tt.load", "tt.dot", "scf.for", "scf.if", "tt.store") \
                and getattr(self, "lazy_ok", None) and op in self.lazy_ok:
            self.lower_op(op)
            return self.env[v.id]
        raise self.error(f"value {v} used before definition")

    def scalar(self, v: Value) -> S:
        x = self.get(v)
        if isinstance(x, S):
            return x
        if isinstance(x, Splat):
            return x.s
        if isinstance(x, Affine) and all(c.is_const and c.const == 0 for c in x.coeffs):
            return x.base
        if isinstance(x, Pred) and x.kind == "const":
            return C(int(x.value))
        if isinstance(x, Pred):
            d = self.pred_dense(x)
            return S(reg=self.reg_of(d, 0))
        raise self.error(f"expected a scalar, got {type(x).__name__}")

    # -- materialisation -----------------------------------------------------------

    def lanevec(self, coeffs, shape, j) -> tuple:
        """Split register j of an affine value into a lane part and a row part.

        Dimensions whose stride is below 16 vary inside the register (or select
        the 16-column block): their contribution, including the column-block
        offset, goes into a lane register that is shared by every row.
        Dimensions with stride >= 16 are constant over the register and give a
        uniform row offset.  Returns (lane_terms, col_offset S, row_offset S).
        """
        pats = index_patterns(shape, j)
        st = strides(shape)
        lane_terms = []
        col = C(0)
        row = C(0)
        row_terms = []
        for d, c in enumerate(coeffs):
            if c.is_const and c.const == 0:
                continue
            pat = pats[d]
            if st[d] >= LANES:
                k = int(pat[0])
                if c.is_const:
                    row = self.s_add(row, C(_wrap(c.const * k)))
                else:
                    row_terms.append((c, k))
                continue
            period = st[d] * shape[d]
            if period > LANES:
                base = int(pat[0])
                lane = pat - base
                if base:
                    col = self.s_add(col, self.s_mul(c, C(base)))
            else:
                lane = pat
            if np.any(lane):
                lane_terms.append((tuple(int(x) for x in lane), c))
        self._row_terms = row_terms
        return lane_terms, col, row

    def s_add(self, a: S, b: S) -> S:
        if b.is_const and b.const == 0:
            return a
        if a.is_const and a.const == 0:
            return b
        return self.s_bin("add", a, b, False)

    def s_mul(self, a: S, b: S, ra=None, rb=None) -> S:
        if a.is_const and a.const == 1:
            return b
        if b.is_const and b.const == 1:
            return a
        if (a.is_const and a.const == 0) or (b.is_const and b.const == 0):
            return C(0)
        return self.s_bin("mul", a, b, False, ra=ra, rb=rb)

    def lane_reg(self, terms) -> Optional[VReg]:
        """Register holding sum_t coeff_t * pattern_t (cached, built from elem_num)."""
        if not terms:
            return None
        key = (self.cur.label,) + tuple((pat, (c.const if c.is_const else ("r", c.reg.id)))
                                        for pat, c in terms)
        if key in self.lanevecs:
            return self.lanevecs[key]
        saved = self.cur
        acc = None
        for pat, c in terms:
            pv = self.pattern_reg(pat)
            if c.is_const:
                cv = _wrap(c.const)
                if cv == 1:
                    t = pv
                else:
                    t = self.new("lv")
                    self.imul(t, pv, self.operand(C(cv)), (0, 15), None, None, C(cv))
            else:
                t = self.new("lv")
                self.imul(t, pv, c.reg, (0, 15), None)
            if acc is None:
                acc = t
            else:
                n = self.new("lv")
                self.op("add", n, acc, t)
                acc = n
        self.cur = saved
        self.lanevecs[key] = acc
        return acc

    def pattern_reg(self, pat: tuple) -> VReg:
        key = ("pat", self.cur.label, pat)
        if key in self.lanevecs:
            return self.lanevecs[key]
        p = np.array(pat)
        v = self.new("pat")
        if np.array_equal(p, np.arange(LANES)):
            self.op("mov", v, isa.ELEM_NUM)
        else:
            # pattern = ((elem >> s) & m) * k for some s, m, k (power-of-two shapes)
            found = False
            for s in range(5):
                for m in (1, 3, 7, 15):
                    base = (np.arange(LANES) >> s) & m
                    if not np.any(base):
                        continue
                    k = p[np.argmax(base)] // base[np.argmax(base)] if base[np.argmax(base)] else 0
                    if k and np.array_equal(base * k, p):
                        t = self.new("pat")
                        if s:
                            self.op("shr", t, isa.ELEM_NUM, Imm(s))
                        else:
                            self.op("mov", t, isa.ELEM_NUM)
                        if m != 15 or s:
                            t2 = self.new("pat")
                            self.op("and", t2, t, Imm(m))
                            t = t2
                        if k != 1:
                            self.imul(v, t, self.operand(C(int(k))), (0, 15), None, None, C(int(k)))
                        else:
                            self.op("mov", v, t)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise self.error(f"unsupported lane pattern {pat}")
        self.lanevecs[key] = v
        return v

    def affine_reg(self, a: Affine, j: int, dst=None):
        """Emit register j of an affine tensor. Returns (lane register or None,
        uniform offset S) with value = lane register + offset."""
        terms, col, row = self.lanevec(a.coeffs, a.shape, j)
        row_terms = self._row_terms
        lr = self.lane_reg(terms)
        if not (col.is_const and col.const == 0):
            # a lane register per column block only pays off when it is reused
            # by several rows and there are few column blocks
            nblocks = max(1, a.shape[-1] // LANES) if a.shape else 1
            rows = numel(a.shape[:-1]) if len(a.shape) > 1 else 1
            if lr is None or nblocks > 4 or rows < 2:
                row = self.s_add(row, col) if not (col.is_const and row.is_const) else \
                    C(_wrap(col.const + row.const))
            else:
                lr = self.lane_plus(lr, col)
        if a.dense is not None:
            dj = self.reg_of(a.dense, j)
            lr = dj if lr is None else self.cached("add", lr, dj)
        base = a.base
        for c, k in row_terms:
            base = self.chain_mul_add(base, c, k)
        return lr, self.offset(base, row)

    def chain_mul_add(self, base: S, c: S, k: int) -> S:
        """base + c*k for a runtime c, as a chain of adds shared between rows."""
        if k == 0:
            return base
        key = (self.cur.label, "cma", repr(base), repr(c), k)
        if key in self.vcache:
            return self.vcache[key]
        if k < 0 or k > 8:
            res = self.s_add(base, self.s_mul(c, C(k)))
        else:
            res = self.s_add(self.chain_mul_add(base, c, k - 1), c)
        self.vcache[key] = res
        return res

    def lane_plus(self, lr: VReg, col: S) -> VReg:
        """lane register + column-block offset.  Successive column blocks are
        built from each other so a constant stride needs one constant."""
        key = ("lane+", lr.id, repr(col))
        if key in self.lanevecs:
            return self.lanevecs[key]
        saved = self.cur
        out = self.new("lc")
        if col.is_const:
            best = None
            for k2, v in self.lanevecs.items():
                if k2[:2] == ("lane+", lr.id) and k2[2].lstrip("-").isdigit():
                    d = col.const - int(k2[2])
                    if best is None or abs(d) < abs(best[0]):
                        best = (d, v)
            src, d = (best[1], best[0]) if best is not None else (lr, col.const)
            self.elementwise("add", out, src, self.operand(C(d)), False, sb=C(d))
        else:
            self.op("add", out, lr, col.reg)
        self.cur = saved
        self.lanevecs[key] = out
        return out

    def _entry_regs(self):
        out = set()
        for ins in self.const_block.insts:
            out.update(ins.defs())
        return out

    def offset(self, base: S, off: S) -> S:
        """base + off, reusing an earlier base + off' of this block when the
        difference is cheap (a small immediate or an existing constant)."""
        if off.is_const and off.const == 0:
            return base
        if not off.is_const or base.is_const:
            return self.s_add(base, off)
        chain = self.vcache.setdefault((self.cur.label, "chain", repr(base)), {})
        if off.const in chain:
            return chain[off.const]
        best = None
        for o, sv in chain.items():
            d = off.const - o
            cheap = is_small_imm(d & 0xFFFFFFFF) or is_small_imm((-d) & 0xFFFFFFFF) or \
                ((self.cur.label, d & 0xFFFFFFFF) in self.consts)
            if cheap and (best is None or abs(d) < abs(best[0])):
                best = (d, sv)
        if best is not None:
            res = self.s_add(best[1], C(best[0]))
        else:
            res = self.s_add(base, off)
        chain[off.const] = res
        return res



    def cached(self, opname, x, y, is_float=False) -> VReg:
        """Block-local CSE of one register-wide operation."""
        def kid(v):
            return ("v", v.id) if isinstance(v, VReg) else ("o", repr(v))
        key = (self.cur.label, opname, kid(x), kid(y))
        if opname in ("add", "mul", "and", "or", "xor", "min", "max", "fadd", "fmul"):
            key2 = (self.cur.label, opname, kid(y), kid(x))
            if key2 in self.vcache:
                return self.vcache[key2]
        if key in self.vcache:
            return self.vcache[key]
        dst = self.new("cv", is_float)
        if opname in ("add", "sub", "and", "or", "xor", "min", "max", "shl", "fadd", "fsub", "fmul"):
            self.op(opname, dst, x, y)
        else:
            self.elementwise(opname, dst, x, y, is_float)
        self.vcache[key] = dst
        return dst

    def materialize(self, x, shape=None) -> Dense:
        if isinstance(x, Dense):
            return x
        if isinstance(x, S):
            return Dense((), None, [self.sreg(x)])
        if isinstance(x, Splat):
            r = self.sreg(x.s)
            return Dense(x.shape, x.dtype, [r] * nregs(x.shape))
        if isinstance(x, Affine):
            regs = []
            cache = {}
            for j in range(nregs(x.shape)):
                lr, off = self.affine_reg(x, j)
                key = (id(lr), repr(off))
                if key in cache:
                    regs.append(cache[key])
                    continue
                v = self.new("af")
                if lr is None:
                    self.op("mov", v, self.operand(off))
                elif off.is_const and off.const == 0:
                    v = lr
                else:
                    self.op("add", v, lr, self.operand(off))
                cache[key] = v
                regs.append(v)
            return Dense(x.shape, x.dtype, regs)
        if isinstance(x, Scal):
            return self.pack(x)
        if isinstance(x, Pred):
            return self.pred_dense(x)
        raise self.error(f"cannot materialise {type(x).__name__}")

    def reg_of(self, d: Dense, j: int, acc=False) -> VReg:
        r = d.regs[j]
        if isinstance(r, Token):
            v = self.tmu.receive(r, acc)
            d.regs[j] = v
            return v
        return r

    # -- Scal <-> Dense ---------------------------------------------------------------

    def to_scal(self, x) -> Scal:
        if isinstance(x, Scal):
            return x
        if isinstance(x, Splat):
            return Scal(x.shape, x.dtype, [x.s] * numel(x.shape))
        if isinstance(x, S):
            return Scal((), None, [x])
        d = self.materialize(x)
        elems = []
        for e in range(numel(d.shape)):
            j, lane = divmod(e, LANES)
            if numel(d.shape) == 1 and False:
                pass
            elems.append(S(reg=self.lane_broadcast(self.reg_of(d, j, acc=True), lane),
                           is_float=d.dtype is not None and d.dtype.is_float))
        return Scal(d.shape, d.dtype, elems)

    def lane_broadcast(self, src: VReg, lane: int) -> VReg:
        if src.cls not in ("acc",):
            t = self.new("ba", cls="acc")
            self.op("mov", t, src)
            src = t
        r5 = VReg("r5", cls="r5")
        self.op("mov", r5, src, rot=(LANES - lane) % LANES or None, unit="mul")
        out = self.new("bc", src.is_float)
        self.op("mov", out, r5)
        return out

    def pack(self, s: Scal) -> Dense:
        n = numel(s.shape)
        regs = []
        for j in range(nregs(s.shape)):
            elems = s.elems[j * LANES:(j + 1) * LANES]
            if all(e is elems[0] or (e.is_const and elems[0].is_const and e.const == elems[0].const and
                                     e.reg is None) for e in elems) and len(elems) == LANES:
                regs.append(self.sreg(elems[0]))
                continue
            dst = self.new("pk", s.dtype is not None and s.dtype.is_float)
            groups = {}
            for lane, e in enumerate(elems):
                key = ("r", id(e.reg)) if e.reg is not None else ("c", e.const, e.is_float)
                groups.setdefault(key, (e, []))[1].append(lane)
            order = sorted(groups.values(), key=lambda g: -len(g[1]))
            self.op("mov", dst, self.operand(order[0][0]))
            for e, lanes in order[1:]:
                cond = self.lane_mask_flags(lanes)
                self.op("mov", dst, self.operand(e), cond=cond)
            regs.append(dst)
        return Dense(s.shape, s.dtype, regs)

    def lane_mask_flags(self, lanes) -> str:
        """Set flags so that the returned condition holds on exactly `lanes`:
        bit 31 - lane of the constant is shifted into the sign bit."""
        k = 0
        for lane in lanes:
            k |= 1 << (31 - lane)
        self.op("shl", isa.NOP, self.const_reg(k), isa.ELEM_NUM, setf=True, unit="add")
        return "ifn"

    def permute(self, srcs, shape, dtype) -> Dense:
        """Gather a tile: element f of the result is lane srcs[f][1] of register
        srcs[f][0] (None: don't care).  Each output register is built from
        rotated source registers with masked moves, so only the source and
        result registers are live (to_scal would hold one register per element)."""
        is_float = dtype is not None and dtype.is_float
        n = numel(shape)
        regs = []
        for J in range(nregs(shape)):
            groups = {}
            elems = set()
            for lane in range(LANES):
                f = J * LANES + lane
                if f >= n:
                    break
                if srcs[f] is None:
                    continue
                r, l = srcs[f]
                elems.add((id(r), l))
                groups.setdefault((id(r), (lane - l) % LANES), (r, []))[1].append(lane)
            if not groups:
                regs.append(self.sreg(C(0.0 if is_float else 0, is_float)))
                continue
            if len(elems) == 1 and len(groups) > 1:
                (r, l), = [srcs[f] for f in range(J * LANES, min(n, J * LANES + LANES))
                           if srcs[f] is not None][:1]
                regs.append(self.lane_broadcast(r, l))
                continue
            # group by source register so its accumulator copy is short-lived
            order = sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1][1])))
            first = max(range(len(order)), key=lambda i: len(order[i][1][1]))
            order = [order[first]] + order[:first] + order[first + 1:]
            dst = self.new("pm", is_float)
            accs = {}
            for i, ((rid, rot), (r, lanes)) in enumerate(order):
                src = r
                if rot:
                    if r.cls != "acc":
                        if rid not in accs:
                            t = self.new("pa", is_float, cls="acc")
                            self.op("mov", t, r)
                            accs[rid] = t
                        src = accs[rid]
                if i == 0:
                    self.op("mov", dst, src, rot=rot or None, unit="mul" if rot else "")
                else:
                    cond = self.lane_mask_flags(lanes)
                    self.op("mov", dst, src, rot=rot or None, unit="mul" if rot else "", cond=cond)
            regs.append(dst)
        return Dense(shape, dtype, regs)

    def permute_dense(self, d: Dense, shape, src_of) -> Dense:
        """permute() with sources given as flat element indices of d."""
        srcs = []
        for g in src_of:
            if g is None:
                srcs.append(None)
                continue
            j, l = divmod(g, LANES)
            srcs.append((self.reg_of(d, j), l))
        return self.permute(srcs, shape, d.dtype)

    def row_reg(self, d: Dense, r: int, n: int) -> VReg:
        """Register holding row r of a [*, n] tile (n < 16) in lanes 0..n-1."""
        j, l = divmod(r * n, LANES)
        src = self.reg_of(d, j)
        if l == 0:
            return src
        if src.cls != "acc":
            t = self.new("ra", src.is_float, cls="acc")
            self.op("mov", t, src)
            src = t
        out = self.new("row", src.is_float)
        self.op("mov", out, src, rot=LANES - l, unit="mul")
        return out

    # -- predicates ---------------------------------------------------------------

    def pred_elem_operands(self, x, j):
        if isinstance(x, S):
            return self.operand(x)
        if isinstance(x, Splat):
            return self.operand(x.s)
        if isinstance(x, Affine):
            lr, off = self.affine_reg(x, j)
            if lr is None:
                return self.operand(off)
            if off.is_const and off.const == 0:
                return lr
            v = self.new("pa")
            self.op("add", v, lr, self.operand(off))
            return v
        if isinstance(x, Scal):
            return self.reg_of(self.pack(x), j)
        d = self.materialize(x)
        return self.reg_of(d, j)

    def set_flags(self, p: Pred, j: int) -> str:
        """Emit flag computation for register j of predicate p; return the
        condition code that is true where p holds."""
        if p.kind == "const":
            return "always" if p.value else "never"
        if p.kind == "cmp":
            pred, a, b, is_float = p.value
            if is_float:
                return self.float_cmp_flags(pred, a, b, j)
            x = self.pred_elem_operands(a, j)
            y = self.pred_elem_operands(b, j)
            if pred in ("ult", "ule", "ugt", "uge"):
                x, y = self.flip_sign(x), self.flip_sign(y)
                pred = "s" + pred[1:]
            base = pred[-2:] if pred not in ("eq", "ne", "oeq", "une") else pred
            if base in ("gt", "le"):
                x, y = y, x
            op = "fsub" if is_float else "sub"
            if isinstance(x, Imm) and isinstance(y, Imm) and x.value != y.value:
                t = self.new("c")
                self.op("mov", t, x)
                x = t
            if p.wide and base in ("lt", "gt", "ge", "le"):
                # x - y may overflow; with x, y swapped for gt/le the question is
                # x < y (lt, gt) or x >= y (ge, le), i.e. min(x, y) != y or == y
                t = self.new("c")
                self.op("min", t, x, y)
                self.op("sub", isa.NOP, t, y, setf=True, unit="add")
                return "ifnz" if base in ("lt", "gt") else "ifz"
            self.op(op, isa.NOP, x, y, setf=True, unit="add")
            if base in ("lt", "gt"):
                return "ifn"
            if base in ("ge", "le"):
                return "ifnn"
            if base in ("eq", "oeq"):
                return "ifz"
            return "ifnz"
        m = self.pred_mask(p, j)
        self.op("mov", isa.NOP, m, setf=True, unit="add")
        return "ifnz"

    def float_cmp_flags(self, pred, a, b, j) -> str:
        """IEEE float compare (olt/ole/ogt/oge/oeq/une) of register j.

        fsub-based flags are wrong for NaN (its sign decides), inf - inf, -0 vs
        +0 and differences that flush to zero, so floats are compared as
        sign-magnitude integers: key = m for x >= 0 and -m for x < 0, where
        m = bits & 0x7FFFFFFF (+0 and -0 both get key 0), and x is NaN iff
        m > 0x7F800000."""
        ca, cb = _float_const(a), _float_const(b)
        if pred in ("ogt", "oge"):
            a, b, ca, cb = b, a, cb, ca
            pred = "olt" if pred == "ogt" else "ole"
        if (ca is not None and math.isnan(ca)) or (cb is not None and math.isnan(cb)):
            return "always" if pred == "une" else "never"
        # ==/!= against a normal constant: x - c is exact (or NaN / inf) there
        if pred in ("oeq", "une") and (ca is None) != (cb is None):
            c = ca if cb is None else cb
            if math.isfinite(c) and abs(c) >= 2.0 ** -100:
                x = self.pred_elem_operands(a if ca is None else b, j)
                self.op("fsub", isa.NOP, x, self.operand(C(c, True)), setf=True, unit="add")
                return "ifz" if pred == "oeq" else "ifnz"
        mags, keys = [], []
        for v, c in ((a, ca), (b, cb)):
            if c is not None:
                bits = int(np.float32(c).view(np.uint32))
                m = bits & 0x7FFFFFFF
                keys.append(C(-m if bits >> 31 else m))
                continue
            x = self.pred_elem_operands(v, j)
            m = self.new("fm")
            self.op("and", m, x, self.const_reg(0x7FFFFFFF))
            sg = self.new("fs")
            self.op("asr", sg, x, Imm(_shift_imm(31)))
            t = self.new("fk")
            self.op("xor", t, m, sg)
            k = self.new("fk")
            self.op("sub", k, t, sg)
            mags.append(m)
            keys.append(S(reg=k))
        if len(mags) == 2:
            mm = self.new("fm")
            self.op("max", mm, mags[0], mags[1])
        else:
            mm = mags[0]
        kx, ky = (self.operand(k) for k in keys)
        if pred == "olt":
            # min(kx, ky) != ky  <=>  kx < ky; masked with "ordered" (-1 / 0)
            o = self.new("fo")
            self.op("sub", o, mm, self.const_reg(0x7F800001))
            self.op("asr", o, o, Imm(_shift_imm(31)))
            t = self.new("fc")
            self.op("min", t, kx, ky)
            self.op("sub", t, t, ky)
            self.op("and", isa.NOP, t, o, setf=True, unit="add")
            return "ifnz"
        # unordered: 1 / 0
        u = self.new("fu")
        self.op("sub", u, self.const_reg(0x7F800000), mm)
        self.op("shr", u, u, Imm(_shift_imm(31)))
        t = self.new("fc")
        if pred == "ole":
            self.op("max", t, kx, ky)          # max(kx, ky) == ky  <=>  kx <= ky
            self.op("sub", t, t, ky)
        else:
            self.op("sub", t, kx, ky)
        self.op("or", isa.NOP, t, u, setf=True, unit="add")
        return "ifnz" if pred == "une" else "ifz"

    def flip_sign(self, x):
        v = self.new("u")
        self.op("xor", v, x, self.const_reg(0x80000000))
        return v

    def pred_mask(self, p: Pred, j: int) -> VReg:
        """Register j of p as 0 / nonzero."""
        if p.kind == "dense":
            return self.reg_of(p.value, j)
        if p.kind in ("and", "or"):
            ms = [self.pred_mask(a, j) for a in p.args]
            out = ms[0]
            for m in ms[1:]:
                n = self.new("pm")
                self.op("and" if p.kind == "and" else "or", n, out, m)
                out = n
            return out
        if p.kind == "not":
            m = self.pred_mask(p.args[0], j)
            out = self.new("pm")
            self.op("mov", out, Imm(1))
            self.op("mov", isa.NOP, m, setf=True, unit="add")
            self.op("mov", out, Imm(0), cond="ifnz")
            return out
        out = self.new("pm")
        self.op("mov", out, Imm(0))
        cond = self.set_flags(p, j)
        if cond == "always":
            self.op("mov", out, Imm(1))
        elif cond != "never":
            self.op("mov", out, Imm(1), cond=cond)
        return out

    def pred_dense(self, p: Pred) -> Dense:
        if p.kind == "dense":
            return p.value
        regs = [self.pred_mask(p, j) for j in range(nregs(p.shape))]
        return Dense(p.shape, ir.int1, regs)

    def as_pred(self, x) -> Pred:
        if isinstance(x, Pred):
            return x
        if isinstance(x, (S, Splat)) and (x.const if isinstance(x, S) else x.s.const) is not None:
            c = x.const if isinstance(x, S) else x.s.const
            return Pred("const", getattr(x, "shape", ()), bool(c))
        d = self.materialize(x)
        return Pred("dense", d.shape, d)

    # -- op dispatch ------------------------------------------------------------------

    def run(self):
        entry = self.const_block
        self.entry_uniform_insts = []
        self.arg_index = {}
        for i, (arg, name) in enumerate(zip(self.fn.args, self.fn.arg_names)):
            self.env[arg.id] = S(reg=self.uniform("arg", name), is_float=arg.dtype.is_float)
            self.arg_index[arg.id] = i
        self.late_plan = self.plan_late_args()
        self.pid_regs = {}
        self.pid_late = {}
        self.nprog_regs = {}
        self.body_start = len(self.mf.blocks)
        # uniforms for program ids are read lazily but appended to the ABI in a
        # fixed order after the arguments
        self.lazy_ok = set()
        self.lower_block(self.fn.body)
        self.tmu.drain()
        if self.vdw_pending or self.stats["dma_stores"]:
            self.op("mov", isa.NOP, isa.VW_WAIT)
        self.emit(MInst(ops=[], sig="thrend"))
        self.mf.blocks.extend(self.cold_blocks)
        entry.insts[0:0] = self.entry_uniform_insts
        return self.mf

    def program_id(self, axis) -> S:
        if axis not in self.pid_regs:
            self.pid_regs[axis] = self.uniform("pid", f"pid{axis}")
        if self.pid_late.get(axis):
            # re-read instead of keeping the id live across loops
            idx = self.uniforms.index(("pid", f"pid{axis}"))
            reg = self.reload_arg(ArgReload(idx)).reg
            if self.region_depth > 0:
                return S(reg=reg)
            self.pid_late[axis] = False
            self.pid_regs[axis] = reg
        return S(reg=self.pid_regs[axis])

    def _uses_pid(self, op) -> bool:
        for o in op.regions[0].ops if op.regions else []:
            if o.name in ("tt.get_program_id",):
                return True
        return False

    def num_programs(self, axis) -> S:
        if axis not in self.nprog_regs:
            self.nprog_regs[axis] = self.uniform("nprog", f"nprog{axis}")
        return S(reg=self.nprog_regs[axis])

    def get_zero_ptr(self) -> VReg:
        if self.zero_ptr is None:
            self.zero_ptr = self.uniform("zero", "zero_ptr")
        return self.zero_ptr

    def lower_block(self, block: ir.Block):
        for op in block.ops:
            if op.results and all(r.id in self.env for r in op.results):
                continue
            self.lower_op(op)
            if block is self.fn.body and op in self.late_plan:
                # drop arguments the rest of the kernel needs only later:
                # they are re-read from the uniform list on demand
                for arg in self.late_plan[op]:
                    self.env[arg.id] = ArgReload(self.arg_index[arg.id], arg.dtype.is_float)
                if not self._uses_pid(op):
                    for axis in list(self.pid_regs):
                        self.pid_late[axis] = True

    def plan_late_args(self) -> dict:
        """top-level loop op -> args unused inside the loop but used after it."""
        top = self.fn.body.ops
        plan = {}
        uses = {}
        for idx, op in enumerate(top):
            vals = set()

            def rec(o):
                for v in o.operands:
                    if v is not None:
                        vals.add(v.id)
                for r in o.regions:
                    for oo in r.ops:
                        rec(oo)
            rec(op)
            uses[idx] = vals
        for idx, op in enumerate(top):
            if op.name != "scf.for":
                continue
            late = []
            for arg in self.fn.args:
                if arg.id in uses[idx]:
                    continue
                if any(arg.id in uses[j] for j in range(idx + 1, len(top))):
                    late.append(arg)
            if late:
                plan[op] = late
        return plan

    def reload_arg(self, r: "ArgReload") -> S:
        if self.unif_base is None:
            self.unif_base = self.uniform("self", "unif_base")
        a = self.new("ua")
        self.op("add", a, self.unif_base, self.operand(C(4 * r.index)))
        self.op("mov", isa.UNIF_ADDR, a)
        v = self.new("arg", r.is_float)
        self.op("mov", v, isa.UNIF)
        return S(reg=v, is_float=r.is_float)

    def lower_op(self, op: ir.Op):
        n = op.name
        fn = getattr(self, "op_" + n.replace(".", "_"), None)
        if fn is None:
            raise self.error(f"unsupported op {n}")
        out = fn(op)
        if out is not None:
            if isinstance(out, tuple) and len(op.results) != 1:
                for r, v in zip(op.results, out):
                    self.env[r.id] = v
            else:
                self.env[op.results[0].id] = out

    def rng(self, v: Value):
        return self.ranges.get(v) if self.ranges is not None else None

    # simple ops
    def op_arith_constant(self, op):
        v = op.attrs["value"]
        r = op.result
        if r.dtype.is_bool:
            return Pred("const", (), bool(v))
        return C(v, r.dtype.is_float)

    def op_tt_get_program_id(self, op):
        return self.program_id(op.attrs["axis"])

    def op_tt_get_num_programs(self, op):
        return self.num_programs(op.attrs["axis"])

    def op_tt_make_range(self, op):
        return Affine(op.result.shape, ir.int32, C(op.attrs["start"]), [C(1)])

    def op_tt_splat(self, op):
        x = self.get(op.operands[0])
        shape = op.result.shape
        dt = op.result.dtype
        if isinstance(x, Pred):
            if x.kind == "const":
                return Pred("const", shape, x.value)
            return Pred("dense", shape, Dense(shape, ir.int1, [self.reg_of(self.pred_dense(x), 0)] * nregs(shape)))
        s = self.scalar(op.operands[0])
        if dt.is_int and not dt.is_bool or dt.is_ptr:
            return Affine(shape, dt, s, [C(0)] * len(shape))
        return Splat(shape, dt, s)

    def reshape_val(self, x, shape, insert_axis=None):
        if isinstance(x, Affine):
            if insert_axis is not None:
                coeffs = list(x.coeffs)
                coeffs.insert(insert_axis, C(0))
                dense = Dense(shape, x.dtype, x.dense.regs) if x.dense is not None else None
                return Affine(shape, x.dtype, x.base, coeffs, dense)
            return None
        if isinstance(x, Splat):
            return Splat(shape, x.dtype, x.s)
        if isinstance(x, S):
            return Splat(shape, None, x)
        if isinstance(x, Dense):
            return Dense(shape, x.dtype, x.regs)
        if isinstance(x, Scal):
            return Scal(shape, x.dtype, x.elems)
        if isinstance(x, Pred):
            if x.kind == "const":
                return Pred("const", shape, x.value)
            if x.kind == "cmp":
                pred, a, b, f = x.value
                return Pred("cmp", shape, wide=x.wide, value=(pred, self.reshape_val(a, shape, insert_axis) or self.materialize(a),
                                           self.reshape_val(b, shape, insert_axis) or self.materialize(b), f))
            if x.kind in ("and", "or", "not"):
                return Pred(x.kind, shape, None, [self.reshape_val(a, shape, insert_axis) for a in x.args])
            return Pred("dense", shape, self.reshape_val(x.value, shape))
        return None

    def op_tt_expand_dims(self, op):
        x = self.get(op.operands[0])
        return self.reshape_val(x, op.result.shape, op.attrs["axis"])

    def op_tt_reshape(self, op):
        x = self.get(op.operands[0])
        shape = op.result.shape
        src = op.operands[0].shape
        if isinstance(x, Affine):
            # only squeeze/unsqueeze of unit dims keeps an affine form
            if [s for s in src if s != 1] == [s for s in shape if s != 1] and x.dense is None:
                it = iter([c for s, c in zip(src, x.coeffs) if s != 1])
                coeffs = [C(0) if s == 1 else next(it) for s in shape]
                return Affine(shape, x.dtype, x.base, coeffs)
            x = self.materialize(x)
        out = self.reshape_val(x, shape)
        if out is None:
            raise self.error("unsupported reshape")
        return out

    def op_tt_broadcast(self, op):
        x = self.get(op.operands[0])
        return self.broadcast(x, op.operands[0].shape, op.result.shape)

    def broadcast(self, x, src, shape):
        if isinstance(x, Affine):
            coeffs = [C(0) if s == 1 and d != 1 else c for s, d, c in zip(src, shape, x.coeffs)]
            dense = None
            if x.dense is not None:
                dense = self.broadcast(x.dense, src, shape)
                if not isinstance(dense, Dense):
                    dense = self.materialize(dense)
            return Affine(shape, x.dtype, x.base, coeffs, dense)
        if isinstance(x, (Splat, S)):
            return Splat(shape, getattr(x, "dtype", None), x if isinstance(x, S) else x.s)
        if isinstance(x, Pred):
            if x.kind == "const":
                return Pred("const", shape, x.value)
            if x.kind == "cmp":
                pred, a, b, f = x.value
                return Pred("cmp", shape, (pred, self.broadcast(a, src, shape), self.broadcast(b, src, shape), f),
                            wide=x.wide)
            if x.kind in ("and", "or", "not"):
                return Pred(x.kind, shape, None, [self.broadcast(a, src, shape) for a in x.args])
            return Pred("dense", shape, self.broadcast(x.value, src, shape))
        if isinstance(x, Dense) and numel(src) >= LANES and src[-1] == shape[-1] and \
                (src[-1] >= LANES or numel(src[:-1]) == 1):
            # whole registers repeat: register (outer..., last) maps to a source register
            regs = []
            st_src = strides(src)
            for j in range(nregs(shape)):
                f0 = j * LANES
                idx = np.unravel_index(f0, shape)
                fs = sum((i if s_ != 1 else 0) * st for i, s_, st in zip(idx, src, st_src))
                regs.append(x.regs[fs // LANES])
            return Dense(shape, x.dtype, regs)
        st_src = strides(src)
        if not isinstance(x, (Scal, Splat, S)) and numel(src) > LANES:
            d = self.materialize(x)
            src_of = [sum((i if s_ != 1 else 0) * st for i, s_, st in zip(idx, src, st_src))
                      for idx in np.ndindex(*shape)]
            return self.permute_dense(d, shape, src_of)
        # data broadcast of a few elements: go through per-element scalars
        s = self.to_scal(x)
        elems = []
        for idx in np.ndindex(*shape):
            f = sum((i if s_ != 1 else 0) * st for i, s_, st in zip(idx, src, st_src))
            elems.append(s.elems[f])
        return Scal(shape, s.dtype, elems)

    def op_tt_trans(self, op):
        x = self.get(op.operands[0])
        m, n = op.operands[0].shape
        shape = op.result.shape
        if isinstance(x, Affine) and x.dense is None:
            return Affine(shape, x.dtype, x.base, [x.coeffs[1], x.coeffs[0]])
        if isinstance(x, (Splat, S)):
            return Splat(shape, op.result.dtype, x if isinstance(x, S) else x.s)
        if not isinstance(x, Scal):
            d = self.materialize(x)
            return self.permute_dense(d, shape, [i * n + j for j in range(n) for i in range(m)])
        s = self.to_scal(x)
        elems = [s.elems[i * n + j] for j in range(n) for i in range(m)]
        return Scal(shape, s.dtype, elems)

    # -- elementwise ------------------------------------------------------------------

    _INT_BIN = {"arith.addi": "add", "arith.subi": "sub", "arith.muli": "mul", "arith.andi": "and",
                "arith.ori": "or", "arith.xori": "xor", "arith.shli": "shl", "arith.shrsi": "shrs",
                "arith.shrui": "shru", "arith.minsi": "min", "arith.maxsi": "max",
                "arith.divsi": "divs", "arith.remsi": "rems", "arith.divui": "divu",
                "arith.remui": "remu", "arith.maxui": "maxu", "arith.minui": "minu"}
    _FLT_BIN = {"arith.addf": "add", "arith.subf": "sub", "arith.mulf": "mul", "arith.divf": "div",
                "arith.minimumf": "min", "arith.maximumf": "max"}

    def binary(self, op):
        n = op.name
        is_float = n in self._FLT_BIN
        kind = self._FLT_BIN.get(n) or self._INT_BIN.get(n)
        a_v, b_v = op.operands
        a, b = self.get(a_v), self.get(b_v)
        res = op.result
        if res.dtype.is_bool and kind in ("and", "or", "xor"):
            return self.pred_logic(kind, self.as_pred(a), self.as_pred(b), res.shape)
        ra, rb = self.rng(a_v), self.rng(b_v)
        if not res.shape:
            return self.s_bin(kind, self.scalar(a_v), self.scalar(b_v), is_float,
                              signed=n.endswith("si"), ra=ra, rb=rb)
        # affine algebra
        if not is_float:
            out = self.affine_binary(kind, a, b, res, ra, rb)
            if out is not None:
                return out
        if isinstance(a, Splat) and isinstance(b, Splat):
            return Splat(res.shape, res.dtype, self.s_bin(kind, a.s, b.s, is_float, ra=ra, rb=rb))
        # Scal operands that are broadcast along the lane dimension stay cheap
        da = self.materialize(a)
        db = self.materialize(b)
        regs = []
        seen = {}
        for j in range(nregs(res.shape)):
            x, y = self.reg_of(da, j), self.reg_of(db, j)
            key = (id(x), id(y))
            if key in seen:          # rows repeated by a broadcast
                regs.append(seen[key])
                continue
            dst = self.new("e", is_float)
            self.elementwise(kind, dst, x, y, is_float, ra=ra, rb=rb)
            seen[key] = dst
            regs.append(dst)
        return Dense(res.shape, res.dtype, regs)

    def dense_op(self, kind, x: Optional[Dense], y: Optional[Dense], shape, dtype):
        """Combine two optional dense terms register by register (with CSE)."""
        if x is None and y is None:
            return None
        if y is None:
            return x if kind == "add" else x
        if x is None:
            if kind == "add":
                return y
            regs = [self.cached("sub", Imm(0), self.reg_of(y, j)) for j in range(nregs(shape))]
            return Dense(shape, dtype, regs)
        regs = [self.cached(kind, self.reg_of(x, j), self.reg_of(y, j)) for j in range(nregs(shape))]
        return Dense(shape, dtype, regs)

    def affine_binary(self, kind, a, b, res, ra, rb):
        def as_aff(x):
            if isinstance(x, Affine):
                return x
            if isinstance(x, Splat):
                return Affine(res.shape, res.dtype, x.s, [C(0)] * len(res.shape))
            if isinstance(x, Dense) and kind in ("add", "sub") and len(x.regs) == nregs(res.shape):
                return Affine(res.shape, res.dtype, C(0), [C(0)] * len(res.shape), x)
            return None
        A, B = as_aff(a), as_aff(b)
        if A is None or B is None:
            return None
        if A.dense is not None and B.dense is not None and kind not in ("add", "sub"):
            return None
        if kind in ("add", "sub"):
            f = self.s_add if kind == "add" else self.s_sub
            dense = self.dense_op(kind, A.dense, B.dense, res.shape, res.dtype)
            return Affine(res.shape, res.dtype, f(A.base, B.base), [f(x, y) for x, y in zip(A.coeffs, B.coeffs)],
                          dense)
        if kind == "mul":
            if isinstance(a, Splat) or (all(c.is_const and c.const == 0 for c in A.coeffs) and A.dense is None):
                A, B = B, A
                ra, rb = rb, ra
            if all(c.is_const and c.const == 0 for c in B.coeffs) and B.dense is None:
                s = B.base
                dense = None
                if A.dense is not None:
                    sv = self.operand(s)
                    regs = [self.scaled(self.reg_of(A.dense, j), s, sv) for j in range(nregs(res.shape))]
                    dense = Dense(res.shape, res.dtype, regs)
                # ra bounds the whole tensor; with a dense term the base alone is not in it
                return Affine(res.shape, res.dtype, self.s_mul(A.base, s, ra if A.dense is None else None, rb),
                              [self.s_mul(c, s) for c in A.coeffs], dense)
        if A.dense is not None or B.dense is not None:
            return None             # only + - and * by a uniform keep a dense term
        if kind == "shl" and all(c.is_const and c.const == 0 for c in B.coeffs) and B.base.is_const \
                and 0 <= B.base.const < 31:
            k = 1 << B.base.const
            return Affine(res.shape, res.dtype, self.s_mul(A.base, C(k)), [self.s_mul(c, C(k)) for c in A.coeffs])
        if all(c.is_const and c.const == 0 for c in A.coeffs + B.coeffs):
            return Splat(res.shape, res.dtype, self.s_bin(kind, A.base, B.base, False, ra=ra, rb=rb))
        return None

    def scaled(self, v: VReg, s: S, sv) -> VReg:
        key = (self.cur.label, "scale", v.id, repr(s))
        if key in self.vcache:
            return self.vcache[key]
        if s.is_const and s.const == 1:
            return v
        dst = self.new("sc")
        self.imul(dst, v, sv, None, None, None, s if s.is_const else None)
        self.vcache[key] = dst
        return dst

    def s_sub(self, a: S, b: S) -> S:
        if b.is_const and b.const == 0:
            return a
        return self.s_bin("sub", a, b, False)

    for _n in list(_INT_BIN) + list(_FLT_BIN):
        locals()["op_" + _n.replace(".", "_")] = binary
    del _n

    def pred_logic(self, kind, a: Pred, b: Pred, shape):
        if kind == "and":
            if a.kind == "const":
                return b if a.value else a
            if b.kind == "const":
                return a if b.value else b
            return Pred("and", shape, None, [a, b])
        if kind == "or":
            if a.kind == "const":
                return a if a.value else b
            if b.kind == "const":
                return b if b.value else a
            return Pred("or", shape, None, [a, b])
        if b.kind == "const":
            return Pred("not", shape, None, [a]) if b.value else a
        if a.kind == "const":
            return Pred("not", shape, None, [b]) if a.value else b
        da, db = self.pred_dense(a), self.pred_dense(b)
        regs = []
        for j in range(nregs(shape)):
            x = self.new("px")
            self.op("xor", x, self.reg_of(da, j), self.reg_of(db, j))
            regs.append(x)
        return Pred("dense", shape, Dense(shape, ir.int1, regs))

    def op_arith_cmpi(self, op):
        a, b = self.get(op.operands[0]), self.get(op.operands[1])
        return self.cmp(op, a, b, False)

    def op_arith_cmpf(self, op):
        a, b = self.get(op.operands[0]), self.get(op.operands[1])
        return self.cmp(op, a, b, True)

    def cmp(self, op, a, b, is_float):
        pred = op.attrs["predicate"]
        shape = op.result.shape
        if not shape:
            sa, sb = self.scalar(op.operands[0]), self.scalar(op.operands[1])
            if sa.is_const and sb.is_const:
                return Pred("const", (), _cmp_const(pred, sa.const, sb.const))
            a, b = sa, sb
        if is_float:
            ca, cb = _float_const(a), _float_const(b)
            if any(c is not None and math.isnan(c) for c in (ca, cb)):
                return Pred("const", shape, pred == "une")
            if ca is not None and cb is not None:
                return Pred("const", shape, _cmp_const(pred, ca, cb))
        else:
            ua, ub = _uniform_scalar(a), _uniform_scalar(b)
            if ua is not None and ub is not None and ua.is_const and ub.is_const:
                return Pred("const", shape, _cmp_const(pred, ua.const, ub.const))
        ra, rb = self.rng(op.operands[0]), self.rng(op.operands[1])
        wide = not is_float and pred not in ("eq", "ne") and not _sub_cannot_overflow(ra, rb, pred[0] == "u")
        if wide and pred[0] == "s":
            # x in [lo, hi] against a uniform y: x < y <=> x < clamp(y, lo - 1, hi + 1)
            # (same for <=, >, >=), and then x - y cannot overflow
            for known, other, i in ((ra, b, 1), (rb, a, 0)):
                u = _uniform_scalar(other)
                if known is not None and u is not None and u.reg is not None and \
                        known[1] - known[0] < 2 ** 30:
                    c = self.clamp_scalar(u, max(known[0] - 1, -2 ** 31), min(known[1] + 1, 2 ** 31 - 1))
                    if isinstance(other, Splat):
                        c = Splat(other.shape, other.dtype, c)
                    elif isinstance(other, Affine):
                        c = Affine(other.shape, other.dtype, c, other.coeffs)
                    a, b = (a, c) if i == 1 else (c, b)
                    wide = False
                    break
        return Pred("cmp", shape, (pred, a, b, is_float), wide=wide)

    def clamp_scalar(self, x: S, lo: int, hi: int) -> S:
        key = ("clamp", id(x.reg), lo, hi, self.cur.label)
        if key not in self.consts:
            t = self.new("cl")
            self.op("max", t, x.reg, self.operand(C(lo)))
            self.op("min", t, t, self.operand(C(hi)))
            self.consts[key] = t
        return S(reg=self.consts[key])

    def op_arith_select(self, op):
        c = self.as_pred(self.get(op.operands[0]))
        x, y = self.get(op.operands[1]), self.get(op.operands[2])
        res = op.result
        if c.kind == "const":
            return x if c.value else y
        is_float = res.dtype.is_float
        if res.dtype.is_bool:
            x, y = self.pred_dense(self.as_pred(x)), self.pred_dense(self.as_pred(y))
        n = nregs(res.shape)
        dx, dy = self.materialize(x), self.materialize(y)
        regs = []
        for j in range(n):
            dst = self.new("sel", is_float)
            # resolve both operands first: a pending masked load sets flags itself
            xj = self.reg_of(dx, j if len(dx.regs) > 1 else 0)
            yj = self.reg_of(dy, j if len(dy.regs) > 1 else 0)
            self.op("mov", dst, yj)
            cond = self.set_flags(c, j)
            self.op("mov", dst, xj, cond=cond)
            regs.append(dst)
        if not res.shape:
            return S(reg=regs[0], is_float=is_float)
        if res.dtype.is_bool:
            return Pred("dense", res.shape, Dense(res.shape, ir.int1, regs))
        return Dense(res.shape, res.dtype, regs)

    def unary_regs(self, op, fn, is_float, const_fn=None):
        x = self.get(op.operands[0])
        res = op.result
        cs = x if isinstance(x, S) else (x.s if isinstance(x, Splat) else None)
        if const_fn is not None and cs is not None and cs.is_const:
            v = const_fn(cs.const)
            if v is not None:
                c = C(v, is_float)
                return c if not res.shape else Splat(res.shape, res.dtype, c)
        if isinstance(x, Scal) and numel(x.shape) > LANES:
            x = self.pack(x)
        if not res.shape:
            s = self.scalar(op.operands[0])
            dst = self.new("u", is_float)
            fn(dst, self.operand(s))
            return S(reg=dst, is_float=is_float)
        if isinstance(x, Splat):
            dst = self.new("u", is_float)
            fn(dst, self.operand(x.s))
            return Splat(res.shape, res.dtype, S(reg=dst, is_float=is_float))
        if isinstance(x, Scal):
            elems = []
            cache = {}
            for e in x.elems:
                k = id(e)
                if k not in cache:
                    dst = self.new("u", is_float)
                    fn(dst, self.operand(e))
                    cache[k] = S(reg=dst, is_float=is_float)
                elems.append(cache[k])
            return Scal(res.shape, res.dtype, elems)
        d = self.materialize(x)
        regs = []
        for j in range(len(d.regs)):
            dst = self.new("u", is_float)
            fn(dst, self.reg_of(d, j))
            regs.append(dst)
        return Dense(res.shape, res.dtype, regs)

    def op_arith_sitofp(self, op):
        return self.unary_regs(op, lambda d, x: self.op("itof", d, x), True,
                               lambda c: float(np.float32(_wrap(c))))

    def op_arith_uitofp(self, op):
        def f(d, x):
            # itof is signed: for values >= 2**31 convert (x >> 1 | x & 1) and double
            h, l, f2 = self.new("u"), self.new("u"), self.new("u", True)
            self.op("shr", h, x, Imm(1))
            self.op("and", l, x, Imm(1))
            self.op("or", h, h, l)
            self.op("itof", f2, h)
            self.op("fadd", f2, f2, f2)
            self.op("itof", d, x)
            self.op("mov", isa.NOP, x, setf=True, unit="add")
            self.op("mov", d, f2, cond="ifn")
        return self.unary_regs(op, f, True, lambda c: float(np.float32(int(c) & 0xFFFFFFFF)))

    def op_arith_fptosi(self, op):
        return self.unary_regs(op, lambda d, x: self.op("ftoi", d, x), False,
                               lambda c: int(np.trunc(np.clip(c, -2**31, 2**31 - 1))) if np.isfinite(c) else None)

    def op_arith_fptoui(self, op):
        def f(d, x):
            # ftoi is signed and saturates: values >= 2**31 go through x - 2**31
            two31 = self.const_reg(0x4F000000)
            g, gi = self.new("u", True), self.new("u")
            self.op("fsub", g, x, two31)
            self.op("ftoi", gi, g)
            self.op("xor", gi, gi, self.const_reg(0x80000000))
            self.op("ftoi", d, x)
            self.op("fsub", isa.NOP, x, two31, setf=True, unit="add")
            self.op("mov", d, gi, cond="ifnn")
        return self.unary_regs(op, f, False,
                               lambda c: int(np.trunc(c)) & 0xFFFFFFFF if np.isfinite(c) and c >= 0 else None)

    def op_arith_negf(self, op):
        # flip the sign bit: exact for -0.0 and NaN (0 - x gives +0 and NaN-as-inf)
        return self.unary_regs(op, lambda d, x: self.op("xor", d, x, self.const_reg(0x80000000)), True,
                               lambda c: -float(c))

    def op_math_absf(self, op):
        return self.unary_regs(op, lambda d, x: self.op("fmaxabs", d, x, x), True, lambda c: abs(float(c)))

    def op_math_absi(self, op):
        if op.result.dtype == ir.uint32:
            return self.get(op.operands[0])
        def f(d, x):
            n = self.new("n")
            self.op("sub", n, Imm(0), x)
            self.op("max", d, x, n)
        return self.unary_regs(op, f, False, lambda c: abs(_wrap(c)))

    # exp / log run on the ALUs: the SFU gives ~13 bits (tools/probes.py) and
    # its exact results are unknown, while these sequences are ~22 bits and
    # deterministic, so the emulator reproduces the hardware bit for bit.

    def fk(self, v: float):
        """Operand for a float constant (small immediate or constant register)."""
        return self.operand(C(float(np.float32(v)), True))

    def _exp_core(self, d, x, base2: bool):
        """d = 2**x (base2) or e**x: x = n*ln2 + r with |r| <= ln2/2, then a
        degree-6 polynomial for e**r scaled by 2**n built in the exponent bits."""
        f = lambda: self.new("ex", True)
        i = lambda: self.new("ex")
        xc = f()
        self.op("fmin", xc, x, self.fk(128.0 if base2 else 89.0))
        self.op("fmax", xc, xc, self.fk(-150.0 if base2 else -104.0))
        y = xc
        if not base2:
            y = f()
            self.op("fmul", y, xc, self.fk(1.4426950408889634))
        half = i()                                    # +-0.5 with the sign of y
        self.op("and", half, y, self.const_reg(0x80000000))
        self.op("or", half, half, self.const_reg(0x3F000000))
        t = f()
        self.op("fadd", t, y, half)
        n = i()
        self.op("ftoi", n, t)                         # round to nearest
        nf = f()
        self.op("itof", nf, n)
        r = f()
        if base2:
            self.op("fsub", r, xc, nf)                # exact
            self.op("fmul", r, r, self.fk(0.6931471805599453))
        else:
            t1, t2 = f(), f()
            self.op("fmul", t1, nf, self.fk(0.693145751953125))   # ln2 high part: exact product
            self.op("fsub", t2, xc, t1)
            self.op("fmul", t1, nf, self.fk(1.428606765330187e-06))
            self.op("fsub", r, t2, t1)
        p = f()
        self.op("fmul", p, r, self.fk(1.0 / 720))
        for c in (1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0):
            q = f()
            self.op("fadd", q, p, self.fk(c))
            p = f()
            self.op("fmul", p, q, r)
        q = f()
        self.op("fadd", q, p, self.fk(1.0))
        # q * 2**n with n clamped to [-127, 127] (2**-127 is built as 0.0); n = 128
        # (results near FLT_MAX) moves one factor 2 into q's exponent
        nc = i()
        self.op("max", nc, n, self.operand(C(-127)))
        self.op("min", nc, nc, self.operand(C(127)))
        c = i()
        self.op("sub", c, n, nc)
        self.op("shl", c, c, Imm(_shift_imm(23)))
        q2 = f()
        self.op("add", q2, q, c)
        self.op("add", nc, nc, self.operand(C(127)))
        self.op("shl", nc, nc, Imm(_shift_imm(23)))
        self.op("fmul", d, q2, nc)
        # overflow: round-toward-zero multiplies saturate at FLT_MAX, so set inf
        # explicitly (from 2**128, or e**x above the largest finite float32 result)
        if base2:
            self.op("fsub", isa.NOP, x, self.fk(128.0), setf=True, unit="add")
            self.op("mov", d, self.const_reg(0x7F800000), cond="ifnn")
        else:
            self.op("fsub", isa.NOP, self.const_reg(0x42B17217), x, setf=True, unit="add")
            self.op("mov", d, self.const_reg(0x7F800000), cond="ifn")
        self.keep_nan(d, x)

    def keep_nan(self, d, x):
        """d = x where x is NaN."""
        m = self.new("nan")
        self.op("and", m, x, self.const_reg(0x7FFFFFFF))
        self.op("sub", isa.NOP, m, self.const_reg(0x7F800001), setf=True, unit="add")
        self.op("mov", d, x, cond="ifnn")

    def _log_core(self, d, x, base2: bool):
        """x = 2**e * m with m in [sqrt(1/2), sqrt(2)); ln m = 2 atanh(s),
        s = (m - 1) / (m + 1), as an odd polynomial in s."""
        f = lambda: self.new("lg", True)
        i = lambda: self.new("lg")
        e = i()
        self.op("shr", e, x, Imm(_shift_imm(23)))
        self.op("and", e, e, self.const_reg(0xFF))
        mb = i()
        self.op("and", mb, x, self.const_reg(0x007FFFFF))
        self.op("or", mb, mb, self.const_reg(0x3F800000))  # m in [1, 2)
        self.op("sub", isa.NOP, mb, self.const_reg(0x3FB504F3), setf=True, unit="add")
        m = f()
        self.op("mov", m, mb)
        self.op("fmul", m, mb, Imm(_f2u(0.5)), cond="ifnn")
        self.op("sub", e, e, self.operand(C(126)), cond="ifnn")
        self.op("sub", e, e, self.operand(C(127)), cond="ifn")
        num, den = f(), f()
        self.op("fsub", num, m, self.fk(1.0))
        self.op("fadd", den, m, self.fk(1.0))
        # 1/den, den in [1.707, 2.415): linear minimax guess (1.5% error), two Newton steps
        t = f()
        self.op("fmul", t, den, self.fk(0.2390158))
        rd = f()
        self.op("fsub", rd, self.fk(0.9850641), t)
        for _ in range(2):
            u, v, w = f(), f(), f()
            self.op("fmul", u, den, rd)
            self.op("fsub", v, self.fk(2.0), u)
            self.op("fmul", w, rd, v)
            rd = w
        sv = f()
        self.op("fmul", sv, num, rd)
        s2 = f()
        self.op("fmul", s2, sv, sv)
        p = f()
        self.op("fmul", p, s2, self.fk(1.0 / 9))
        for c in (1.0 / 7, 1.0 / 5, 1.0 / 3):
            q = f()
            self.op("fadd", q, p, self.fk(c))
            p = f()
            self.op("fmul", p, q, s2)
        q = f()
        self.op("fadd", q, p, self.fk(1.0))
        lm = f()
        self.op("fmul", lm, q, sv)
        ef = f()
        self.op("itof", ef, e)
        if base2:
            self.op("fmul", lm, lm, self.fk(2.0 * 1.4426950408889634))
            self.op("fadd", d, lm, ef)
        else:
            self.op("fmul", lm, lm, Imm(_f2u(2.0)))
            self.op("fmul", ef, ef, self.fk(0.6931471805599453))
            self.op("fadd", d, lm, ef)
        # specials: inf / NaN -> x; x < 0 -> NaN; +-0 (and denormals, flushed) -> -inf
        eb = i()
        self.op("and", eb, x, self.const_reg(0x7F800000))
        self.op("sub", isa.NOP, eb, self.const_reg(0x7F800000), setf=True, unit="add")
        self.op("mov", d, x, cond="ifz")
        self.op("mov", isa.NOP, x, setf=True, unit="add")
        self.op("mov", d, self.const_reg(0x7FC00000), cond="ifn")
        self.op("mov", isa.NOP, eb, setf=True, unit="add")
        self.op("mov", d, self.const_reg(0xFF800000), cond="ifz")

    def _alu_unary(self, op, core, base2):
        def fn(d, x):
            if isinstance(x, Imm) or (isinstance(x, VReg) and x.cls == "r4"):
                t = self.new("p", True)
                self.op("mov", t, x)
                x = t
            core(d, x, base2)
        return self.unary_regs(op, fn, True, const_fn=lambda c: _math_const(op.name, c))

    def op_math_exp2(self, op):
        return self._alu_unary(op, self._exp_core, True)

    def op_math_exp(self, op):
        return self._alu_unary(op, self._exp_core, False)

    def op_math_log2(self, op):
        return self._alu_unary(op, self._log_core, True)

    def op_math_log(self, op):
        return self._alu_unary(op, self._log_core, False)

    def op_math_rsqrt(self, op):
        def f(d, x):
            if isinstance(x, Imm):
                t = self.new("p", True)
                self.op("mov", t, x)
                x = t
            self.op("mov", d, self.rsqrt_core(x))
            eb = self.new("ex")
            self.op("and", eb, x, self.const_reg(0x7F800000))
            # +inf -> 0; x < 0 -> NaN; +-0 (and denormals, flushed) -> +-inf; NaN -> NaN
            self.op("sub", isa.NOP, eb, self.const_reg(0x7F800000), setf=True, unit="add")
            self.op("mov", d, Imm(0), cond="ifz")
            self.op("mov", isa.NOP, x, setf=True, unit="add")
            self.op("mov", d, self.const_reg(0x7FC00000), cond="ifn")
            si = self.new("ex")
            self.op("and", si, x, self.const_reg(0x80000000))
            self.op("or", si, si, self.const_reg(0x7F800000))
            self.op("mov", isa.NOP, eb, setf=True, unit="add")
            self.op("mov", d, si, cond="ifz")
            self.keep_nan(d, x)
        return self.unary_regs(op, f, True)

    def op_math_sqrt(self, op):
        def f(d, x):
            if isinstance(x, Imm):
                t = self.new("p", True)
                self.op("mov", t, x)
                x = t
            self.op("fmul", d, x, self.rsqrt_core(x))
            eb = self.new("ex")
            self.op("and", eb, x, self.const_reg(0x7F800000))
            # inf / NaN -> x; x < 0 -> NaN; +-0 (and denormals, flushed) -> +-0
            self.op("sub", isa.NOP, eb, self.const_reg(0x7F800000), setf=True, unit="add")
            self.op("mov", d, x, cond="ifz")
            self.op("mov", isa.NOP, x, setf=True, unit="add")
            self.op("mov", d, self.const_reg(0x7FC00000), cond="ifn")
            sz = self.new("ex")
            self.op("and", sz, x, self.const_reg(0x80000000))
            self.op("mov", isa.NOP, eb, setf=True, unit="add")
            self.op("mov", d, sz, cond="ifz")
        return self.unary_regs(op, f, True)

    def op_arith_bitcast(self, op):
        x = self.get(op.operands[0])
        return self.retype(x, op.result)

    def retype(self, x, res):
        if isinstance(x, Splat):
            s = self.retype(x.s, res)
            return Splat(x.shape, res.dtype, s)
        if isinstance(x, S):
            if x.is_const and x.is_float != res.dtype.is_float:
                # reinterpret the constant's bits (float constants are python floats)
                bits = _f2u(x.const) if x.is_float else int(x.const) & 0xFFFFFFFF
                if res.dtype.is_float:
                    f = float(np.array(bits, np.uint32).view(np.float32))
                    if math.isnan(f) and _f2u(f) != bits:
                        return S(reg=self.const_reg(bits), is_float=True)   # keep a NaN's payload
                    return C(f, True)
                return C(_wrap(bits))
            return S(const=x.const, reg=x.reg, is_float=res.dtype.is_float)
        if isinstance(x, Affine):
            return Affine(x.shape, res.dtype, x.base, x.coeffs, x.dense)
        if isinstance(x, Dense):
            return Dense(x.shape, res.dtype, x.regs)
        if isinstance(x, Scal):
            return Scal(x.shape, res.dtype, x.elems)
        return x

    op_tt_bitcast = op_arith_bitcast
    op_tt_ptr_to_int = op_arith_bitcast
    op_tt_int_to_ptr = op_arith_bitcast

    def op_arith_extui(self, op):
        p = self.as_pred(self.get(op.operands[0]))
        if p.kind == "const":
            c = C(int(p.value))
            return c if not op.result.shape else Splat(op.result.shape, op.result.dtype, c)
        d = self.pred_dense(p)
        if not op.result.shape:
            return S(reg=self.reg_of(d, 0))
        regs = []
        for j in range(len(d.regs)):
            out = self.new("ext")
            self.op("mov", out, Imm(0))
            self.op("mov", isa.NOP, self.reg_of(d, j), setf=True, unit="add")
            self.op("mov", out, Imm(1), cond="ifnz")
            regs.append(out)
        return Dense(op.result.shape, op.result.dtype, regs)

    def op_tt_addptr(self, op):
        p, off = self.get(op.operands[0]), self.get(op.operands[1])
        res = op.result
        esize = 4
        if not res.shape:
            sp, so = self.scalar(op.operands[0]), self.scalar(op.operands[1])
            return self.s_add(sp, self.s_mul(so, C(esize)))
        # scale offset by element size, then add
        if isinstance(off, Dense) and len(off.regs) == nregs(res.shape):
            off = Affine(res.shape, ir.int32, C(0), [C(0)] * len(res.shape), off)
        if isinstance(off, (Affine, Splat)):
            off_a = off if isinstance(off, Affine) else Affine(res.shape, ir.int32, off.s, [C(0)] * len(res.shape))
            dense = None
            if off_a.dense is not None:
                regs = [self.scaled(self.reg_of(off_a.dense, j), C(esize), Imm(2)) for j in range(nregs(res.shape))]
                dense = Dense(res.shape, ir.int32, regs)
            scaled = Affine(res.shape, ir.int32, self.s_mul(off_a.base, C(esize)),
                            [self.s_mul(c, C(esize)) for c in off_a.coeffs], dense)
            pa = p if isinstance(p, Affine) else (Affine(res.shape, res.dtype, p.s, [C(0)] * len(res.shape))
                                                 if isinstance(p, Splat) else None)
            if pa is not None:
                return Affine(res.shape, res.dtype, self.s_add(pa.base, scaled.base),
                              [self.s_add(x, y) for x, y in zip(pa.coeffs, scaled.coeffs)],
                              self.dense_op("add", pa.dense, scaled.dense, res.shape, res.dtype))
            off = self.materialize(scaled)
            dp = self.materialize(p)
            regs = [self.cached("add", self.reg_of(dp, j), self.reg_of(off, j)) for j in range(nregs(res.shape))]
            return Dense(res.shape, res.dtype, regs)
        dp, do = self.materialize(p), self.materialize(off)
        regs = []
        for j in range(nregs(res.shape)):
            o = self.reg_of(do, j)
            if not isinstance(off, Affine):
                so = self.new("off")
                self.op("shl", so, o, Imm(2))
                o = so
            dst = self.new("ptr")
            self.op("add", dst, self.reg_of(dp, j), o)
            regs.append(dst)
        return Dense(res.shape, res.dtype, regs)

    # -- loads ------------------------------------------------------------------------

    def load_tokens(self, ptr, mask: Optional[Pred], other, shape, dtype, stream=None, entry0=0):
        """Issue TMU requests for a (shape) tensor of pointers."""
        n = nregs(shape)
        tokens = []
        need_mask = mask is not None and not (mask.kind == "const" and mask.value)
        if mask is not None and mask.kind == "const" and not mask.value:
            o = other if other is not None else Splat(shape, dtype, C(0, dtype.is_float))
            return self.materialize(o) if not isinstance(o, (Splat, S)) else o
        other_d = None
        other_zero = other is None or _is_zero(other)      # masked lanes read the zero page
        if need_mask and not other_zero:
            other_d = self.materialize(other)
        for j in range(n):
            emit = self.addr_emitter(ptr, mask if need_mask else None, j)
            if stream is not None:
                tok = stream.token(entry0 + j, emit)
            else:
                tok = self.tmu.issue(emit)
                self.stats["tmu_requests"] += 1
            if other_d is not None:
                tok.post = self.make_other_fixup(mask, other_d, j)
            tokens.append(tok)
        return Dense(shape, dtype, tokens)

    def addr_emitter(self, ptr, mask, j):
        """Closure emitting the address of register j into the TMU register."""
        def emit(tmu_reg):
            self.stats["tmu_requests"] += 1
            if isinstance(ptr, (S, Splat)):
                s = ptr if isinstance(ptr, S) else ptr.s
                base, off, lr = s, C(0), None
            elif isinstance(ptr, Affine):
                lr, off = self.affine_reg(ptr, j)
                base = C(0)
            else:
                d = ptr
                lr, off, base = self.reg_of(d, j), C(0), C(0)
            if mask is None:
                if lr is None:
                    self.op("mov", tmu_reg, self.operand(self.s_add(base, off)))
                elif off.is_const and off.const == 0:
                    self.op("mov", tmu_reg, lr)
                else:
                    self.op("add", tmu_reg, lr, self.operand(off))
                return
            a = self.new("addr")
            if lr is None:
                self.op("mov", a, self.operand(self.s_add(base, off)))
            elif off.is_const and off.const == 0:
                self.op("mov", a, lr)
            else:
                self.op("add", a, lr, self.operand(off))
            cond = self.set_flags(mask, j)
            if cond != "always":
                self.op("mov", a, self.get_zero_ptr(), cond=isa.COND_INVERT[cond])
            self.op("mov", tmu_reg, a)
        return emit

    def make_other_fixup(self, mask, other_d, j):
        def post(v):
            o = self.reg_of(other_d, j if len(other_d.regs) > 1 else 0)
            cond = self.set_flags(mask, j)
            if cond != "always":
                self.op("mov", v, o, cond=isa.COND_INVERT[cond])
            return v
        return post

    def op_tt_load(self, op):
        ptr = self.get(op.operands[0])
        mask = self.as_pred(self.get(op.operands[1])) if op.attrs.get("has_mask") else None
        other = self.get(op.operands[2]) if op.attrs.get("has_other") else None
        res = op.result
        if not res.shape and mask is not None and mask.kind == "const" and not mask.value:
            return self.scalar(op.operands[2]) if other is not None else C(0.0 if res.dtype.is_float else 0,
                                                                          res.dtype.is_float)
        if not res.shape:
            emit = self.addr_emitter(ptr, None if mask is None or mask.kind == "const" else mask, 0)
            tok = self.tmu.issue(emit)
            v = self.tmu.receive(tok)
            if mask is not None and mask.kind != "const" and other is not None:
                cond = self.set_flags(mask, 0)
                self.op("mov", v, self.operand(self.scalar(op.operands[2])), cond=isa.COND_INVERT[cond])
            return S(reg=v, is_float=res.dtype.is_float)
        stream = self.pipelined.get(op) if getattr(self, "pipelined", None) else None
        if stream is not None:
            return stream.dense_for(op)
        return self.load_tokens(ptr, mask, other, res.shape, res.dtype)

    # -- stores -----------------------------------------------------------------------

    def vpm_bases(self):
        if self.vpm_setup_base is None or self.vpm_block is not self.cur:
            self.vpm_block = self.cur
            saved = self.cur
            row = self.new("vrow")
            self.op("shl", row, isa.QPU_NUM, Imm(2))         # 4 VPM rows per QPU
            self.vpm_setup_base = self.new("vws")
            self.op("add", self.vpm_setup_base, row, self.const_reg(0x00001A00))   # h32, stride 1
            r7 = self.new("vrow7")
            self.op("shl", r7, row, Imm(7))
            self.vdw_base = self.new("vdw")
            self.op("add", self.vdw_base, r7, self.const_reg(0x80004000))      # VDW horiz at (row, 0)
            self.cur = saved
        return self.vpm_setup_base, self.vdw_base

    def op_tt_store(self, op):
        ptr = self.get(op.operands[0])
        val = self.get(op.operands[1])
        mask = self.as_pred(self.get(op.operands[2])) if op.attrs.get("has_mask") else None
        if mask is not None and mask.kind == "const":
            if not mask.value:
                return None
            mask = None
        shape = op.operands[0].shape
        self.stats["dma_stores"] += 1
        return self._store(op, ptr, val, mask, shape)

    def stride_word(self, stride: S):
        """VDW stride-setup word (computed outside the critical section)."""
        if stride.is_const:
            return self.operand(C(0xC0000000 | (int(stride.const) & 0xFFFF)))
        v = self.new("vs")
        self.op("and", v, stride.reg, self.const_reg(0xFFFF))
        v2 = self.new("vs")
        self.op("or", v2, v, self.const_reg(0xC0000000))
        return v2

    def dma_kick(self, setup, stride: S, addr):
        """Start a VDW DMA store.

        The VDW stride register is one register shared by all 12 QPUs
        (tools/probes.py vdw_stride: written once, it gets overwritten by
        other QPUs), so basic setup + stride + start address are written
        under the hardware mutex.  The DMA latches them when it starts, so
        the mutex is released right away (probe: early release is safe) and
        this QPU waits for its own DMA later, outside the lock.
        """
        word = self.stride_word(stride)
        if self.opts.get("store_mutex", True):
            self.op("mov", isa.NOP, isa.MUTEX)
        self.op("mov", isa.VW_SETUP, setup)
        self.op("mov", isa.VW_SETUP, word)
        self.op("mov", isa.VW_ADDR, addr)
        if self.opts.get("store_mutex", True):
            self.op("mov", isa.MUTEX, Imm(0))
        self.vdw_pending = True

    def _store(self, op, ptr, val, mask, shape):
        if not shape:
            self.store_scalar(ptr, val, mask)
            return None
        layout = self.store_layout(ptr, shape)
        if layout is None:
            self.store_scatter(ptr, val, mask, shape)
            return None
        bounds = self.mask_bounds(mask, shape) if mask is not None else None
        if mask is not None and bounds is None:
            self.store_scatter(ptr, val, mask, shape)
            return None
        dv = self.materialize(val)
        for j in range(len(dv.regs)):
            self.reg_of(dv, j)
        if layout["kind"] == "2d" and layout["R"] > 1 and not layout["row"].is_const:
            # one check of the run-time row stride against the VDW stride field,
            # then a fast version and a rare one-DMA-per-row version
            self.row_gaps_fit_flags(layout["row"])
            slow_label = f"store_rows{self.blocks}"
            done_label = f"store_rows_done{self.blocks + 1}"
            self.blocks += 2
            pending = self.vdw_pending
            self.emit(MInst(kind="branch", bcond="alln", target=slow_label))
            self.cur = self.new_block("store_fast")
            self.row_gap_fits = True
            self._store_versions(layout, dv, bounds, shape)
            done = mir.Block(done_label, loop_depth=self.loop_depth)
            self.add_block(done)
            saved = self.begin_cold(slow_label)
            self.vdw_pending = pending
            self.row_gap_fits = False
            self.store_chunks(layout, dv, bounds if bounds is not None else {})
            self.row_gap_fits = None
            self.end_cold(saved, done_label)
            self.cur = done
            self.vdw_pending = True
            return None
        self._store_versions(layout, dv, bounds, shape)
        return None

    def _store_versions(self, layout, dv, bounds, shape):
        if bounds is None:
            self.store_chunks(layout, dv, None)
            return None
        # versioned: unmasked fast path when the whole tile is in bounds
        self.emit(self.bounds_all_valid(bounds, shape))
        slow_label = f"store_part{self.blocks}"
        done_label = f"store_done{self.blocks + 1}"
        self.blocks += 2
        pending = self.vdw_pending
        self.emit(MInst(kind="branch", bcond="allnz", target=slow_label))
        self.cur = self.new_block("store_full")
        self.store_chunks(layout, dv, None)
        self.dma_wait()
        self.emit(MInst(kind="branch", bcond="always", target=done_label))
        slow = mir.Block(slow_label, loop_depth=self.loop_depth)
        self.add_block(slow)
        self.cur = slow
        self.vdw_pending = pending          # the branch skipped the full path's wait
        self.vdw_stride = None
        self.store_chunks(layout, dv, bounds)
        done = mir.Block(done_label, loop_depth=self.loop_depth)
        self.add_block(done)
        self.cur = done
        self.vdw_pending = True
        self.vdw_stride = None
        return None

    def store_scalar(self, ptr, val, mask):
        s = ptr if isinstance(ptr, S) else self.scalar_of(ptr)
        v = val if isinstance(val, S) else self.scalar_of(val)
        vws, vdw = self.vpm_bases()
        self.dma_wait()
        self.op("mov", isa.VW_SETUP, vws)
        self.op("mov", isa.VPM, self.operand(v))
        setup = self.new("vd")
        self.op("add", setup, vdw, self.const_reg((1 << 23) | (1 << 16)))
        if mask is not None:
            cond = self.set_flags(mask, 0)
            a = self.new("a")
            self.op("mov", a, self.operand(s))
            if cond != "always":
                self.op("mov", a, self.get_scratch_ptr(), cond=isa.COND_INVERT[cond])
        else:
            a = self.operand(s)
        self.dma_kick(setup, C(0), a)

    def scalar_of(self, x) -> S:
        if isinstance(x, S):
            return x
        if isinstance(x, Splat):
            return x.s
        if isinstance(x, Affine):
            return x.base
        d = self.materialize(x)
        return S(reg=self.reg_of(d, 0))

    def get_scratch_ptr(self) -> VReg:
        if self.scratch_ptr is None:
            self.scratch_ptr = self.uniform("scratch", "scratch_ptr")
        return self.scratch_ptr

    def dma_wait(self):
        if self.vdw_pending:
            self.op("mov", isa.NOP, isa.VW_WAIT)
            self.vdw_pending = False

    def store_layout(self, ptr, shape):
        """Describe a contiguous-rows store: list of (rows, row_stride S, chunks).

        Returns None if the pointer tensor is not affine with unit stride on
        the last dimension (then a scatter store is used).
        """
        if not isinstance(ptr, Affine) or ptr.dense is not None:
            return None
        L = shape[-1]
        c_last = ptr.coeffs[-1]
        if numel(shape) < LANES:
            return None
        if len(shape) == 1:
            if not (c_last.is_const and c_last.const == 4):
                return None
            return {"kind": "1d", "ptr": ptr, "rows": numel(shape) // LANES}
        if L < LANES or not (c_last.is_const and c_last.const == 4):
            return None
        if len(shape) > 2:
            # collapse leading dims if they are a contiguous row walk
            for d in range(len(shape) - 2):
                need = self.s_mul(ptr.coeffs[d + 1], C(shape[d + 1]))
                c = ptr.coeffs[d]
                if not (c.is_const and need.is_const and c.const == need.const):
                    return None
        return {"kind": "2d", "ptr": ptr, "R": numel(shape[:-1]), "L": L, "row": ptr.coeffs[-2]
                if len(shape) == 2 else ptr.coeffs[-2]}

    def store_chunks(self, layout, dv: Dense, bounds):
        vws, vdw = self.vpm_bases()
        ptr = layout["ptr"]
        if layout["kind"] == "1d":
            nrows = layout["rows"]
            groups = [(list(range(g, min(g + VPM_ROWS_PER_QPU, nrows))), C(0)) for g in
                      range(0, nrows, VPM_ROWS_PER_QPU)]
            row_stride_bytes = C(64)
        else:
            R, L = layout["R"], layout["L"]
            nb = L // LANES
            groups = []
            for cb in range(nb):
                for r0 in range(0, R, VPM_ROWS_PER_QPU):
                    regs = [r * nb + cb for r in range(r0, min(r0 + VPM_ROWS_PER_QPU, R))]
                    groups.append((regs, C(cb)))
            row_stride_bytes = layout["row"]
        for regs, cb in groups:
            self.dma_wait()
            self.op("mov", isa.VW_SETUP, vws)
            for j in regs:
                self.op("mov", isa.VPM, self.reg_of(dv, j))
            # address of the first register of the chunk
            lr, off = self.affine_reg(ptr, regs[0])
            a = self.new("sa")
            if lr is None:
                self.op("mov", a, self.operand(off))
            elif off.is_const and off.const == 0:
                a = lr
            else:
                self.op("add", a, lr, self.operand(off))
            units = len(regs)
            if bounds is None:
                self.dma_rows_any(vdw, units, C(16), a, row_stride_bytes if layout["kind"] == "2d"
                                  else C(64), units)
                continue
            self.store_masked_chunk(layout, bounds, regs, a, vdw, row_stride_bytes)

    def bounds_all_valid(self, bounds, shape) -> MInst:
        """Compute flags: Z set when the whole tile is in bounds; return the setf inst."""
        conds = []
        for d, lim in bounds.items():
            # valid iff lim >= shape[d]  (lim = number of valid indices along d)
            t = self.new("bv")
            self.op("sub", t, self.operand(lim), self.operand(C(shape[d])))
            conds.append(t)
        acc = conds[0]
        for c in conds[1:]:
            n = self.new("bv")
            self.op("min", n, acc, c)
            acc = n
        # acc >= 0 -> all valid; produce Z flag = "all valid"
        z = self.new("bv")
        self.op("asr", z, acc, Imm(_shift_imm(31)))      # 0 if valid, -1 otherwise
        inst = alu("mov", isa.NOP, z, setf=True, unit="add")
        return inst

    def mask_bounds(self, mask: Pred, shape):
        """Turn `mask` into {dim: number of valid leading indices} or None."""
        terms = []

        def collect(p):
            if p.kind == "and":
                for a in p.args:
                    if not collect(a):
                        return False
                return True
            if p.kind == "cmp":
                terms.append(p)
                return True
            return False

        if len(shape) > 2 or not collect(mask):
            return None
        bounds = {}
        for t in terms:
            pred, a, b, is_float = t.value
            if is_float:
                return None
            if pred not in ("slt", "sle"):
                if pred in ("sgt", "sge"):
                    a, b = b, a
                    pred = "slt" if pred == "sgt" else "sle"
                else:
                    return None
            if not isinstance(a, Affine) or not isinstance(b, (Splat, S, Affine)):
                return None
            if a.dense is not None:
                return None
            if isinstance(b, Affine):
                if b.dense is not None or any(not (c.is_const and c.const == 0) for c in b.coeffs):
                    return None
                bs = b.base
            else:
                bs = b.s if isinstance(b, Splat) else b
            dims = [d for d, c in enumerate(a.coeffs) if not (c.is_const and c.const == 0)]
            if len(dims) != 1:
                return None
            d = dims[0]
            if not (a.coeffs[d].is_const and a.coeffs[d].const == 1):
                return None
            # base + i < b  <=>  i < b - base, as a count in [0, extent] (b - base and
            # b - base + 1 can exceed 32 bits even when the compare itself cannot overflow)
            lim = self.bound_count(bs, a.base, shape[d], pred == "sle")
            if d in bounds:
                lim = self.s_bin("min", bounds[d], lim, False)
            bounds[d] = lim
        if len(shape) == 1:
            return {0: bounds.get(0)} if 0 in bounds else None
        out = {}
        last = len(shape) - 1
        for d, lim in bounds.items():
            if d != last and d != last - 1:
                return None
            out[d] = lim
        return out

    def bound_count(self, b: S, base: S, ext: int, le: bool) -> S:
        """#{i in [0, ext): base + i < b} (<= b if le) without 32-bit overflow:
        b - base is read as unsigned where b > base (b >= base)."""
        if b.is_const and base.is_const:
            n = int(b.const) - int(base.const) + (1 if le else 0)
            return C(max(0, min(n, ext)))
        k = ext - 1 if le else ext
        d, t, m = self.new("bc"), self.new("bc"), self.new("bc")
        self.op("sub", d, self.operand(b), self.operand(base))
        self.op("min", t, d, self.operand(C(k)))
        self.op("mov", isa.NOP, d, setf=True, unit="add")
        self.op("mov", t, self.operand(C(k)), cond="ifn")       # unsigned min(d, k)
        if le:
            self.op("add", t, t, Imm(1))
        self.op("min", m, self.operand(b), self.operand(base))
        if le:   # min(b, base) == base  <=>  b >= base
            self.op("sub", isa.NOP, m, self.operand(base), setf=True, unit="add")
            self.op("mov", t, Imm(0), cond="ifnz")
        else:    # min(b, base) == b  <=>  b <= base
            self.op("sub", isa.NOP, m, self.operand(b), setf=True, unit="add")
            self.op("mov", t, Imm(0), cond="ifz")
        return S(reg=t)

    def store_masked_chunk(self, layout, bounds, regs, addr, vdw, row_stride_bytes):
        """DMA a chunk with a runtime number of valid rows / columns."""
        if layout["kind"] == "1d":
            lim = bounds[0]
            first = regs[0] * LANES
            cnt = self.new("cnt")
            self.op("sub", cnt, self.operand(lim), self.operand(C(first)))
            self.op("min", cnt, cnt, self.operand(C(len(regs) * LANES)))
            self.op("max", cnt, cnt, Imm(0))
            full = self.new("full")
            self.op("shr", full, cnt, Imm(4))
            rem = self.new("rem")
            self.op("and", rem, cnt, Imm(15))
            # full rows
            self.guarded_dma(full, lambda: self._dma_rows(vdw, full, C(16), addr, C(0)))
            # remainder row at VPM row `full`, memory addr + 64*full
            def tail():
                self.dma_wait()
                setup = self.new("vd")
                sh = self.new("t")
                self.op("shl", sh, full, Imm(7))
                self.op("add", setup, vdw, sh)
                d16 = self.new("t")
                self.op("shl", d16, rem, Imm(_shift_imm(16)))
                self.op("add", setup, setup, d16)
                self.op("add", setup, setup, self.const_reg(1 << 23))
                a2 = self.new("t")
                o = self.new("t")
                self.op("shl", o, full, Imm(6))
                self.op("add", a2, addr, o)
                self.dma_kick(setup, C(0), a2)
            self.guarded_dma(rem, tail)
            return
        nb = layout["L"] // LANES
        R = layout["R"]
        r0, cb = divmod(regs[0], nb)
        rows = len(regs)
        row_lim = bounds.get(0)
        col_lim = bounds.get(1)
        vr = self.new("vr")
        if row_lim is not None:
            self.op("sub", vr, self.operand(row_lim), self.operand(C(r0)))
            self.op("min", vr, vr, self.operand(C(rows)))
            self.op("max", vr, vr, Imm(0))
        else:
            self.op("mov", vr, self.operand(C(rows)))
        vc = self.new("vc")
        if col_lim is not None:
            t = self.new("t")
            self.op("sub", t, self.operand(col_lim), self.operand(C(cb * LANES)))
            self.op("min", t, t, self.operand(C(16)))
            self.op("max", vc, t, Imm(0))
        else:
            self.op("mov", vc, self.operand(C(16)))
        ok = self.new("ok")
        self.op("min", ok, vr, vc)

        def body():
            self.dma_rows_any(vdw, vr if rows > 1 else 1, S(reg=vc), addr, row_stride_bytes, rows)
        self.guarded_dma(ok, body)

    def _dma_rows(self, vdw, units, depth: S, addr, stride: S):
        self.dma_wait()
        setup = self.new("vd")
        if isinstance(units, int) and depth.is_const:
            self.op("add", setup, vdw, self.const_reg((units << 23) | (int(depth.const) << 16)))
            return self.dma_kick(setup, stride, addr)
        u = self.new("t")
        if isinstance(units, int):
            self.op("mov", u, self.const_reg(units << 23))
        else:
            self.op("shl", u, units if isinstance(units, VReg) else self.sreg(units), Imm(_shift_imm(23)))
        self.op("add", setup, vdw, u)
        if depth.is_const:
            self.op("add", setup, setup, self.const_reg(int(depth.const) << 16))
        else:
            d = self.new("t")
            self.op("shl", d, depth.reg, Imm(_shift_imm(16)))
            self.op("add", setup, setup, d)
        self.dma_kick(setup, stride, addr)

    def dma_rows_any(self, vdw, units, depth: S, addr, row_bytes: S, max_units: int):
        """DMA `units` (int or register, <= max_units) VPM rows of `depth` words to
        memory rows `row_bytes` apart.  The VDW stride field holds the gap between
        rows in 16 unsigned bits; when the gap does not fit (huge or negative row
        strides, e.g. rows stored in reverse order) every row gets its own DMA."""
        if isinstance(units, int) and units == 1:
            return self._dma_rows(vdw, 1, depth, addr, C(0))
        if row_bytes.is_const:
            row = int(row_bytes.const)
            dmin, dmax = (int(depth.const),) * 2 if depth.is_const else (1, LANES)
            if 0 <= row - 4 * dmax and row - 4 * dmin <= 0xFFFF:
                return self._dma_rows(vdw, units, depth, addr, self.row_gap(row_bytes, depth))
            return self._dma_each_row(vdw, units, depth, addr, row_bytes, max_units)
        if self.row_gap_fits is None:
            raise self.error("internal: runtime row stride outside a checked store")
        if self.row_gap_fits:
            return self._dma_rows(vdw, units, depth, addr, self.row_gap(row_bytes, depth))
        return self._dma_row_loop(vdw, units, depth, addr, row_bytes)

    def row_gap(self, row_bytes: S, depth: S) -> S:
        """VDW stride: bytes between the end of one row and the start of the next."""
        if depth.is_const:
            return self.s_sub(row_bytes, C(4 * int(depth.const)))
        d4 = self.new("t")
        self.op("shl", d4, depth.reg, Imm(2))
        g = self.new("gap")
        self.op("sub", g, self.operand(row_bytes), d4)
        return S(reg=g)

    def row_gaps_fit_flags(self, row_bytes: S):
        """N flag set (all lanes) when some row gap of a store with this runtime
        row stride (gaps row - 64 .. row - 4) does not fit 16 unsigned bits."""
        x = self.new("t")
        self.op("sub", x, row_bytes.reg, self.const_reg(64))
        y = self.new("t")
        self.op("sub", y, self.const_reg(0x10000 - 64), x)
        self.op("or", isa.NOP, x, y, setf=True, unit="add")

    def _dma_each_row(self, vdw, units, depth: S, addr, row_bytes: S, max_units: int):
        """One single-row DMA per VPM row, unrolled (any constant row stride)."""
        a = addr
        for r in range(max_units):
            if r:
                a2 = self.new("ra")
                self.op("add", a2, a, self.operand(row_bytes))
                a = a2
            row_vdw = vdw
            if r:
                row_vdw = self.new("vd")
                self.op("add", row_vdw, vdw, self.const_reg(r << 7))
            if isinstance(units, int):
                if r < units:
                    self._dma_rows(row_vdw, 1, depth, a, C(0))
                continue
            cnt = self.new("cnt")
            self.op("sub", cnt, units, self.operand(C(r)))
            self.op("max", cnt, cnt, Imm(0))
            self.guarded_dma(cnt, lambda rv=row_vdw, aa=a: self._dma_rows(rv, 1, depth, aa, C(0)))

    def _dma_row_loop(self, vdw, units, depth: S, addr, row_bytes: S):
        """One single-row DMA per VPM row in a run-time loop (units >= 1)."""
        cnt = self.new("rc")
        self.op("mov", cnt, units if isinstance(units, VReg) else self.operand(C(units)))
        setup = self.new("vd")
        if depth.is_const:
            self.op("add", setup, vdw, self.const_reg((1 << 23) | (int(depth.const) << 16)))
        else:
            d = self.new("t")
            self.op("shl", d, depth.reg, Imm(_shift_imm(16)))
            self.op("add", d, d, self.const_reg(1 << 23))
            self.op("add", setup, vdw, d)
        a = self.new("ra")
        self.op("mov", a, addr)
        self.loop_depth += 1
        loop = self.new_block("rowdma")
        self.cur = loop
        self.vdw_pending = True
        self.dma_wait()
        self.dma_kick(setup, C(0), a)
        self.op("add", setup, setup, self.const_reg(1 << 7))
        self.op("add", a, a, self.operand(row_bytes))
        self.op("sub", cnt, cnt, Imm(1), setf=True, unit="add")
        self.emit(MInst(kind="branch", bcond="anynz", target=loop.label))
        self.loop_depth -= 1
        self.cur = self.new_block("rowdma_end")
        self.vdw_pending = True

    def guarded_dma(self, count: VReg, body):
        """Run `body` only if count > 0 (uniform branch)."""
        skip = mir.Block(f"skip{self.blocks}", loop_depth=self.loop_depth)
        self.blocks += 1
        self.op("mov", isa.NOP, count, setf=True, unit="add")
        self.emit(MInst(kind="branch", bcond="allz", target=skip.label))
        blk = self.new_block("dma")
        self.cur = blk
        # whether an earlier guarded DMA ran is only known at run time
        self.vdw_pending = self.vdw_pending or self.stats["dma_stores"] > 0
        body()
        self.add_block(skip)
        self.cur = skip
        self.vdw_pending = True
        self.vdw_stride = None

    def store_scatter(self, ptr, val, mask, shape):
        """Element-by-element DMA (slow path for non-contiguous stores)."""
        vws, vdw = self.vpm_bases()
        dv = self.materialize(val)
        n = numel(shape)
        for j in range(nregs(shape)):
            addr = self.pred_elem_operands(ptr, j)       # one register at a time
            if not isinstance(addr, VReg):
                t = self.new("a")
                self.op("mov", t, addr)
                addr = t
            if mask is not None:
                a = self.new("a")
                self.op("mov", a, addr)
                cond = self.set_flags(mask, j)
                if cond != "always":
                    self.op("mov", a, self.get_scratch_ptr(), cond=isa.COND_INVERT[cond])
                addr = a
            acc_addr = self.new("aa", cls="acc")
            self.op("mov", acc_addr, addr)
            self.dma_wait()
            self.op("mov", isa.VW_SETUP, vws)
            self.op("mov", isa.VPM, self.reg_of(dv, j))
            for lane in range(min(LANES, n - j * LANES)):
                self.dma_wait()
                setup = self.new("vd")
                self.op("add", setup, vdw, self.const_reg((1 << 23) | (1 << 16) | (lane << 3)))
                if lane == 0:
                    a = acc_addr
                else:
                    a = self.new("al")
                    self.op("mov", a, acc_addr, rot=(LANES - lane) % LANES, unit="mul")
                self.dma_kick(setup, C(0), a)

    # -- dot --------------------------------------------------------------------------

    def op_tt_dot(self, op):
        a_v, b_v, c_v = op.operands
        M, K = a_v.shape
        K2, N = b_v.shape
        a, b, c = self.get(a_v), self.get(b_v), self.get(c_v)
        if N % LANES:
            if LANES % N:
                raise self.error(f"tl.dot: N = {N} must divide 16 or be a multiple of 16")
            return self.dot_small(a, b, c, M, N, K, op.result)
        NV = N // LANES
        da = self.materialize(a)
        db = self.materialize(b)
        # accumulator registers (updated in place when the input dies here)
        if isinstance(c, (Splat, S)) or (isinstance(c, Dense) and len(c.regs) == 1 and M * NV > 1):
            init = c.s if isinstance(c, Splat) else (c if isinstance(c, S) else S(reg=c.regs[0]))
            acc = []
            for _ in range(M * NV):
                v = self.new("acc", True)
                self.op("mov", v, self.operand(init))
                acc.append(v)
        else:
            dc = self.materialize(c)
            if self.use_counts.get(c_v.id, 0) > 1:
                acc = []
                for j in range(M * NV):
                    v = self.new("acc", True)
                    self.op("mov", v, self.reg_of(dc, j))
                    acc.append(v)
            else:
                acc = [self.reg_of(dc, j) for j in range(M * NV)]
                if len({id(r) for r in acc}) != len(acc):
                    # e.g. a broadcast accumulator: rows share registers
                    acc2 = []
                    for r in acc:
                        v = self.new("acc", True)
                        self.op("mov", v, r)
                        acc2.append(v)
                    acc = acc2
        n_a = nregs((M, K))
        a_acc_all = n_a <= 2
        a_regs = {}

        def a_src(vi):
            if vi in a_regs:
                return a_regs[vi]
            r = da.regs[vi]
            if isinstance(r, Token):
                v = self.tmu.receive(r, acc=True)
                da.regs[vi] = v
                if v.cls != "acc":
                    t = self.new("aa", True, cls="acc")
                    self.op("mov", t, v)
                    v = t
            else:
                t = self.new("aa", True, cls="acc")
                self.op("mov", t, r)
                v = t
            if a_acc_all:
                a_regs[vi] = v
            return v

        pending = None          # (acc reg, product reg) waiting for its fadd

        def b_reg(k, cb):
            return self.reg_of(db, k * NV + cb)

        # receive the first B row up front
        for cb in range(NV):
            b_reg(0, cb)
        cur_vi, cur_src = None, None
        for k in range(K):
            for m in range(M):
                f = m * K + k
                vi, lane = divmod(f, LANES)
                if vi != cur_vi:
                    cur_vi, cur_src = vi, a_src(vi)
                r5 = VReg("r5", cls="r5", is_float=True)
                self.emit(MInst(ops=[MOp("mov", r5, [cur_src], rot=(LANES - lane) % LANES or None,
                                         unit="mul")], tag="bcast"))
                for cb in range(NV):
                    p = self.new("prod", True, cls="acc")
                    mul = MOp("fmul", p, [r5, b_reg(k, cb)])
                    ops = [mul]
                    if pending is not None:
                        ops.insert(0, MOp("fadd", pending[0], [pending[0], pending[1]]))
                    self.emit(MInst(ops=ops, atomic=len(ops) == 2, tag="mac"))
                    pending = (acc[m * NV + cb], p)
                # prefetch the next B row while the remaining rows run
                if k + 1 < K and m < NV:
                    b_reg(k + 1, m)
            if k + 1 < K:
                for cb in range(NV):
                    b_reg(k + 1, cb)
        if pending is not None:
            self.op("fadd", pending[0], pending[0], pending[1])
        return Dense(op.result.shape, ir.float32, acc)

    def dot_small(self, a, b, c, M, N, K, res):
        """tl.dot with N < 16: one register per output row (lanes 0..N-1),
        acc_m += a[m, k] (lane broadcast) * row k of b."""
        da, db = self.materialize(a), self.materialize(b)
        dc = None if isinstance(c, (Splat, S)) else self.materialize(c)
        init = c.s if isinstance(c, Splat) else c
        brows = [self.row_reg(db, k, N) for k in range(K)] if K <= 8 else None
        rows = []
        for m in range(M):
            acc = self.new("dacc", True)
            self.op("mov", acc, self.operand(init) if dc is None else self.row_reg(dc, m, N))
            a_acc = {}
            for k in range(K):
                j, lane = divmod(m * K + k, LANES)
                if j not in a_acc:
                    r = self.reg_of(da, j)
                    if r.cls != "acc":
                        t = self.new("da", True, cls="acc")
                        self.op("mov", t, r)
                        r = t
                    a_acc = {j: r}
                s = self.lane_broadcast(a_acc[j], lane)
                br = brows[k] if brows is not None else self.row_reg(db, k, N)
                t = self.new("dm", True)
                self.op("fmul", t, s, br)
                self.op("fadd", acc, acc, t)
            rows.append(acc)
        return self.permute([(rows[f // N], f % N) for f in range(M * N)], (M, N), ir.float32)

    # -- reductions ---------------------------------------------------------------------

    def op_tt_reduce(self, op):
        x = self.get(op.operands[0])
        axis = op.attrs["axis"]
        kind = op.attrs["kind"]
        src = op.operands[0].shape
        res = op.result
        is_float = res.dtype.is_float
        opname = {"sum": "add", "max": "max", "min": "min"}[kind]
        if res.dtype.kind == "uint" and kind != "sum":
            opname += "u"
        if isinstance(x, Splat) and kind in ("max", "min"):
            return x.s if not res.shape else Splat(res.shape, res.dtype, x.s)
        rank = len(src)
        L = src[-1]
        if isinstance(x, Scal):
            return self.reduce_scal(x, axis, opname, is_float, res)
        d = self.materialize(x)
        if axis == rank - 1 and L >= LANES:
            nb = L // LANES
            rows = numel(src[:-1])
            elems = []
            for r in range(rows):
                cur = self.reg_of(d, r * nb)
                for cbi in range(1, nb):
                    n = self.new("rd", is_float)
                    self.elementwise(opname, n, cur, self.reg_of(d, r * nb + cbi), is_float)
                    cur = n
                elems.append(S(reg=self.lane_reduce(cur, opname, is_float), is_float=is_float))
            if not res.shape:
                return elems[0]
            return Scal(res.shape, res.dtype, elems)
        if axis < rank - 1 and numel(src[axis + 1:]) % LANES == 0:
            # combine whole registers
            outer = numel(src[:axis])
            red = src[axis]
            inner = numel(src[axis + 1:]) // LANES
            regs = []
            for o in range(outer):
                for i in range(inner):
                    cur = self.reg_of(d, (o * red) * inner + i)
                    for r in range(1, red):
                        n = self.new("rd", is_float)
                        self.elementwise(opname, n, cur, self.reg_of(d, (o * red + r) * inner + i), is_float)
                        cur = n
                    regs.append(cur)
            return Dense(res.shape, res.dtype, regs)
        if rank == 1 and L < LANES or (rank == 1):
            # sub-register 1D: mask unused lanes with the identity, then lane-reduce
            cur = self.reg_of(d, 0)
            for j in range(1, len(d.regs)):
                n = self.new("rd", is_float)
                self.elementwise(opname, n, cur, self.reg_of(d, j), is_float)
                cur = n
            if L < LANES:
                if is_float and kind in ("max", "min"):
                    # fmin / fmax order +NaN above +inf and -NaN below -inf
                    pad = self.const_reg(0x7FFFFFFF if kind == "min" else 0xFFFFFFFF)
                else:
                    pad = self.operand(self.identity(kind, is_float, unsigned=opname.endswith("u")))
                t = self.new("rd", is_float)
                self.op("mov", t, cur)
                self.op("sub", isa.NOP, isa.ELEM_NUM, self.operand(C(L)), setf=True, unit="add")
                self.op("mov", t, pad, cond="ifnn")
                cur = t
            return S(reg=self.lane_reduce(cur, opname, is_float), is_float=is_float)
        inner, red, outer = numel(src[axis + 1:]), src[axis], numel(src[:axis])
        span = red * inner
        if LANES % inner == 0 and (span % LANES == 0 or LANES % span == 0) and red & (red - 1) == 0:
            return self.reduce_small(d, src, axis, opname, is_float, res)
        return self.reduce_scal(self.to_scal(d), axis, opname, is_float, res)

    def lane_fold(self, v: VReg, s_hi: int, s_lo: int, opname, is_float) -> VReg:
        """v[i] = op(v[i], v[i + s]) for s = s_hi, s_hi/2, ..., s_lo: lane i ends
        up combining lanes i + k * s_lo (within a segment of 2 * s_hi lanes)."""
        cur = v
        s = s_hi
        while s >= s_lo:
            src = cur
            if src.cls != "acc":
                t = self.new("lf", is_float, cls="acc")
                self.op("mov", t, src)
                src = t
            rot = self.new("lf", is_float)
            self.op("mov", rot, src, rot=LANES - s, unit="mul")
            n = self.new("lf", is_float)
            self.elementwise(opname, n, src, rot, is_float)
            cur = n
            s //= 2
        return cur

    def reduce_small(self, d: Dense, src, axis, opname, is_float, res):
        """Reduce a tile whose reduced group (red elements `inner` apart) fits in
        a register: combine whole registers, fold lanes, then gather the results."""
        inner, red, outer = numel(src[axis + 1:]), src[axis], numel(src[:axis])
        span = red * inner
        at = []                                   # (register, lane) of result (o, 0)
        if span >= LANES:
            per = span // LANES
            for o in range(outer):
                cur = self.reg_of(d, o * per)
                for r in range(1, per):
                    n = self.new("rd", is_float)
                    self.elementwise(opname, n, cur, self.reg_of(d, o * per + r), is_float)
                    cur = n
                at.append((self.lane_fold(cur, LANES // 2, inner, opname, is_float), 0))
        else:
            per_reg = LANES // span
            for j in range(len(d.regs)):
                cur = self.lane_fold(self.reg_of(d, j), span // 2, inner, opname, is_float)
                for b in range(per_reg):
                    if j * per_reg + b < outer:
                        at.append((cur, b * span))
        srcs = [(at[f // inner][0], at[f // inner][1] + f % inner) for f in range(outer * inner)]
        return self.permute(srcs, res.shape, res.dtype)

    def identity(self, kind, is_float, unsigned=False) -> S:
        if kind == "sum":
            return C(0.0 if is_float else 0, is_float)
        if unsigned:
            return C(0 if kind == "max" else -1)
        if kind == "max":
            return C(float("-inf"), True) if is_float else C(-2 ** 31)
        return C(float("inf"), True) if is_float else C(2 ** 31 - 1)

    def lane_reduce(self, v: VReg, opname, is_float) -> VReg:
        """Butterfly reduction across the 16 lanes (result in every lane)."""
        cur = v
        for s in (8, 4, 2, 1):
            src = cur
            if src.cls != "acc":
                t = self.new("lr", is_float, cls="acc")
                self.op("mov", t, src)
                src = t
            rot = self.new("lr", is_float)
            self.op("mov", rot, src, rot=s, unit="mul")
            # intermediate results are rotation sources (accumulators); the
            # final one is an ordinary value
            n = self.new("lr", is_float, cls="acc" if s > 1 or is_float else "any")
            self.elementwise(opname, n, src, rot, is_float)
            cur = n
        if is_float:
            # fmin / fmax ties (+0 / -0) and inf + -inf depend on the operand order, so
            # lanes can end with different bits: use lane 0's everywhere
            cur = self.lane_broadcast(cur, 0)
        return cur

    def reduce_scal(self, x: Scal, axis, opname, is_float, res):
        src = x.shape
        st = strides(src)
        out_shape = res.shape
        elems = []
        for idx in np.ndindex(*out_shape) if out_shape else [()]:
            full = list(idx)
            full.insert(axis, 0)
            cur = None
            for r in range(src[axis]):
                full[axis] = r
                f = sum(i * s for i, s in zip(full, st))
                e = x.elems[f]
                if cur is None:
                    cur = e
                else:
                    cur = self.s_bin(opname, cur, e, is_float)
            elems.append(cur)
        if not out_shape:
            return elems[0]
        return Scal(out_shape, res.dtype, elems)

    # -- control flow -----------------------------------------------------------------

    def op_tt_return(self, op):
        return None

    def op_scf_yield(self, op):
        return None

    def carried_form(self, init, shape, dtype):
        """Loop-carried representation: (value used inside, list of (reg, source) copies)."""
        if isinstance(init, Affine):
            b = self.new("cb")
            dense = None
            if init.dense is not None:
                dense = Dense(init.dense.shape, init.dense.dtype,
                              [self.reg_of(init.dense, j) for j in range(len(init.dense.regs))])
            return Affine(shape, dtype, S(reg=b), init.coeffs, dense), [(b, init.base)], "affine"
        if isinstance(init, S):
            r = self.new("cs", init.is_float)
            return S(reg=r, is_float=init.is_float), [(r, init)], "scalar"
        if isinstance(init, Splat) and not shape:
            r = self.new("cs", init.s.is_float)
            return S(reg=r, is_float=init.s.is_float), [(r, init.s)], "scalar"
        if isinstance(init, Pred):
            init = self.pred_dense(init)
            d = init
            regs = [self.new("cd") for _ in d.regs]
            return Pred("dense", shape, Dense(shape, ir.int1, regs)), \
                [(r, S(reg=self.reg_of(d, j))) for j, r in enumerate(regs)], "pred"
        d = self.materialize(init)
        isf = dtype.is_float
        if isinstance(init, Splat):
            # e.g. accumulator = tl.zeros(...)
            regs = [self.new("cd", isf) for _ in range(nregs(shape))]
            return Dense(shape, dtype, regs), [(r, init.s) for r in regs], "dense"
        regs = [self.new("cd", isf) for _ in d.regs]
        return Dense(shape, dtype, regs), [(r, S(reg=self.reg_of(d, j))) for j, r in enumerate(regs)], "dense"

    def copy_into(self, dst: VReg, src: S):
        if src.reg is dst:
            return
        self.op("mov", dst, self.operand(src))

    def carried_moves(self, form, kind, newval):
        """(carried register, source) pairs that make `form` hold `newval`."""
        if kind == "affine":
            if isinstance(newval, Affine) and _same_coeffs(newval.coeffs, form.coeffs) \
                    and _same_dense(newval.dense, form.dense):
                return [(form.base.reg, newval.base)]
            # _affine_carry_ok proved newval = form + a uniform increment, however the
            # increment was lowered (e.g. a compare turned into a dense register), so
            # register 0 of newval - form holds the increment in every lane
            n0 = self.reg_of(self.materialize(newval), 0)
            o0 = self.reg_of(self.materialize(form), 0)
            d, nb = self.new("ci"), self.new("cb")
            self.op("sub", d, n0, o0)
            self.op("add", nb, form.base.reg, d)
            return [(form.base.reg, S(reg=nb))]
        if kind == "scalar":
            return [(form.reg, newval if isinstance(newval, S) else self.scalar_of(newval))]
        if kind == "pred":
            d = self.pred_dense(self.as_pred(newval))
            return [(r, S(reg=self.reg_of(d, j))) for j, r in enumerate(form.value.regs)]
        d = self.materialize(newval)
        return [(r, S(reg=self.reg_of(d, j if len(d.regs) > 1 else 0))) for j, r in enumerate(form.regs)]

    def parallel_copy(self, moves):
        """Emit dst <- src for all moves as if simultaneous: loop-carried values
        may be permuted (`a, b = b, a`), so sources that are themselves
        overwritten are staged in temporaries first."""
        moves = [(d, src) for d, src in moves if src.reg is not d]
        dsts = {d for d, _ in moves}
        staged = []
        for d, src in moves:
            if src.reg is not None and src.reg in dsts:
                t = self.new("pc", src.reg.is_float)
                self.op("mov", t, src.reg)
                src = S(reg=t, is_float=src.is_float)
            staged.append((d, src))
        for d, src in staged:
            self.op("mov", d, self.operand(src))

    def _uniform_value(self, v) -> bool:
        """True if every element of v is the same value."""
        if not v.shape:
            return True
        op = v.op
        if op is None:
            return False
        if op.name in ("tt.splat", "arith.constant"):
            return True
        if op.name in ("tt.broadcast", "tt.expand_dims", "tt.reshape", "arith.bitcast"):
            return self._uniform_value(op.operands[0])
        if op.name.startswith("arith.") and op.name not in ("arith.select",):
            return all(o is not None and self._uniform_value(o) for o in op.operands)
        return False

    def _affine_carry_ok(self, arg, v) -> bool:
        """The yielded value is `arg` plus/minus uniform increments, so an affine
        loop-carried value only needs its scalar base carried."""
        if v is arg:
            return True
        op = v.op
        if op is None or op.name not in ("tt.addptr", "arith.addi", "arith.subi"):
            return False
        a, b = op.operands
        if self._affine_carry_ok(arg, a) and self._uniform_value(b):
            return True
        return op.name != "arith.subi" and self._affine_carry_ok(arg, b) and self._uniform_value(a)

    def op_scf_for(self, op):
        lb, ub, step = (self.scalar(o) for o in op.operands[:3])
        inits = [self.get(v) for v in op.operands[3:]]
        body = op.regions[0]
        yield_op = body.ops[-1]
        forms = []
        for i, (init, arg) in enumerate(zip(inits, body.args[1:])):
            if isinstance(init, Affine) and not self._affine_carry_ok(arg, yield_op.operands[i]):
                init = self.materialize(init)
                inits[i] = init
            forms.append(self.carried_form(init, arg.shape, arg.dtype))
        res = self._lower_loop(op, lb, ub, step, inits, forms)
        return tuple(res) if len(op.results) != 1 else res[0]

    def _lower_loop(self, op, lb, ub, step, inits, forms):
        body = op.regions[0]
        has_store = any(o.name == "tt.store" for o in _walk_ops(body))
        self.tmu.drain()
        self.dma_wait()
        for (form, copies, kind) in forms:
            for reg, src in copies:
                self.copy_into(reg, src)
        iv = self.new("iv")
        self.op("mov", iv, self.operand(lb))
        # zero-trip guard (skipped when the range is provably non-empty)
        lb_r = self.rng(op.operands[0])
        ub_r = self.rng(op.operands[1])
        known_nonempty = lb_r is not None and ub_r is not None and lb_r[1] < ub_r[0]
        exit_label = f"loop_exit{self.blocks}"
        # software pipelining of loads
        self.lazy_ok = set(o for o in body.ops if o.name not in ("tt.load", "tt.store", "tt.dot",
                                                                    "scf.for", "scf.if"))
        stream = self.plan_pipeline(op, forms)
        self.pipelined = {}
        # iv < ub as one subtraction unless iv - ub can overflow (then min(iv, ub) != ub)
        st_r = self.rng(op.operands[2])
        iv_r = None
        if lb_r is not None and ub_r is not None and st_r is not None and st_r[0] > 0:
            iv_r = (lb_r[0], max(lb_r[1], ub_r[1] - 1 + st_r[1]))
        wide = self.loop_cmp_wide = not _sub_cannot_overflow(iv_r, ub_r, False)
        # iv + step can wrap past INT_MAX unless the ranges say otherwise
        sat = self.iv_saturate = iv_r is None or iv_r[1] > 2 ** 31 - 1
        if not known_nonempty:
            c = self.loop_lt_flags(iv, ub, wide=wide)
            self.emit(MInst(kind="branch", bcond="allnn" if c == "ifn" else "allz", target=exit_label))
            self.cur = self.new_block("loop_pre")
        if stream is not None:
            stream.wide = wide
            stream.sat = sat
            stream.prologue(lb, inits)     # after the zero-trip guard: no speculative loads
        self.loop_depth += 1
        self.region_depth += 1
        loop = self.new_block("loop")
        self.cur = loop
        # a store of the previous iteration may still be running
        self.vdw_pending = has_store
        for arg, (form, copies, kind) in zip(body.args[1:], forms):
            self.env[arg.id] = form
        self.env[body.args[0].id] = S(reg=iv)
        if stream is not None:
            stream.begin_body(iv, step, ub, forms)
            for ld in stream.loads:
                self.pipelined[ld] = stream
            self.tmu.stream = stream
        self.lower_block(body)
        if stream is not None:
            stream.end_body()
            self.tmu.stream = None
        else:
            self.tmu.drain()
        yield_op = body.ops[-1]
        moves = []
        for v, (form, copies, kind) in zip(yield_op.operands, forms):
            moves += self.carried_moves(form, kind, self.get(v))
        self.parallel_copy(moves)
        self.iv_step(iv, iv, step, sat)
        c = self.loop_lt_flags(iv, ub, wide=wide)      # (an inner loop changed self.loop_cmp_wide)
        self.emit(MInst(kind="branch", bcond="anyn" if c == "ifn" else "anynz", target=loop.label))
        self.loop_depth -= 1
        self.region_depth -= 1
        if stream is not None:
            # the last prefetches (clamped to this iteration) are received on the way
            # out of the loop, not on the zero-trip path, which issued none
            self.cur = self.new_block("loop_end")
            self.tmu.drain()
        exit_blk = mir.Block(exit_label, loop_depth=self.loop_depth)
        self.blocks += 1
        self.add_block(exit_blk)
        self.cur = exit_blk
        self.vdw_pending = self.vdw_pending or has_store
        self.pipelined = {}
        self.lazy_ok = set()
        return [form for form, _, _ in forms]

    def iv_step(self, dst, iv, step, saturate):
        """dst = iv + step (step > 0); with saturate, a sum past INT_MAX gives INT_MAX
        (which is never < ub, so the loop ends instead of wrapping around)."""
        if not saturate:
            self.op("add", dst, iv, self.operand(step))
            return
        n, t = self.new("ivs"), self.new("ivs")
        self.op("add", n, iv, self.operand(step))
        self.op("min", t, n, iv)
        self.op("sub", isa.NOP, t, iv, setf=True, unit="add")       # n < iv: wrapped
        self.op("mov", n, self.const_reg(0x7FFFFFFF), cond="ifnz")
        self.op("mov", dst, n)

    def loop_lt_flags(self, iv, ub, wide=None) -> str:
        """Set flags for iv < ub; returns the condition that holds when it is true."""
        if wide is None:
            wide = self.loop_cmp_wide
        if not wide:
            self.op("sub", isa.NOP, iv, self.operand(ub), setf=True, unit="add")
            return "ifn"
        t = self.new("lt")
        self.op("min", t, iv, self.operand(ub))
        self.op("sub", isa.NOP, t, self.operand(ub), setf=True, unit="add")
        return "ifnz"

    def plan_pipeline(self, op, forms):
        if self.opts.get("num_stages", 2) < 2:
            return None
        body = op.regions[0]
        if any(o.regions for o in body.ops):
            return None
        loads = [o for o in body.ops if o.name == "tt.load" and o.result.shape]
        if not loads:
            return None
        body_ids = {r.id for o in body.ops for r in o.results} | {a.id for a in body.args}
        yield_of = {a.id: y for a, y in zip(body.args[1:], body.ops[-1].operands)}

        def pure_slice(v, seen):
            """v is computed without loads (or dots, stores, reductions) of this
            iteration; for a loop-carried value that includes its next value, which
            the prefetch for the next iteration needs early."""
            if v is None or v.id not in body_ids:
                return True
            if v.id in seen:
                return True
            seen.add(v.id)
            if v.op is None:
                return v.id not in yield_of or pure_slice(yield_of[v.id], seen)
            if v.op.name in ("tt.load", "tt.dot", "tt.store", "tt.reduce"):
                return False
            return all(pure_slice(o, seen) for o in v.op.operands)

        for ld in loads:
            if not all(pure_slice(o, set()) for o in ld.operands):
                return None
        # any other TMU user in the body would break the stream invariant
        if any(o.name == "tt.load" and not o.result.shape for o in body.ops):
            return None
        # a prefetch must not overtake a store of the same iteration
        if any(o.name == "tt.store" for o in body.ops):
            return None
        return LoopStream(self, op, loads)

    def op_scf_if(self, op):
        cond = self.as_pred(self.get(op.operands[0]))
        if cond.kind == "const":
            region = op.regions[0] if cond.value else op.regions[1]
            self.lower_block(region)
            y = region.ops[-1]
            vals = [self.get(v) for v in y.operands]
            return tuple(vals) if len(vals) != 1 else (vals[0] if vals else None)
        self.tmu.drain()
        self.dma_wait()
        self.region_depth += 1
        outs = []
        for r in op.results:
            if r.shape:
                outs.append(Dense(r.shape, r.dtype, [self.new("if", r.dtype.is_float) for _ in range(nregs(r.shape))]))
            else:
                outs.append(S(reg=self.new("if", r.dtype.is_float), is_float=r.dtype.is_float))
        c = self.set_flags(cond, 0)
        else_lbl = f"else{self.blocks}"
        end_lbl = f"endif{self.blocks + 1}"
        self.blocks += 2
        if c != "always":                      # branch to else where the condition fails
            self.emit(MInst(kind="branch", bcond={"ifz": "allnz", "ifnz": "allz", "ifn": "allnn",
                                                  "ifnn": "alln", "never": "always"}[c],
                            target=else_lbl))
        then_blk = self.new_block("then")
        self.cur = then_blk
        self.lower_block(op.regions[0])
        self.tmu.drain()
        self.dma_wait()
        self._if_assign(outs, op.regions[0].ops[-1])
        self.emit(MInst(kind="branch", bcond="always", target=end_lbl))
        eb = mir.Block(else_lbl, loop_depth=self.loop_depth)
        self.add_block(eb)
        self.cur = eb
        self.lower_block(op.regions[1])
        self.tmu.drain()
        self.dma_wait()
        self._if_assign(outs, op.regions[1].ops[-1])
        end = mir.Block(end_lbl, loop_depth=self.loop_depth)
        self.add_block(end)
        self.cur = end
        self.region_depth -= 1
        return tuple(outs) if len(outs) != 1 else (outs[0] if outs else None)

    def _if_assign(self, outs, yield_op):
        for o, v in zip(outs, yield_op.operands):
            val = self.get(v)
            if isinstance(o, S):
                self.copy_into(o.reg, self.scalar_of(val) if not isinstance(val, S) else val)
            else:
                d = self.materialize(val)
                for j, r in enumerate(o.regs):
                    self.op("mov", r, self.reg_of(d, j if len(d.regs) > 1 else 0))


# ---------------------------------------------------------------------------
# Software pipelining of loop loads
# ---------------------------------------------------------------------------


def body_dependents(body: ir.Block) -> set:
    """Ids of body values that (transitively) depend on the block arguments."""
    deps = {a.id for a in body.args}
    for op in body.ops:
        if any(o is not None and o.id in deps for o in op.operands):
            for r in op.results:
                deps.add(r.id)
    return deps


class _EnvScope:
    """Run lowering with block arguments rebound (values depending on them
    are re-lowered lazily into `env`, which persists between uses)."""

    def __init__(self, lw, env):
        self.lw = lw
        self.env = env

    def __enter__(self):
        self.saved = self.lw.env
        self.lw.env = self.env
        return self

    def __exit__(self, *a):
        self.lw.env = self.saved


class LoopStream:
    """Keeps the TMU FIFOs full across loop iterations.

    The pipelined loads' registers form a list of E entries per iteration.
    Invariant at the top of the loop body: the first n0 entries of the
    iteration are already requested.  The body receives entries in order
    and every receive frees a FIFO slot that is immediately refilled with
    the next entry - first the rest of this iteration, then the first n0
    entries of the next one (addresses computed from the yielded pointers,
    clamped to the current ones on the last iteration).
    """

    def __init__(self, lw: Lowering, op, loads):
        self.lw = lw
        self.op = op
        self.loads = loads
        self.body = op.regions[0]
        self.deps = body_dependents(self.body)
        self.entries = []
        for ld in loads:
            for j in range(nregs(ld.result.shape)):
                self.entries.append((ld, j))
        self.E = len(self.entries)
        self.n0 = min(self.E, 2 * TMU_DEPTH)
        self.dense = {}
        # block arguments the load addresses / masks depend on
        self.addr_args = set()
        arg_ids = {a.id for a in self.body.args}
        seen = set()
        stack = [o for ld in loads for o in ld.operands if o is not None]
        while stack:
            v = stack.pop()
            if v.id in seen:
                continue
            seen.add(v.id)
            if v.id in arg_ids:
                self.addr_args.add(v.id)
            elif v.op is not None and v.op.parent is self.body:
                stack.extend(o for o in v.op.operands if o is not None)

    def unit_of(self, e):
        return e % 2

    def scoped_env(self, bindings):
        env = {k: v for k, v in self.lw.env.items() if k not in self.deps}
        env.update(bindings)
        return env

    def _emitter(self, e):
        """Address emitter for entry e, evaluated in the *current* lw.env."""
        ld, j = self.entries[e]
        lw = self.lw
        ptr = lw.get(ld.operands[0])
        mask = lw.as_pred(lw.get(ld.operands[1])) if ld.attrs.get("has_mask") else None
        if mask is not None and mask.kind == "const" and mask.value:
            mask = None
        return lw.addr_emitter(ptr, mask, j)

    def _push(self, e, tok):
        q = self.lw.tmu.q[tok.unit]
        if len(q) >= TMU_DEPTH:
            raise CompilationError("internal: TMU FIFO overflow in pipeline")
        self._emitter(e)(isa.TMU_S[tok.unit])
        tok.issued = True
        q.append(tok)

    # -- preheader ------------------------------------------------------------

    def prologue(self, lb, inits):
        lw = self.lw
        env = self.scoped_env({self.body.args[0].id: lb,
                               **{a.id: v for a, v in zip(self.body.args[1:], inits)}})
        self.pre_tokens = []
        with _EnvScope(lw, env):
            for e in range(self.n0):
                tok = Token(self.unit_of(e))
                self._push(e, tok)
                self.pre_tokens.append(tok)

    # -- body -----------------------------------------------------------------

    def begin_body(self, iv, step, ub, forms):
        self.iv, self.step, self.ub, self.forms = iv, step, ub, forms
        self.cur_tokens = {}
        for e, tok in enumerate(self.pre_tokens):
            tok.stream, tok.entry = self, e
            self.cur_tokens[e] = tok
        self.cur_issued = self.n0
        self.next_issued = 0
        self.next_env = None
        self.next_tokens = []

    def dense_for(self, ld) -> Dense:
        if ld in self.dense:
            return self.dense[ld]
        lw = self.lw
        mask = lw.as_pred(lw.get(ld.operands[1])) if ld.attrs.get("has_mask") else None
        other = lw.get(ld.operands[2]) if ld.attrs.get("has_other") else None
        if mask is not None and mask.kind == "const" and mask.value:
            mask = None
        other_d = None
        if mask is not None and other is not None and not _is_zero(other):
            other_d = lw.materialize(other)
        regs = []
        for e, (l, j) in enumerate(self.entries):
            if l is not ld:
                continue
            tok = self.cur_tokens.get(e)
            if tok is None:
                tok = Token(self.unit_of(e))
                tok.stream, tok.entry = self, e
                self.cur_tokens[e] = tok
            if other_d is not None:
                tok.post = lw.make_other_fixup(mask, other_d, j)
            regs.append(tok)
        d = Dense(ld.result.shape, ld.result.dtype, regs)
        self.dense[ld] = d
        return d

    def issue_until(self, tok):
        while self.cur_issued <= tok.entry:
            self._issue_cur(self.cur_issued, force=True)

    def _issue_cur(self, e, force=False):
        lw = self.lw
        tok = self.cur_tokens.get(e)
        if tok is None:
            tok = Token(self.unit_of(e))
            tok.stream, tok.entry = self, e
            self.cur_tokens[e] = tok
        u = tok.unit
        while len(lw.tmu.q[u]) >= TMU_DEPTH:
            if not force:
                return False
            lw.tmu.stream = None
            lw.tmu._recv_one(u)
            lw.tmu.stream = self
        self._push(e, tok)
        self.cur_issued = e + 1
        return True

    def on_receive(self):
        self.top_up()

    def top_up(self):
        lw = self.lw
        while True:
            if self.cur_issued < self.E:
                if not self._issue_cur(self.cur_issued):
                    return
                continue
            if self.next_issued >= self.n0:
                return
            e = self.next_issued
            if len(lw.tmu.q[self.unit_of(e)]) >= TMU_DEPTH:
                return
            if self.next_env is None:
                self.next_env = self._make_next_env()
            tok = Token(self.unit_of(e))
            with _EnvScope(lw, self.next_env):
                self._push(e, tok)
            self.next_tokens.append(tok)
            self.next_issued = e + 1

    def _make_next_env(self):
        """Values of the loop-carried address inputs in the next iteration.
        On the last iteration they are clamped to the current ones, so a
        prefetch never reads past the data."""
        lw = self.lw
        body = self.body
        yield_op = body.ops[-1]
        nexts = []
        for arg, yv, (form, copies, kind) in zip(body.args[1:], yield_op.operands, self.forms):
            if arg.id in self.addr_args:
                # (old register, next value) pairs, computed before the flags below
                moves = lw.carried_moves(form, kind, lw.get(yv))
                nexts.append((arg, form, kind, [(old, lw.operand(src)) for old, src in moves]))
        iv_next = lw.new("ivn")
        lw.iv_step(iv_next, self.iv, self.step, self.sat)
        # flags: cond holds <=> iv_next < ub (another iteration follows)
        cond = lw.loop_lt_flags(iv_next, self.ub, wide=self.wide)
        ivp = lw.new("ivp")
        lw.op("mov", ivp, self.iv)
        lw.op("mov", ivp, iv_next, cond=cond)
        bindings = {body.args[0].id: S(reg=ivp)}
        for arg, form, kind, moves in nexts:
            regs = []
            for old, new in moves:
                t = lw.new("pf", old.is_float)
                lw.op("mov", t, old)
                lw.op("mov", t, new, cond=cond)
                regs.append(t)
            if kind == "affine":
                bindings[arg.id] = Affine(form.shape, form.dtype, S(reg=regs[0]), form.coeffs, form.dense)
            elif kind == "scalar":
                bindings[arg.id] = S(reg=regs[0], is_float=form.is_float)
            elif kind == "pred":
                bindings[arg.id] = Pred("dense", form.shape, Dense(form.shape, ir.int1, regs))
            else:
                bindings[arg.id] = Dense(form.shape, form.dtype, regs)
        return self.scoped_env(bindings)

    def end_body(self):
        lw = self.lw
        for e in range(self.E):
            tok = self.cur_tokens.get(e)
            if tok is None:
                tok = Token(self.unit_of(e))
                tok.stream, tok.entry = self, e
                self.cur_tokens[e] = tok
            if tok.vreg is None:
                lw.tmu.receive(tok)
        self.top_up()
        if self.next_issued != self.n0 or lw.tmu.outstanding() != self.n0:
            raise CompilationError("internal: pipeline invariant broken")
        self.pre_tokens = self.next_tokens


def _walk_ops(block):
    for o in block.ops:
        yield o
        for r in o.regions:
            yield from _walk_ops(r)


def _is_zero(x):
    """x is the constant with all bits zero (int 0 or +0.0, not -0.0)."""
    s = x.s if isinstance(x, Splat) else x if isinstance(x, S) else None
    if s is None or not s.is_const:
        return False
    return (_f2u(s.const) if s.is_float else int(s.const) & 0xFFFFFFFF) == 0


def _same_coeffs(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.is_const != y.is_const:
            return False
        if x.is_const and x.const != y.const:
            return False
        if not x.is_const and x.reg is not y.reg:
            return False
    return True


def _same_dense(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a.regs) == len(b.regs) and all(x is y for x, y in zip(a.regs, b.regs))


def _wrap(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


def _shift_imm(k: int) -> int:
    """Small-immediate value whose low 5 bits are k (shifts use b & 31)."""
    k &= 31
    return k if k < 16 else k - 32


def _imm_const(x):
    if isinstance(x, Imm):
        v = x.value
        if v < 16 or v >= 0xFFFFFFF0:
            return _wrap(v)
    return None


def _imm_const_f(x):
    if isinstance(x, Imm):
        code = isa.SMALL_IMM_CODE[x.value]
        if code >= 32:
            return isa.u2f(x.value)
    return None


def _fits24(r) -> bool:
    return r is not None and r[0] >= 0 and r[1] < (1 << 24)


def _fkey(x) -> int:
    """fmin / fmax order of the QPU (sign-magnitude bits; -0 and denormals count as
    positive): see the emulator."""
    bits = int(np.array(x, np.float32).view(np.uint32))
    m = bits & 0x7FFFFFFF
    return -m if bits >> 31 and (bits >> 23) & 0xFF else m


def _const_op(opname, a, b, is_float):
    if is_float:
        if opname in ("min", "max"):
            le = _fkey(a) <= _fkey(b)
            return float(np.float32(a if le == (opname == "min") else b))
        a, b = np.float32(a), np.float32(b)
        with np.errstate(all="ignore"):
            return float({"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[opname])
    if opname in ("divu", "remu", "maxu", "minu"):
        ua, ub = int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF
        r = {"divu": lambda: ua // ub, "remu": lambda: ua % ub,
             "maxu": lambda: max(ua, ub), "minu": lambda: min(ua, ub)}[opname]()
        return _wrap(r)
    a, b = _wrap(a), _wrap(b)
    if opname == "add":
        return _wrap(a + b)
    if opname == "sub":
        return _wrap(a - b)
    if opname == "mul":
        return _wrap(a * b)
    if opname == "divs":
        q = abs(a) // abs(b)
        return _wrap(q if (a >= 0) == (b >= 0) else -q)
    if opname == "rems":
        r = abs(a) % abs(b)
        return r if a >= 0 else -r
    if opname == "and":
        return a & b
    if opname == "or":
        return a | b
    if opname == "xor":
        return _wrap(a ^ b)
    if opname == "shl":
        return _wrap(a << (b & 31))
    if opname == "shrs":
        return a >> (b & 31)
    if opname == "shru":
        return (a & 0xFFFFFFFF) >> (b & 31)
    if opname == "min":
        return min(a, b)
    if opname == "max":
        return max(a, b)
    raise ValueError(opname)


def _uniform_scalar(x):
    """The scalar of a uniform integer value (S, Splat), else None."""
    if isinstance(x, Splat):
        x = x.s
    elif isinstance(x, Affine) and x.dense is None and all(c.is_const and c.const == 0 for c in x.coeffs):
        x = x.base
    return x if isinstance(x, S) and not x.is_float else None


def _sub_cannot_overflow(ra, rb, unsigned) -> bool:
    """x - y fits 32 signed bits for all x in ra, y in rb (sign-flipped if unsigned)."""
    if ra is None or rb is None:
        return False
    if unsigned and (ra[0] < 0 or rb[0] < 0):
        return False
    return ra[1] - rb[0] <= 2 ** 31 - 1 and ra[0] - rb[1] >= -2 ** 31


def _math_const(name, c):
    with np.errstate(all="ignore"):
        x = np.float32(c)
        f = {"math.exp": np.exp, "math.exp2": np.exp2, "math.log": np.log, "math.log2": np.log2}[name]
        return float(f(x))


def _float_const(x):
    """Python value of a uniform float constant operand, else None."""
    if isinstance(x, Splat):
        x = x.s
    if isinstance(x, S) and x.is_const:
        return float(x.const)
    return None


def _cmp_const(pred, a, b):
    if pred in ("olt", "ole", "ogt", "oge", "oeq", "une"):
        return {"olt": a < b, "ole": a <= b, "ogt": a > b, "oge": a >= b, "oeq": a == b,
                "une": a != b}[pred]
    sa, sb = _wrap(int(a)), _wrap(int(b))
    ua, ub = int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF
    return {"slt": sa < sb, "sle": sa <= sb, "sgt": sa > sb, "sge": sa >= sb, "eq": ua == ub,
            "ne": ua != ub, "ult": ua < ub, "ule": ua <= ub, "ugt": ua > ub, "uge": ua >= ub}[pred]
