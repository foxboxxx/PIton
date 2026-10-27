"""VideoCore IV QPU instruction set.

Everything in here comes from the "VideoCore IV 3D Architecture Reference
Guide" (Broadcom, 2013), sections "QPU Instruction Encoding" (p26-34),
"QPU Instruction Set" (p35-36) and "QPU Register Address Map" (p37-38).

The module provides:
  * register objects for every readable / writable location,
  * `Instr`, a fully-resolved 64-bit QPU instruction,
  * `encode` / `decode` between `Instr` and the 64-bit machine word,
  * `format_instr`, which prints an instruction in vc4asm syntax such that
    `vc4asm` assembles it back to exactly the bits `encode` produces.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Opcodes and condition codes
# ---------------------------------------------------------------------------

SIG = {
    "bkpt": 0, "none": 1, "thrsw": 2, "thrend": 3, "sbwait": 4, "sbdone": 5,
    "lthrsw": 6, "loadcv": 7, "loadc": 8, "ldcend": 9, "ldtmu0": 10,
    "ldtmu1": 11, "loadam": 12, "smallimm": 13, "ldi": 14, "branch": 15,
}
SIG_NAME = {v: k for k, v in SIG.items()}

ADD_OPS = {
    "nop": 0, "fadd": 1, "fsub": 2, "fmin": 3, "fmax": 4, "fminabs": 5,
    "fmaxabs": 6, "ftoi": 7, "itof": 8, "add": 12, "sub": 13, "shr": 14,
    "asr": 15, "ror": 16, "shl": 17, "min": 18, "max": 19, "and": 20,
    "or": 21, "xor": 22, "not": 23, "clz": 24, "v8adds": 30, "v8subs": 31,
}
ADD_OP_NAME = {v: k for k, v in ADD_OPS.items()}

MUL_OPS = {
    "nop": 0, "fmul": 1, "mul24": 2, "v8muld": 3, "v8min": 4, "v8max": 5,
    "v8adds": 6, "v8subs": 7,
}
MUL_OP_NAME = {v: k for k, v in MUL_OPS.items()}

# Ops that only read their first operand (vc4asm duplicates it into port b).
UNARY_OPS = {"mov", "not", "itof", "ftoi", "clz"}
FLOAT_ADD_OPS = {"fadd", "fsub", "fmin", "fmax", "fminabs", "fmaxabs", "ftoi"}

CONDS = {"never": 0, "always": 1, "ifz": 2, "ifnz": 3, "ifn": 4, "ifnn": 5,
         "ifc": 6, "ifnc": 7}
COND_NAME = {v: k for k, v in CONDS.items()}
# Condition that is true exactly when `c` is false.
COND_INVERT = {"ifz": "ifnz", "ifnz": "ifz", "ifn": "ifnn", "ifnn": "ifn",
               "ifc": "ifnc", "ifnc": "ifc", "always": "never", "never": "always"}

BRANCH_CONDS = {
    "allz": 0, "allnz": 1, "anyz": 2, "anynz": 3, "alln": 4, "allnn": 5,
    "anyn": 6, "anynn": 7, "allc": 8, "allnc": 9, "anyc": 10, "anync": 11,
    "always": 15,
}
BRANCH_COND_NAME = {v: k for k, v in BRANCH_CONDS.items()}

# ---------------------------------------------------------------------------
# Registers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reg:
    """A location in the QPU register space.

    acc    -- accumulator index (0-5) when read through the accumulator muxes.
    rfile  -- 'A', 'B' or 'AB' if readable through a regfile read port.
    raddr  -- read address for regfile-space reads.
    wfile  -- 'A', 'B' or 'AB': which write address space(s) contain it.
    waddr  -- write address.
    """
    name: str
    acc: Optional[int] = None
    rfile: Optional[str] = None
    raddr: Optional[int] = None
    wfile: Optional[str] = None
    waddr: Optional[int] = None

    def __repr__(self) -> str:
        return self.name

    @property
    def readable(self) -> bool:
        return self.acc is not None or self.rfile is not None

    @property
    def writable(self) -> bool:
        return self.waddr is not None

    @property
    def is_regfile(self) -> bool:
        """True for the 64 general purpose register-file locations."""
        return self.rfile in ("A", "B") and self.raddr is not None and self.raddr < 32


ACC = [Reg(f"r{i}", acc=i, wfile="AB", waddr=32 + i) for i in range(4)]
R0, R1, R2, R3 = ACC
R4 = Reg("r4", acc=4)                       # read-only: TMU / SFU results
R5 = Reg("r5", acc=5)                       # read side of r5
R5QUAD = Reg("r5quad", wfile="A", waddr=37)  # write: replicate element 0 per quad
R5REP = Reg("r5rep", wfile="B", waddr=37)    # write: replicate element 0 to all
RA = [Reg(f"ra{i}", rfile="A", raddr=i, wfile="A", waddr=i) for i in range(32)]
RB = [Reg(f"rb{i}", rfile="B", raddr=i, wfile="B", waddr=i) for i in range(32)]

NOP = Reg("-", wfile="AB", waddr=39)                    # write sink
UNIF = Reg("unif", rfile="AB", raddr=32)
VARYING = Reg("vary", rfile="AB", raddr=35)
ELEM_NUM = Reg("elem_num", rfile="A", raddr=38)
QPU_NUM = Reg("qpu_num", rfile="B", raddr=38)
VPM = Reg("vpm", rfile="AB", raddr=48, wfile="AB", waddr=48)
VR_BUSY = Reg("vr_busy", rfile="A", raddr=49)
VW_BUSY = Reg("vw_busy", rfile="B", raddr=49)
VR_WAIT = Reg("vr_wait", rfile="A", raddr=50)
VW_WAIT = Reg("vw_wait", rfile="B", raddr=50)
MUTEX = Reg("mutex", rfile="AB", raddr=51, wfile="AB", waddr=51)
TMU_NOSWAP = Reg("tmu_noswap", wfile="AB", waddr=36)
HOST_INT = Reg("host_int", wfile="AB", waddr=38)
# Write address 40 is the absolute uniforms address only in regfile-A space;
# in B space it is vc4asm's `unif_addr_rel` (found on hardware: a mul-ALU
# write with ws=0 silently changed the stream to a relative address).
UNIF_ADDR = Reg("unif_addr", wfile="A", waddr=40)
UNIF_ADDR_REL = Reg("unif_addr_rel", wfile="B", waddr=40)
VR_SETUP = Reg("vr_setup", wfile="A", waddr=49)
VW_SETUP = Reg("vw_setup", wfile="B", waddr=49)
VR_ADDR = Reg("vr_addr", wfile="A", waddr=50)
VW_ADDR = Reg("vw_addr", wfile="B", waddr=50)
SFU_RECIP = Reg("sfu_recip", wfile="AB", waddr=52)
SFU_RECIPSQRT = Reg("sfu_recipsqrt", wfile="AB", waddr=53)
SFU_EXP = Reg("sfu_exp", wfile="AB", waddr=54)      # 2**x
SFU_LOG = Reg("sfu_log", wfile="AB", waddr=55)      # log2(x)
TMU0_S = Reg("t0s", wfile="AB", waddr=56)
TMU1_S = Reg("t1s", wfile="AB", waddr=60)

SFU_REGS = {SFU_RECIP, SFU_RECIPSQRT, SFU_EXP, SFU_LOG}
TMU_S = (TMU0_S, TMU1_S)

_READ_SPECIAL = {32: UNIF, 35: VARYING, 48: VPM, 51: MUTEX}
_READ_SPECIAL_A = {38: ELEM_NUM, 49: VR_BUSY, 50: VR_WAIT}
_READ_SPECIAL_B = {38: QPU_NUM, 49: VW_BUSY, 50: VW_WAIT}
_WRITE_SPECIAL = {r.waddr: r for r in (
    NOP, VPM, MUTEX, TMU_NOSWAP, HOST_INT, SFU_RECIP,
    SFU_RECIPSQRT, SFU_EXP, SFU_LOG, TMU0_S, TMU1_S)}
_WRITE_SPECIAL_A = {37: R5QUAD, 40: UNIF_ADDR, 49: VR_SETUP, 50: VR_ADDR}
_WRITE_SPECIAL_B = {37: R5REP, 40: UNIF_ADDR_REL, 49: VW_SETUP, 50: VW_ADDR}
_TMU_OTHER = {57: "t0t", 58: "t0r", 59: "t0b", 61: "t1t", 62: "t1r", 63: "t1b"}

REG_BY_NAME = {r.name: r for r in (
    ACC + [R4, R5, R5QUAD, R5REP] + RA + RB +
    [NOP, UNIF, VARYING, ELEM_NUM, QPU_NUM, VPM, VR_BUSY, VW_BUSY, VR_WAIT,
     VW_WAIT, MUTEX, TMU_NOSWAP, HOST_INT, UNIF_ADDR, VR_SETUP, VW_SETUP,
     VR_ADDR, VW_ADDR, SFU_RECIP, SFU_RECIPSQRT, SFU_EXP, SFU_LOG, TMU0_S,
     TMU1_S, UNIF_ADDR_REL])}


def read_reg(mux: int, raddr_a: int, raddr_b: int) -> Reg:
    """Register selected by an ALU input mux (not valid for small immediates)."""
    if mux < 4:
        return ACC[mux]
    if mux == 4:
        return R4
    if mux == 5:
        return R5
    addr = raddr_a if mux == 6 else raddr_b
    if addr < 32:
        return RA[addr] if mux == 6 else RB[addr]
    if addr in _READ_SPECIAL:
        return _READ_SPECIAL[addr]
    table = _READ_SPECIAL_A if mux == 6 else _READ_SPECIAL_B
    if addr in table:
        return table[addr]
    if addr == 39:
        return Reg("-", rfile="A" if mux == 6 else "B", raddr=39)
    return Reg(f"{'ra' if mux == 6 else 'rb'}{addr}?", rfile="A" if mux == 6 else "B", raddr=addr)


def write_reg(waddr: int, file: str) -> Reg:
    """Register written through `waddr` in regfile write space `file`."""
    if waddr < 32:
        return RA[waddr] if file == "A" else RB[waddr]
    if 32 <= waddr <= 35:
        return ACC[waddr - 32]
    if waddr in _WRITE_SPECIAL:
        return _WRITE_SPECIAL[waddr]
    table = _WRITE_SPECIAL_A if file == "A" else _WRITE_SPECIAL_B
    if waddr in table:
        return table[waddr]
    if waddr in _TMU_OTHER:
        return Reg(_TMU_OTHER[waddr], wfile="AB", waddr=waddr)
    return Reg(f"w{file}{waddr}?", wfile=file, waddr=waddr)


# ---------------------------------------------------------------------------
# Small immediates
# ---------------------------------------------------------------------------


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(x: int) -> float:
    return struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0]


def small_imm_value(code: int) -> int:
    """32-bit value of small immediate encoding `code` (0..47)."""
    if code < 16:
        return code
    if code < 32:
        return (code - 32) & 0xFFFFFFFF
    if code < 40:
        return f2u(float(1 << (code - 32)))
    if code < 48:
        return f2u(1.0 / (1 << (48 - code)))
    raise ValueError("rotation code, not an immediate")


SMALL_IMM_CODE = {small_imm_value(c): c for c in range(48)}


@dataclass(frozen=True)
class Imm:
    """A small immediate operand (an ALU input taken from the raddr_b field)."""
    value: int      # 32-bit pattern

    def __post_init__(self):
        object.__setattr__(self, "value", int(self.value) & 0xFFFFFFFF)
        if self.value not in SMALL_IMM_CODE:
            raise ValueError(f"0x{self.value:x} is not a small immediate")

    @property
    def code(self) -> int:
        return SMALL_IMM_CODE[self.value]

    def __repr__(self) -> str:
        return format_small_imm(self.value)


def is_small_imm(value: int) -> bool:
    return (value & 0xFFFFFFFF) in SMALL_IMM_CODE


def format_small_imm(value: int) -> str:
    code = SMALL_IMM_CODE[value & 0xFFFFFFFF]
    if code < 32:
        return str(code if code < 16 else code - 32)
    return repr(u2f(value))


Operand = Union[Reg, Imm]

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


@dataclass
class AluOp:
    """One of the two ALU operations of an instruction.

    `op` is an add-unit or mul-unit opcode name, or 'mov' (encoded as `or`
    on the add unit and `v8min` on the mul unit, as vc4asm does).
    `rot` is the mul-output rotation: None, 1..15 (rotate "upwards": element
    i moves to element i+rot) or 'r5' (amount from r5 element 0).
    """
    op: str
    dst: Reg
    a: Operand
    b: Optional[Operand] = None
    cond: str = "always"
    rot: Union[None, int, str] = None

    def srcs(self) -> list:
        if self.op in UNARY_OPS or self.b is None:
            return [self.a]
        return [self.a, self.b]


@dataclass
class Instr:
    kind: str = "alu"                   # 'alu' | 'ldi' | 'branch' | 'sem'
    add: Optional[AluOp] = None
    mul: Optional[AluOp] = None
    sig: str = "none"                   # signal for 'alu' instructions
    setf: bool = False
    # load immediate: value written by both pipes (add/mul carry dst + cond)
    imm: int = 0
    ldi_mode: int = 0                   # 0: 32 bit, 1: per-elmt signed, 3: unsigned
    # branch
    bcond: str = "always"
    rel: bool = True
    breg: Optional[Reg] = None          # regfile-A register added to target
    target: Optional[str] = None        # label name (resolved by assembler)
    # semaphore
    acquire: bool = False
    sema: int = 0
    comment: str = ""

    @staticmethod
    def nop(comment: str = "") -> "Instr":
        return Instr(comment=comment)


class EncodeError(Exception):
    pass


def _check_rotation_source(op: AluOp) -> None:
    for s in op.srcs():
        if not (isinstance(s, Reg) and s.acc is not None and s.acc < 4):
            raise EncodeError(f"vector rotation needs r0-r3 sources: {op}")


def encode(ins: Instr, pc: int = 0, labels: Optional[dict] = None) -> int:
    """Encode `ins` (located at byte address `pc`) as a 64-bit word."""
    if ins.kind == "branch":
        return _encode_branch(ins, pc, labels or {})
    if ins.kind in ("ldi", "sem"):
        return _encode_ldi(ins)
    return _encode_alu(ins)


def _waddr_fields(ins: Instr):
    """Pick the write-swap bit and the two write addresses."""
    add, mul = ins.add, ins.mul
    need = set()            # required ws values
    if add is not None and add.dst.wfile in ("A", "B"):
        need.add(0 if add.dst.wfile == "A" else 1)
    if mul is not None and mul.dst.wfile in ("A", "B"):
        need.add(0 if mul.dst.wfile == "B" else 1)
    if len(need) > 1:
        raise EncodeError(f"add and mul destinations need different write swaps: {add} / {mul}")
    if (add is not None and mul is not None and add.dst.waddr == mul.dst.waddr
            and add.dst.wfile == "AB" and add.dst != NOP):
        raise EncodeError(f"add and mul write the same register: {add.dst}")
    ws = need.pop() if need else 0
    if add is not None and add.dst.wfile not in ("AB", "A" if ws == 0 else "B"):
        raise EncodeError(f"bad add destination {add.dst}")
    if mul is not None and mul.dst.wfile not in ("AB", "B" if ws == 0 else "A"):
        raise EncodeError(f"bad mul destination {mul.dst}")
    waddr_add = add.dst.waddr if add is not None else 39
    waddr_mul = mul.dst.waddr if mul is not None else 39
    return ws, waddr_add, waddr_mul


def _alu_cond(op: Optional[AluOp], setf: bool) -> int:
    if op is None:
        return 0
    if op.dst == NOP and not setf and op.cond == "always":
        return 0                    # vc4asm: result discarded -> never
    return CONDS[op.cond]


NULL_A = Reg("-", rfile="A", raddr=39)       # reading address 39 returns 0
NULL_B = Reg("-", rfile="B", raddr=39)


def _zero_moves(ins: Instr) -> Instr:
    """Spell `mov x, 0` as `xor x, s, s` / `v8subs x, s, s` (always 0).

    This keeps the small-immediate and signal fields free.  `s` is the
    register the other op already reads through regfile port A, or else the
    A-file null register (printed `-`): both print as text that vc4asm
    assembles back to exactly these bits (the B-file null cannot be spelled).
    """
    def is_zero_mov(op):
        return op is not None and op.op == "mov" and isinstance(op.a, Imm) and op.a.value == 0 \
            and op.rot is None
    if not (is_zero_mov(ins.add) or is_zero_mov(ins.mul)):
        return ins
    others = [op for op in (ins.add, ins.mul) if op is not None and not is_zero_mov(op)]
    if any(op.rot for op in others):
        return ins
    src = NULL_A
    for op in others:
        for x in op.srcs():
            if isinstance(x, Reg) and x.acc is None and x.rfile == "A":
                src = x
    out = replace(ins)
    if is_zero_mov(ins.add):
        out.add = replace(ins.add, op="xor", a=src, b=src)
    if is_zero_mov(ins.mul):
        out.mul = replace(ins.mul, op="v8subs", a=src, b=src)
    return out


def _encode_alu(ins: Instr) -> int:
    ins = _zero_moves(ins)
    add, mul = ins.add, ins.mul
    raddr_a = raddr_b = 39
    a_used = b_used = False
    small = None                    # raddr_b small immediate / rotation code

    def port(src: Operand) -> int:
        nonlocal raddr_a, raddr_b, a_used, b_used, small
        if isinstance(src, Imm):
            code = src.code
            if b_used or (small is not None and small != code):
                raise EncodeError(f"small immediate conflicts with B read ({src})")
            small = code
            return 7
        if src.acc is not None:
            return src.acc
        if src.rfile is None:
            raise EncodeError(f"{src} is not readable")
        files = ("A", "B") if src.rfile == "AB" else (src.rfile,)
        # Re-use a port that already reads this address.
        if "A" in files and a_used and raddr_a == src.raddr:
            return 6
        if "B" in files and b_used and raddr_b == src.raddr:
            return 7
        if "A" in files and not a_used:
            a_used, raddr_a = True, src.raddr
            return 6
        if "B" in files and not b_used and small is None:
            b_used, raddr_b = True, src.raddr
            return 7
        raise EncodeError(f"no free read port for {src}")

    muxes = [0, 0, 0, 0]
    for i, op in ((0, add), (2, mul)):
        if op is None:
            continue
        srcs = op.srcs()
        muxes[i] = port(srcs[0])
        muxes[i + 1] = port(srcs[1]) if len(srcs) > 1 else muxes[i]
    if mul is not None and mul.rot is not None:
        _check_rotation_source(mul)
        if mul.rot == "r5":
            code = 48
        else:
            code = 48 + mul.rot % 16 if mul.rot % 16 else None
        if code is not None:
            if b_used or (small is not None and small != code):
                raise EncodeError("rotation conflicts with B read / small immediate")
            small = code
    if add is not None and add.rot is not None:
        raise EncodeError("only the mul ALU can rotate")

    sig = ins.sig
    if small is not None:
        if sig != "none":
            raise EncodeError(f"signal {sig} cannot be combined with small immediates")
        sig_code = SIG["smallimm"]
        raddr_b = small
    else:
        sig_code = SIG[sig]

    ws, waddr_add, waddr_mul = _waddr_fields(ins)
    op_add = ADD_OPS["or" if add is not None and add.op == "mov" else (add.op if add else "nop")]
    op_mul = MUL_OPS["v8min" if mul is not None and mul.op == "mov" else (mul.op if mul else "nop")]
    cond_mul = _alu_cond(mul, ins.setf)
    if mul is None and add is not None and add.op == "xor" and add.a == add.b:
        # vc4asm reads `xor x, s, s` as its zero-move idiom and sets the (unused)
        # mul condition to always; match it so the .qasm assembles to these bits
        cond_mul = CONDS["always"]
    hi = (sig_code << 28) | (CONDS_FIELD(_alu_cond(add, ins.setf), cond_mul)) \
        | (int(ins.setf) << 13) | (ws << 12) | (waddr_add << 6) | waddr_mul
    lo = (op_mul << 29) | (op_add << 24) | (raddr_a << 18) | (raddr_b << 12) \
        | (muxes[0] << 9) | (muxes[1] << 6) | (muxes[2] << 3) | muxes[3]
    return (hi << 32) | lo


def CONDS_FIELD(cond_add: int, cond_mul: int) -> int:
    return (cond_add << 17) | (cond_mul << 14)


def _encode_ldi(ins: Instr) -> int:
    add, mul = ins.add, ins.mul
    ws, waddr_add, waddr_mul = _waddr_fields(ins)
    cond_add = CONDS[add.cond] if add is not None else 0
    cond_mul = CONDS[mul.cond] if mul is not None else 0
    if ins.kind == "sem":
        top = (0b1110100 << 25)
        lo = (int(ins.acquire) << 4) | (ins.sema & 15)
    else:
        top = (SIG["ldi"] << 28) | (ins.ldi_mode << 25)
        lo = ins.imm & 0xFFFFFFFF
    hi = (top >> 0) | CONDS_FIELD(cond_add, cond_mul) | (int(ins.setf) << 13) \
        | (ws << 12) | (waddr_add << 6) | waddr_mul
    return (hi << 32) | lo


def _encode_branch(ins: Instr, pc: int, labels: dict) -> int:
    add, mul = ins.add, ins.mul
    ws, waddr_add, waddr_mul = _waddr_fields(ins)
    imm = 0
    if ins.target is not None:
        if ins.target not in labels:
            raise EncodeError(f"undefined label {ins.target}")
        imm = labels[ins.target] - (pc + 32) if ins.rel else labels[ins.target]
    raddr_a = 0
    if ins.breg is not None:
        if ins.breg.rfile != "A" or not ins.breg.is_regfile:
            raise EncodeError("branch register must be in regfile A")
        raddr_a = ins.breg.raddr
    hi = (SIG["branch"] << 28) | (BRANCH_CONDS[ins.bcond] << 20) | (int(ins.rel) << 19) \
        | (int(ins.breg is not None) << 18) | (raddr_a << 13) | (ws << 12) \
        | (waddr_add << 6) | waddr_mul
    return (hi << 32) | (imm & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(word: int) -> Instr:
    hi, lo = word >> 32, word & 0xFFFFFFFF
    sig = hi >> 28
    ws = (hi >> 12) & 1
    waddr_add, waddr_mul = (hi >> 6) & 63, hi & 63
    add_file, mul_file = ("A", "B") if ws == 0 else ("B", "A")
    cond_add, cond_mul = (hi >> 17) & 7, (hi >> 14) & 7
    setf = bool((hi >> 13) & 1)

    if sig == SIG["branch"]:
        ins = Instr(kind="branch", bcond=BRANCH_COND_NAME.get((hi >> 20) & 15, "always"),
                    rel=bool((hi >> 19) & 1),
                    breg=RA[(hi >> 13) & 31] if (hi >> 18) & 1 else None)
        ins.imm = lo - (1 << 32) if lo & 0x80000000 else lo
        ins.add = AluOp("mov", write_reg(waddr_add, add_file), NOP)
        ins.mul = AluOp("mov", write_reg(waddr_mul, mul_file), NOP)
        return ins
    if sig == SIG["ldi"]:
        mode = (hi >> 25) & 7
        if mode == 4:
            ins = Instr(kind="sem", acquire=bool((lo >> 4) & 1), sema=lo & 15)
        else:
            ins = Instr(kind="ldi", imm=lo, ldi_mode=mode)
        ins.setf = setf
        if cond_add != 0 or waddr_add != 39:
            ins.add = AluOp("mov", write_reg(waddr_add, add_file), NOP, cond=COND_NAME[cond_add])
        if cond_mul != 0 or waddr_mul != 39:
            ins.mul = AluOp("mov", write_reg(waddr_mul, mul_file), NOP, cond=COND_NAME[cond_mul])
        return ins

    op_mul, op_add = lo >> 29, (lo >> 24) & 31
    raddr_a, raddr_b = (lo >> 18) & 63, (lo >> 12) & 63
    muxes = [(lo >> 9) & 7, (lo >> 6) & 7, (lo >> 3) & 7, lo & 7]
    small = sig == SIG["smallimm"]

    def src(mux: int) -> Operand:
        if mux == 7 and small:
            if raddr_b >= 48:
                raise ValueError("rotation code used as an operand")
            return Imm(small_imm_value(raddr_b))
        return read_reg(mux, raddr_a, raddr_b)

    def b_src(mux: int) -> Operand:
        if mux == 7 and small and raddr_b >= 48:
            return Imm(0)            # rotation: B mux is meaningless
        return src(mux)

    ins = Instr(kind="alu", sig="none" if small else SIG_NAME[sig], setf=setf)
    if op_add != 0 or cond_add != 0 or waddr_add != 39:
        name = ADD_OP_NAME.get(op_add, f"add?{op_add}")
        ins.add = AluOp(name, write_reg(waddr_add, add_file), b_src(muxes[0]), b_src(muxes[1]),
                        cond=COND_NAME[cond_add])
    if op_mul != 0 or cond_mul != 0 or waddr_mul != 39:
        name = MUL_OP_NAME[op_mul]
        rot = None
        if small and raddr_b >= 48:
            rot = "r5" if raddr_b == 48 else raddr_b - 48
        ins.mul = AluOp(name, write_reg(waddr_mul, mul_file), b_src(muxes[2]), b_src(muxes[3]),
                        cond=COND_NAME[cond_mul], rot=rot)
    return ins


# ---------------------------------------------------------------------------
# Assembly text (vc4asm dialect)
# ---------------------------------------------------------------------------


def _fmt_src(s: Operand, rot=None) -> str:
    text = format_small_imm(s.value) if isinstance(s, Imm) else s.name
    if rot is not None:
        text += " >> r5" if rot == "r5" else f" >> {rot}"
    return text


def _fmt_op(op: AluOp, setf: bool) -> str:
    name = op.op
    if name == "v8min" and op.a == op.b:
        name = "mov"
    if name == "or" and op.a == op.b:
        name = "mov"
    suffix = ""
    if op.cond not in ("always",):
        suffix += "." + op.cond
    if setf:
        suffix += ".setf"
    if name in UNARY_OPS:
        return f"{name}{suffix} {op.dst.name}, {_fmt_src(op.a, op.rot)}"
    return (f"{name}{suffix} {op.dst.name}, {_fmt_src(op.a, op.rot)}, "
            f"{_fmt_src(op.b, op.rot)}")


def format_instr(ins: Instr) -> str:
    if ins.kind == "branch":
        name = "brr" if ins.rel else "bra"
        cond = "" if ins.bcond == "always" else "." + ins.bcond
        dst = ins.add.dst.name if ins.add is not None and ins.add.dst != NOP else "-"
        args = []
        if ins.breg is not None:
            args.append(ins.breg.name)
        if ins.target is not None:
            args.append(":" + ins.target)
        return f"{name}{cond} {dst}, {', '.join(args)}"
    if ins.kind == "sem":
        return f"{'sacq' if ins.acquire else 'srel'} -, {ins.sema}"
    if ins.kind == "ldi":
        parts = []
        for op in (ins.add, ins.mul):
            if op is None:
                continue
            cond = "" if op.cond == "always" else "." + op.cond
            parts.append(f"ldi{cond}{'.setf' if ins.setf and not parts else ''} {op.dst.name}, "
                         f"0x{ins.imm & 0xFFFFFFFF:08x}")
        return "; ".join(parts) if parts else f"ldi -, 0x{ins.imm & 0xFFFFFFFF:08x}"
    parts = []
    setf_on_add = ins.setf and ins.add is not None and ins.add.op != "nop"
    parts.append(_fmt_op(ins.add, setf_on_add) if ins.add is not None else "nop")
    if ins.mul is not None:
        parts.append(_fmt_op(ins.mul, ins.setf and not setf_on_add))
    if ins.sig != "none":
        parts.append(ins.sig)
    return "; ".join(parts)


def canonicalize(ins: Instr) -> Instr:
    """Rewrite forms that vc4asm would assemble differently from `encode`.

    A lone `mov dst, <immediate>` (no mul op, no signal) is turned into a load
    immediate by vc4asm; do the same so text and binary agree.
    """
    if (ins.kind == "alu" and ins.mul is None and ins.sig == "none" and ins.add is not None
            and ins.add.op == "mov" and isinstance(ins.add.a, Imm)):
        return Instr(kind="ldi", imm=ins.add.a.value, setf=ins.setf,
                     add=AluOp("mov", ins.add.dst, NOP, cond=ins.add.cond),
                     comment=ins.comment)
    if ins.kind == "alu":
        return _zero_moves(ins)
    return ins


def instr_words(words) -> bytes:
    return b"".join(struct.pack("<Q", w) for w in words)
