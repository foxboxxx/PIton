"""Machine IR: QPU operations on virtual registers.

A `MInst` is the unit the scheduler moves around.  It normally holds a
single ALU operation (`MOp`), but the lowering can also emit pre-paired
add+mul instructions (`atomic=True`), e.g. the fused multiply-accumulate
pairs of the matmul micro-kernel, plus pseudo instructions that only carry
a signal (`ldtmu0`), load immediates, branches and semaphores.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Union

from ..qpu import isa
from ..qpu.isa import Imm, Reg

_vid = itertools.count()


class VReg:
    """A 16-lane virtual register.

    cls: 'any' (accumulator or regfile), 'acc' (must be r0-r3, e.g. rotation
    source), 'regfile' (never an accumulator), 'r4' (result of a TMU/SFU
    read, lives in r4 until moved), 'r5' (the r5 broadcast register).
    """
    __slots__ = ("id", "name", "cls", "phys", "is_float", "hint")

    def __init__(self, name: str = "", cls: str = "any", is_float: bool = False):
        self.id = next(_vid)
        self.name = name
        self.cls = cls
        self.phys: Optional[Reg] = None
        self.is_float = is_float
        self.hint = None

    def __repr__(self):
        if self.phys is not None:
            return self.phys.name
        return f"%{self.name}{self.id}" if self.name else f"%v{self.id}"


Operand = Union[VReg, Reg, Imm]

ADD_ONLY = {"fadd", "fsub", "fmin", "fmax", "fminabs", "fmaxabs", "ftoi", "itof", "add", "sub",
            "shr", "asr", "ror", "shl", "min", "max", "and", "or", "xor", "not", "clz"}
MUL_ONLY = {"fmul", "mul24", "v8muld", "v8min", "v8max"}


@dataclass(eq=False)
class MOp:
    op: str                         # opcode name, or 'mov' (either unit)
    dst: Union[VReg, Reg]
    srcs: list
    cond: str = "always"
    setf: bool = False
    rot: Union[None, int, str] = None   # upward mul-output rotation
    unit: str = ""                  # 'add' | 'mul' | 'any' (derived when empty)

    def __post_init__(self):
        if not self.unit:
            if self.rot is not None or self.op in MUL_ONLY:
                self.unit = "mul"
            elif self.op in ADD_ONLY:
                self.unit = "add"
            else:
                self.unit = "any"

    def uses(self):
        return [s for s in self.srcs if isinstance(s, VReg)]

    def __repr__(self):
        c = "" if self.cond == "always" else "." + self.cond
        sf = ".setf" if self.setf else ""
        r = f" rot{self.rot}" if self.rot is not None else ""
        return f"{self.op}{c}{sf} {self.dst!r}, {', '.join(map(repr, self.srcs))}{r}"


@dataclass(eq=False)
class MInst:
    kind: str = "alu"               # 'alu' | 'ldi' | 'branch' | 'sem' | 'sig'
    ops: list = field(default_factory=list)      # MOps (alu), or ldi destinations
    sig: Optional[str] = None       # ldtmu0 / ldtmu1 / thrend
    atomic: bool = False            # keep `ops` together in one instruction
    imm: int = 0                    # ldi value
    ldi_mode: int = 0
    bcond: str = "always"           # branch
    target: Optional[str] = None
    acquire: bool = False           # semaphore
    sema: int = 0
    comment: str = ""
    tag: str = ""                   # free-form (e.g. 'mac', 'bcast') for statistics

    def defs(self):
        out = []
        for op in self.ops:
            if isinstance(op.dst, VReg):
                out.append(op.dst)
        return out

    def uses(self):
        out = []
        for op in self.ops:
            out += op.uses()
            if op.cond != "always" and isinstance(op.dst, VReg):
                out.append(op.dst)  # conditional write keeps the old value
        return out

    @property
    def reads_flags(self) -> bool:
        return self.kind == "branch" and self.bcond != "always" or \
            any(op.cond not in ("always", "never") for op in self.ops)

    @property
    def writes_flags(self) -> bool:
        return any(op.setf for op in self.ops)

    def __repr__(self):
        if self.kind == "ldi":
            d = ", ".join(f"{op.dst!r}{'' if op.cond == 'always' else '.' + op.cond}" for op in self.ops)
            return f"ldi {d}, 0x{self.imm & 0xFFFFFFFF:x}"
        if self.kind == "branch":
            return f"brr.{self.bcond} {self.target}"
        if self.kind == "sem":
            return f"{'sacq' if self.acquire else 'srel'} {self.sema}"
        parts = [repr(o) for o in self.ops]
        if self.sig:
            parts.append(self.sig)
        s = " ; ".join(parts) if parts else "nop"
        return ("[" + s + "]") if self.atomic else s


def alu(op, dst, *srcs, cond="always", setf=False, rot=None, unit="", comment="") -> MInst:
    return MInst(ops=[MOp(op, dst, list(srcs), cond=cond, setf=setf, rot=rot, unit=unit)],
                 comment=comment)


@dataclass
class Block:
    label: str
    insts: list = field(default_factory=list)
    succs: list = field(default_factory=list)    # labels (filled by Function.cfg)
    loop_depth: int = 0

    def __repr__(self):
        return f"Block({self.label}, {len(self.insts)} insts)"


@dataclass
class Function:
    name: str
    blocks: list = field(default_factory=list)

    def block(self, label) -> Block:
        for b in self.blocks:
            if b.label == label:
                return b
        raise KeyError(label)

    def compute_cfg(self):
        for i, b in enumerate(self.blocks):
            b.succs = []
            last = b.insts[-1] if b.insts else None
            if last is not None and last.kind == "branch":
                b.succs.append(last.target)
                if last.bcond != "always" and i + 1 < len(self.blocks):
                    b.succs.append(self.blocks[i + 1].label)
            elif i + 1 < len(self.blocks):
                b.succs.append(self.blocks[i + 1].label)
            if last is not None and last.sig == "thrend":
                b.succs = []

    def preds(self) -> dict:
        self.compute_cfg()
        out = {b.label: [] for b in self.blocks}
        for b in self.blocks:
            for s in b.succs:
                out[s].append(b.label)
        return out

    def dump(self) -> str:
        lines = [f"mir.func @{self.name}"]
        for b in self.blocks:
            lines.append(f"{b.label}:  (loop depth {b.loop_depth})")
            for ins in b.insts:
                c = f"    # {ins.comment}" if ins.comment else ""
                lines.append(f"    {ins!r}{c}")
        return "\n".join(lines)

    def num_insts(self) -> int:
        return sum(len(b.insts) for b in self.blocks)
