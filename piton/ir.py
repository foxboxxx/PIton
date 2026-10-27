"""Tile-level IR, modelled on Triton's TTIR.

Values are scalars or statically shaped blocks ("tensors") of i1 / i32 /
f32 / pointer elements.  Ops live in blocks; `scf.for` and `scf.if` carry
nested regions, exactly like TTIR's structured control flow, so the
textual dump (`Function.dump()`) reads much like `triton.compile(...).asm['ttir']`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class DType:
    def __init__(self, name: str, bits: int, kind: str, elem: Optional["DType"] = None):
        self.name = name
        self.bits = bits
        self.kind = kind            # 'int' | 'uint' | 'float' | 'bool' | 'ptr'
        self.elem = elem

    @property
    def is_float(self):
        return self.kind == "float"

    @property
    def is_int(self):
        return self.kind in ("int", "uint", "bool")

    @property
    def is_bool(self):
        return self.kind == "bool"

    @property
    def is_ptr(self):
        return self.kind == "ptr"

    @property
    def is_signed(self):
        return self.kind == "int"

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, DType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def to_mlir(self) -> str:
        if self.is_ptr:
            return f"!tt.ptr<{self.elem.to_mlir()}>"
        return {"i1": "i1", "int32": "i32", "uint32": "i32", "fp32": "f32"}[self.name]


int1 = DType("i1", 1, "bool")
int32 = DType("int32", 32, "int")
uint32 = DType("uint32", 32, "uint")
float32 = DType("fp32", 32, "float")


def pointer_type(elem: DType) -> DType:
    return DType(f"*{elem.name}", 32, "ptr", elem)


SCALAR_TYPES = {"i1": int1, "int1": int1, "i32": int32, "int32": int32, "u32": uint32,
                "uint32": uint32, "fp32": float32, "f32": float32, "float32": float32}


def parse_type(text: str) -> DType:
    text = text.strip()
    if text.startswith("*"):
        return pointer_type(parse_type(text[1:]))
    if text not in SCALAR_TYPES:
        raise ValueError(f"unsupported type '{text}' (supported: *fp32, *i32, i32, fp32)")
    return SCALAR_TYPES[text]


# ---------------------------------------------------------------------------
# Values, ops, blocks
# ---------------------------------------------------------------------------

_ids = itertools.count()


class Value:
    def __init__(self, dtype: DType, shape: tuple = (), op: Optional["Op"] = None,
                 name: Optional[str] = None):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.op = op
        self.id = next(_ids)
        self.name = name
        self.uses: list = []

    @property
    def is_tensor(self) -> bool:
        return len(self.shape) > 0

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def type_str(self) -> str:
        if not self.shape:
            return self.dtype.to_mlir()
        return f"tensor<{'x'.join(map(str, self.shape))}x{self.dtype.to_mlir()}>"

    def __repr__(self):
        return f"%{self.name or self.id}"


@dataclass(eq=False)
class Op:
    name: str
    operands: list
    results: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)      # list[Block]
    parent: Optional["Block"] = None

    @property
    def result(self) -> Value:
        return self.results[0]

    def __repr__(self):
        return f"Op({self.name})"


class Block:
    def __init__(self, args=None):
        self.args: list = list(args or [])
        self.ops: list = []

    def append(self, op: Op) -> Op:
        op.parent = self
        self.ops.append(op)
        return op


@dataclass
class Function:
    name: str
    args: list                     # list[Value]
    arg_names: list
    body: Block
    constants: dict = field(default_factory=dict)   # arg name -> python value

    def walk(self):
        """Yield every op (pre-order, including nested regions)."""
        def rec(block):
            for op in block.ops:
                yield op
                for r in op.regions:
                    yield from rec(r)
        yield from rec(self.body)

    def dump(self) -> str:
        names = {}
        counter = itertools.count()

        def n(v: Value) -> str:
            if v.id not in names:
                names[v.id] = v.name if v.name else str(next(counter))
            return "%" + names[v.id]

        lines = []
        sig = ", ".join(f"{n(a)}: {a.type_str()}" for a in self.args)
        lines.append(f"tt.func public @{self.name}({sig}) {{")

        def fmt_attr(k, v):
            return f"{k} = {v!r}" if not isinstance(v, str) else f'{k} = "{v}"'

        def rec(block, indent):
            pad = "  " * indent
            for op in block.ops:
                res = ", ".join(n(r) for r in op.results)
                lhs = f"{res} = " if res else ""
                ops = ", ".join(n(o) for o in op.operands if o is not None)
                attrs = ""
                if op.attrs:
                    attrs = " {" + ", ".join(fmt_attr(k, v) for k, v in op.attrs.items()) + "}"
                types = ""
                if op.results:
                    types = " : " + ", ".join(r.type_str() for r in op.results)
                if op.regions:
                    lines.append(f"{pad}{lhs}{op.name} {ops}{attrs}{types} {{")
                    for i, reg in enumerate(op.regions):
                        if i:
                            lines.append(f"{pad}}} else {{")
                        if reg.args:
                            lines.append(f"{pad}  ^bb({', '.join(n(a) + ': ' + a.type_str() for a in reg.args)}):")
                        rec(reg, indent + 1)
                    lines.append(f"{pad}}}")
                else:
                    lines.append(f"{pad}{lhs}{op.name} {ops}{attrs}{types}")

        rec(self.body, 1)
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class Builder:
    """Appends ops at an insertion point (a Block)."""

    def __init__(self, block: Block):
        self.block = block

    def create(self, name: str, operands, result_types=(), attrs=None, regions=()) -> Op:
        op = Op(name, list(operands), attrs=dict(attrs or {}), regions=list(regions))
        for dtype, shape in result_types:
            op.results.append(Value(dtype, shape, op))
        for o in op.operands:
            if o is not None:
                o.uses.append(op)
        self.block.append(op)
        return op


def replace_all_uses(old: Value, new: Value, root: Block) -> None:
    """Replace uses of `old` by `new` in every op under `root`."""
    def rec(block):
        for op in block.ops:
            for i, o in enumerate(op.operands):
                if o is old:
                    op.operands[i] = new
                    new.uses.append(op)
            for r in op.regions:
                rec(r)
    rec(root)
    old.uses = []
