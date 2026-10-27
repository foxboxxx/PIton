"""Tile-IR optimisation passes.

  canonicalize      constant folding, algebraic identities, CSE, DCE
  licm              hoist loop-invariant pure ops out of scf.for bodies
  sink_broadcasts   push tt.broadcast / tt.expand_dims below elementwise
                    index arithmetic so it is evaluated directly in the
                    broadcast shape (avoids cross-lane shuffles on the QPU)
  RangeAnalysis     integer interval analysis; folds comparisons whose
                    outcome is known (e.g. the K-mask of a matmul when K is
                    a constexpr multiple of BLOCK_K) and tells the backend
                    when 24-bit multiplies are exact
  prune_loop_args   drop loop-carried values nobody reads
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import ir
from .ir import Block, Builder, Op, Value

PURE_PREFIXES = ("arith.", "math.")
PURE_OPS = {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.make_range", "tt.addptr",
            "tt.reshape", "tt.trans", "tt.get_program_id", "tt.get_num_programs",
            "tt.ptr_to_int", "tt.int_to_ptr", "tt.bitcast", "tt.reduce"}
ELEMENTWISE = {"arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui",
               "arith.remsi", "arith.remui", "arith.andi", "arith.ori", "arith.xori",
               "arith.shli", "arith.shrsi", "arith.shrui", "arith.minsi", "arith.maxsi",
               "arith.minui", "arith.maxui", "arith.cmpi", "arith.select", "tt.addptr",
               "arith.extui", "arith.bitcast", "arith.sitofp", "arith.uitofp",
               "tt.ptr_to_int", "tt.int_to_ptr", "tt.bitcast"}


def is_pure(op: Op) -> bool:
    return (op.name.startswith(PURE_PREFIXES) or op.name in PURE_OPS) and not op.regions


def _walk_blocks(block: Block):
    yield block
    for op in block.ops:
        for r in op.regions:
            yield from _walk_blocks(r)


def _replace(fn: ir.Function, old: Value, new: Value):
    ir.replace_all_uses(old, new, fn.body)


def _const_value(v: Value):
    if v.op is not None and v.op.name == "arith.constant":
        return v.op.attrs["value"]
    if v.op is not None and v.op.name == "tt.splat":
        return _const_value(v.op.operands[0])
    return None


def _mk_const(builder_block: Block, before: Op, value, dtype, shape) -> Value:
    op = Op("arith.constant", [], attrs={"value": value})
    op.results.append(Value(dtype, (), op))
    op.parent = builder_block
    idx = builder_block.ops.index(before)
    builder_block.ops.insert(idx, op)
    v = op.result
    if shape:
        sp = Op("tt.splat", [v])
        sp.results.append(Value(dtype, shape, sp))
        sp.parent = builder_block
        builder_block.ops.insert(idx + 1, sp)
        v.uses.append(sp)
        return sp.result
    return v


def _wrap32(x: int, signed=True) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if signed and x >= 1 << 31 else x


def _f32(x) -> float:
    return float(np.float32(x))


def _f32_bits(x) -> int:
    return int(np.array(x, np.float32).view(np.uint32))


def norm_const(value, dtype: ir.DType):
    """Canonical python value of a constant of `dtype`: int32 values are wrapped
    to [-2**31, 2**31), uint32 values to [0, 2**32), floats rounded to float32."""
    if dtype.is_bool:
        return bool(value)
    if dtype.is_float:
        return _f32(value)
    if dtype.kind in ("int", "uint") and isinstance(value, (int, bool, np.integer)):
        return _wrap32(int(value), dtype.kind == "int")
    return value


def _fold(op: Op):
    """Return a python constant for `op` if all operands are constants."""
    vals = [_const_value(o) for o in op.operands]
    if any(v is None for v in vals) or not op.results:
        return None
    r = _fold_raw(op, vals)
    return None if r is None else norm_const(r, op.result.dtype)


def _fold_raw(op: Op, vals):
    n = op.name
    s = [_wrap32(int(v)) if isinstance(v, (int, bool)) else v for v in vals]     # signed view
    u = [int(v) & 0xFFFFFFFF if isinstance(v, (int, bool)) else v for v in vals]  # unsigned view
    is_bool = bool(op.operands) and op.operands[0].dtype.is_bool
    try:
        if n == "arith.addi":
            return u[0] + u[1]
        if n == "arith.subi":
            return u[0] - u[1]
        if n == "arith.muli":
            return u[0] * u[1]
        if n == "arith.divsi" and s[1] != 0:
            q = abs(s[0]) // abs(s[1])
            return q if (s[0] >= 0) == (s[1] >= 0) else -q
        if n == "arith.remsi" and s[1] != 0:
            r = abs(s[0]) % abs(s[1])
            return r if s[0] >= 0 else -r
        if n == "arith.divui" and u[1] != 0:
            return u[0] // u[1]
        if n == "arith.remui" and u[1] != 0:
            return u[0] % u[1]
        if n == "arith.minsi":
            return min(s)
        if n == "arith.maxsi":
            return max(s)
        if n == "arith.minui":
            return min(u)
        if n == "arith.maxui":
            return max(u)
        if n == "arith.andi":
            return (vals[0] and vals[1]) if is_bool else u[0] & u[1]
        if n == "arith.ori":
            return (vals[0] or vals[1]) if is_bool else u[0] | u[1]
        if n == "arith.xori":
            return (bool(vals[0]) != bool(vals[1])) if is_bool else u[0] ^ u[1]
        if n == "arith.shli" and 0 <= s[1] < 32:
            return u[0] << s[1]
        if n == "arith.shrsi" and 0 <= s[1] < 32:
            return s[0] >> s[1]
        if n == "arith.shrui" and 0 <= s[1] < 32:
            return u[0] >> s[1]
        if n in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf"):
            a, b = np.float32(vals[0]), np.float32(vals[1])
            with np.errstate(all="ignore"):
                r = {"arith.addf": a + b, "arith.subf": a - b, "arith.mulf": a * b,
                     "arith.divf": a / b if b != 0 else None}[n]
            return None if r is None else float(r)
        if n == "arith.sitofp":
            return _f32(s[0])
        if n == "arith.uitofp":
            return _f32(u[0])
        if n == "arith.cmpi":
            p = op.attrs["predicate"]
            if is_bool:
                a, b = bool(vals[0]), bool(vals[1])
                return {"eq": a == b, "ne": a != b}.get(p)
            return {"slt": s[0] < s[1], "sle": s[0] <= s[1], "sgt": s[0] > s[1], "sge": s[0] >= s[1],
                    "eq": u[0] == u[1], "ne": u[0] != u[1], "ult": u[0] < u[1], "ule": u[0] <= u[1],
                    "ugt": u[0] > u[1], "uge": u[0] >= u[1]}[p]
        if n == "arith.extui":
            return int(vals[0])
        if n in ("arith.bitcast", "tt.bitcast"):
            src, dst = op.operands[0].dtype, op.result.dtype
            if src.is_float and dst.kind in ("int", "uint"):
                return _f32_bits(vals[0])
            if src.kind in ("int", "uint") and dst.is_float:
                return float(np.array(u[0], np.uint32).view(np.float32))
            if src.kind in ("int", "uint") and dst.kind in ("int", "uint"):
                return u[0]
            return None
    except Exception:
        return None
    return None


def canonicalize(fn: ir.Function, ranges: Optional["RangeAnalysis"] = None) -> None:
    for block in _walk_blocks(fn.body):
        for op in block.ops:
            if op.name == "arith.constant" and op.results:
                op.attrs["value"] = norm_const(op.attrs["value"], op.result.dtype)
    changed = True
    while changed:
        changed = False
        for block in list(_walk_blocks(fn.body)):
            for op in list(block.ops):
                if not op.results or op.name == "arith.constant" or not is_pure(op):
                    continue
                c = _fold(op)
                if c is None and ranges is not None and op.name == "arith.cmpi":
                    c = ranges.fold_cmp(op)
                if c is not None:
                    r = op.result
                    new = _mk_const(block, op, c, r.dtype, r.shape)
                    _replace(fn, r, new)
                    changed = True
                    continue
                simp = _simplify(op)
                if simp is not None:
                    _replace(fn, op.result, simp)
                    changed = True
        if cse(fn):
            changed = True
        if dce(fn):
            changed = True


def _simplify(op: Op) -> Optional[Value]:
    n = op.name
    if n in ("arith.addi", "tt.addptr") and len(op.operands) == 2:
        if _const_value(op.operands[1]) == 0 and op.operands[0].shape == op.result.shape:
            return op.operands[0]
    if n == "arith.addi" and _const_value(op.operands[0]) == 0 and op.operands[1].shape == op.result.shape:
        return op.operands[1]
    if n == "arith.muli":
        for i in (0, 1):
            if _const_value(op.operands[i]) == 1 and op.operands[1 - i].shape == op.result.shape:
                return op.operands[1 - i]
    if n == "arith.subi" and _const_value(op.operands[1]) == 0 and op.operands[0].shape == op.result.shape:
        return op.operands[0]
    if n == "arith.andi" and op.result.dtype.is_bool:
        for i in (0, 1):
            if _const_value(op.operands[i]) is True and op.operands[1 - i].shape == op.result.shape:
                return op.operands[1 - i]
    if n == "arith.select":
        c = _const_value(op.operands[0])
        if c is True and op.operands[1].shape == op.result.shape:
            return op.operands[1]
        if c is False and op.operands[2].shape == op.result.shape:
            return op.operands[2]
    if n == "arith.bitcast" and op.operands[0].dtype == op.result.dtype:
        return op.operands[0]
    if n in ("tt.broadcast",) and op.operands[0].shape == op.result.shape:
        return op.operands[0]
    return None


def _key(op: Op):
    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, float):
            return ("f", _f32_bits(v))          # keeps -0.0 apart from +0.0
        if isinstance(v, bool):
            return ("b", v)
        return v
    return (op.name, tuple(o.id for o in op.operands),
            tuple(sorted((k, norm(v)) for k, v in op.attrs.items())),
            tuple((r.dtype.name, r.shape) for r in op.results))


def cse(fn: ir.Function) -> bool:
    changed = False

    def rec(block: Block, seen: dict):
        nonlocal changed
        local = dict(seen)
        for op in list(block.ops):
            if op.regions:
                for r in op.regions:
                    rec(r, local)
                continue
            if not is_pure(op) or not op.results:
                continue
            k = _key(op)
            if k in local:
                for old, new in zip(op.results, local[k].results):
                    _replace(fn, old, new)
                changed = True
            else:
                local[k] = op

    rec(fn.body, {})
    return changed


def _uses_count(fn: ir.Function) -> dict:
    counts = {}

    def rec(block):
        for op in block.ops:
            for o in op.operands:
                if o is not None:
                    counts[o.id] = counts.get(o.id, 0) + 1
            for r in op.regions:
                rec(r)
    rec(fn.body)
    return counts


def dce(fn: ir.Function) -> bool:
    changed = False
    while True:
        counts = _uses_count(fn)
        removed = False
        for block in _walk_blocks(fn.body):
            keep = []
            for op in block.ops:
                if op.results and is_pure(op) and all(counts.get(r.id, 0) == 0 for r in op.results):
                    removed = True
                    continue
                if op.name == "tt.load" and all(counts.get(r.id, 0) == 0 for r in op.results):
                    removed = True
                    continue
                keep.append(op)
            block.ops = keep
        if not removed:
            break
        changed = True
    if prune_loop_args(fn):
        changed = True
    return changed


def prune_loop_args(fn: ir.Function) -> bool:
    """Remove scf.for iter_args whose block arg and result are both unused."""
    changed = False
    counts = _uses_count(fn)
    for block in list(_walk_blocks(fn.body)):
        for op in block.ops:
            if op.name != "scf.for":
                continue
            body = op.regions[0]
            yield_op = body.ops[-1]
            n = len(op.results)
            keep = []
            for i in range(n):
                arg = body.args[1 + i]
                res = op.results[i]
                # the block arg may only feed its own yield
                arg_uses = counts.get(arg.id, 0)
                self_yield = 1 if yield_op.operands[i] is arg else 0
                if counts.get(res.id, 0) == 0 and arg_uses - self_yield == 0:
                    continue
                keep.append(i)
            if len(keep) == n:
                continue
            changed = True
            op.operands = op.operands[:3] + [op.operands[3 + i] for i in keep]
            op.results = [op.results[i] for i in keep]
            body.args = [body.args[0]] + [body.args[1 + i] for i in keep]
            yield_op.operands = [yield_op.operands[i] for i in keep]
            counts = _uses_count(fn)
    return changed


# ---------------------------------------------------------------------------
# LICM
# ---------------------------------------------------------------------------


def licm(fn: ir.Function) -> bool:
    changed = False

    def rec(block: Block):
        nonlocal changed
        i = 0
        while i < len(block.ops):
            op = block.ops[i]
            for r in op.regions:
                rec(r)
            if op.name == "scf.for":
                body = op.regions[0]
                inside = {v.id for v in body.args}
                for bop in body.ops:
                    for r in bop.results:
                        inside.add(r.id)
                    _collect_nested_defs(bop, inside)
                moved = True
                while moved:
                    moved = False
                    for bop in list(body.ops):
                        if not is_pure(bop) or not bop.results or bop.name == "tt.reduce":
                            continue
                        if any(o is not None and o.id in inside for o in bop.operands):
                            continue
                        body.ops.remove(bop)
                        bop.parent = block
                        block.ops.insert(block.ops.index(op), bop)
                        for r in bop.results:
                            inside.discard(r.id)
                        i += 1
                        moved = True
                        changed = True
            i += 1

    rec(fn.body)
    return changed


def _collect_nested_defs(op: Op, out: set):
    for r in op.regions:
        for a in r.args:
            out.add(a.id)
        for o in r.ops:
            for res in o.results:
                out.add(res.id)
            _collect_nested_defs(o, out)


# ---------------------------------------------------------------------------
# Broadcast sinking
# ---------------------------------------------------------------------------

_SINKABLE = ELEMENTWISE


def sink_broadcasts(fn: ir.Function) -> bool:
    """broadcast(f(a, b)) -> f(broadcast(a), broadcast(b)) for index math.

    Only applied when `f`'s operands are themselves cheap (ranges, splats,
    constants or further sinkable elementwise ops), so data never needs to
    move between SIMD lanes.
    """
    changed = False

    def cheap(v: Value, depth=0) -> bool:
        if depth > 12 or v.op is None:
            return v.op is None and not v.shape
        n = v.op.name
        if n in ("tt.make_range", "tt.splat", "arith.constant", "tt.get_program_id",
                 "tt.get_num_programs"):
            return True
        if n in ("tt.expand_dims", "tt.broadcast"):
            return cheap(v.op.operands[0], depth + 1)
        if n in _SINKABLE:
            return all(cheap(o, depth + 1) for o in v.op.operands)
        return False

    for block in list(_walk_blocks(fn.body)):
        i = 0
        while i < len(block.ops):
            op = block.ops[i]
            if op.name in ("tt.broadcast", "tt.expand_dims"):
                src = op.operands[0]
                d = src.op
                if d is not None and d.name in _SINKABLE and cheap(src) and src.shape:
                    b = Builder(block)
                    saved = block.ops
                    block.ops = []
                    new_operands = []
                    for o in d.operands:
                        if op.name == "tt.expand_dims":
                            if not o.shape:
                                new_operands.append(o)
                                continue
                            shape = list(o.shape)
                            shape.insert(op.attrs["axis"], 1)
                            nop = b.create("tt.expand_dims", [o], [(o.dtype, tuple(shape))], dict(op.attrs))
                        else:
                            if not o.shape:
                                new_operands.append(o)
                                continue
                            nop = b.create("tt.broadcast", [o], [(o.dtype, op.result.shape)])
                        new_operands.append(nop.result)
                    # scalars operands of elementwise ops must be splatted to the new shape
                    fixed = []
                    for o in new_operands:
                        if not o.shape:
                            sp = b.create("tt.splat", [o], [(o.dtype, op.result.shape)])
                            fixed.append(sp.result)
                        else:
                            fixed.append(o)
                    nd = b.create(d.name, fixed, [(op.result.dtype, op.result.shape)], dict(d.attrs))
                    new_ops = block.ops
                    block.ops = saved
                    idx = block.ops.index(op)
                    for k, nop_ in enumerate(new_ops):
                        nop_.parent = block
                        block.ops.insert(idx + k, nop_)
                    _replace(fn, op.result, nd.result)
                    block.ops.remove(op)
                    changed = True
                    continue
            i += 1
    if changed:
        dce(fn)
    return changed


# ---------------------------------------------------------------------------
# Range analysis
# ---------------------------------------------------------------------------

INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31
PID_MAX = 65535


class RangeAnalysis:
    """Conservative [lo, hi] bounds for i32 values (None = unknown)."""

    def __init__(self, fn: ir.Function):
        self.fn = fn
        self.r = {}
        self.run()

    def get(self, v: Value):
        return self.r.get(v.id)

    def _set(self, v: Value, rng):
        if rng is None:
            return
        lo, hi = rng
        if lo < INT_MIN or hi > INT_MAX:
            return
        self.r[v.id] = (lo, hi)

    def run(self):
        for _ in range(3):
            self._block(self.fn.body)

    def _block(self, block: Block):
        for op in block.ops:
            self._op(op)

    def _op(self, op: Op):
        n = op.name
        g = lambda i: self.get(op.operands[i]) if op.operands[i] is not None else None
        res = op.results[0] if op.results else None
        if n == "arith.constant":
            v = op.attrs["value"]
            if isinstance(v, bool):
                self._set(res, (int(v), int(v)))
            elif isinstance(v, int):
                self._set(res, (v, v))
        elif n == "tt.make_range":
            self._set(res, (op.attrs["start"], op.attrs["end"] - 1))
        elif n == "tt.get_program_id":
            self._set(res, (0, PID_MAX))
        elif n == "tt.get_num_programs":
            self._set(res, (1, PID_MAX + 1))
        elif n in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans",
                   "arith.bitcast"):
            self._set(res, g(0))
        elif n in ("arith.addi", "arith.subi", "arith.muli", "arith.minsi", "arith.maxsi",
                   "arith.divsi", "arith.remsi", "arith.andi", "arith.shli", "arith.shrsi"):
            a, b = g(0), g(1)
            if a is None or b is None:
                if n == "arith.remsi" and b is not None and b[0] > 0 and a is not None and a[0] >= 0:
                    self._set(res, (0, b[1] - 1))
                elif n == "arith.andi" and ((a and a[0] >= 0) or (b and b[0] >= 0)):
                    hi = min(x[1] for x in (a, b) if x is not None and x[0] >= 0)
                    self._set(res, (0, hi))
                elif n == "arith.minsi" and (a or b):
                    known = [x for x in (a, b) if x is not None]
                    self._set(res, (INT_MIN, min(x[1] for x in known)))
                elif n == "arith.maxsi" and (a or b):
                    known = [x for x in (a, b) if x is not None]
                    self._set(res, (max(x[0] for x in known), INT_MAX))
                return
            if n == "arith.addi":
                self._set(res, (a[0] + b[0], a[1] + b[1]))
            elif n == "arith.subi":
                self._set(res, (a[0] - b[1], a[1] - b[0]))
            elif n == "arith.muli":
                c = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
                self._set(res, (min(c), max(c)))
            elif n == "arith.minsi":
                self._set(res, (min(a[0], b[0]), min(a[1], b[1])))
            elif n == "arith.maxsi":
                self._set(res, (max(a[0], b[0]), max(a[1], b[1])))
            elif n == "arith.divsi" and a[0] >= 0 and b[0] > 0:
                self._set(res, (a[0] // b[1], a[1] // b[0]))
            elif n == "arith.remsi" and a[0] >= 0 and b[0] > 0:
                self._set(res, (0, min(a[1], b[1] - 1)))
            elif n == "arith.andi" and a[0] >= 0 and b[0] >= 0:
                self._set(res, (0, min(a[1], b[1])))
            elif n == "arith.shli" and a[0] >= 0 and b[0] == b[1] and 0 <= b[0] < 31:
                self._set(res, (a[0] << b[0], a[1] << b[0]))
            elif n == "arith.shrsi" and b[0] == b[1] and 0 <= b[0] < 32:
                self._set(res, (a[0] >> b[0], a[1] >> b[0]))
        elif n == "arith.select":
            a, b = g(1), g(2)
            if a and b:
                self._set(res, (min(a[0], b[0]), max(a[1], b[1])))
        elif n == "arith.extui":
            self._set(res, (0, 1))
        elif n == "scf.for":
            lb, ub, st = g(0), g(1), g(2)
            body = op.regions[0]
            if lb is not None and ub is not None:
                hi = ub[1] - 1
                self._set(body.args[0], (lb[0], max(lb[0], hi)))
            for i, init in enumerate(op.operands[3:]):
                # loop-carried ints: only the init is known; leave unknown
                pass
            self._block(body)
        elif n == "scf.if":
            for r in op.regions:
                self._block(r)

    def fold_cmp(self, op: Op):
        a, b = self.get(op.operands[0]), self.get(op.operands[1])
        if a is None or b is None:
            return None
        p = op.attrs["predicate"]
        if p == "slt":
            return True if a[1] < b[0] else (False if a[0] >= b[1] else None)
        if p == "sle":
            return True if a[1] <= b[0] else (False if a[0] > b[1] else None)
        if p == "sgt":
            return True if a[0] > b[1] else (False if a[1] <= b[0] else None)
        if p == "sge":
            return True if a[0] >= b[1] else (False if a[1] < b[0] else None)
        return None

    def fits_mul24(self, v: Value) -> bool:
        r = self.get(v)
        return r is not None and r[0] >= 0 and r[1] < (1 << 24)


REMAT_OPS = {"arith.constant", "tt.get_program_id", "tt.get_num_programs", "arith.addi", "arith.subi",
             "arith.muli", "arith.divsi", "arith.remsi", "arith.minsi", "arith.maxsi", "arith.andi",
             "arith.ori", "arith.shli", "arith.shrsi", "arith.shrui", "tt.splat", "tt.expand_dims",
             "tt.broadcast", "tt.make_range", "tt.addptr"}
# tensor ops that are free in the lowering (affine index math)
_FREE_TENSOR_OPS = {"tt.splat", "tt.expand_dims", "tt.broadcast", "tt.make_range", "tt.addptr",
                    "arith.addi", "arith.subi", "arith.muli", "arith.constant"}


def remat_across_loops(fn: ir.Function, max_ops: int = 64) -> int:
    """Recompute scalar index math after a top-level loop instead of keeping
    it live through the loop (register pressure on the QPU is the limit)."""
    top = fn.body.ops
    count = 0
    for li, loop in enumerate(list(top)):
        if loop.name != "scf.for":
            continue
        li = top.index(loop)
        inside = set()

        def rec(o):
            for v in o.operands:
                if v is not None:
                    inside.add(v.id)
            for r in o.regions:
                for oo in r.ops:
                    rec(oo)
        rec(loop)
        before = top[:li]
        after = top[li + 1:]
        after_uses = set()
        for o in after:
            def rec2(o):
                for v in o.operands:
                    if v is not None:
                        after_uses.add(v.id)
                for r in o.regions:
                    for oo in r.ops:
                        rec2(oo)
            rec2(o)
        for op in before:
            if op.name not in REMAT_OPS or not op.results:
                continue
            if op.result.shape and op.name not in _FREE_TENSOR_OPS:
                continue
            v = op.result
            if v.id in inside or v.id not in after_uses or op.name == "arith.constant":
                continue
            # clone v's slice (pure scalar ops on args / program ids)
            slice_ops = []
            ok = True
            seen = set()

            def collect(x):
                nonlocal ok
                if x.op is None or x.id in seen:
                    return
                seen.add(x.id)
                if x.op.name not in REMAT_OPS or x.op.parent is not fn.body or \
                        (x.shape and x.op.name not in _FREE_TENSOR_OPS):
                    ok = False
                    return
                for o in x.op.operands:
                    collect(o)
                slice_ops.append(x.op)
            collect(v)
            if not ok or len(slice_ops) > max_ops:
                continue
            mapping = {}
            insert_at = top.index(loop) + 1
            for so in slice_ops:
                nop = Op(so.name, [mapping.get(o.id, o) for o in so.operands], attrs=dict(so.attrs))
                for r in so.results:
                    nop.results.append(Value(r.dtype, r.shape, nop))
                nop.parent = fn.body
                top.insert(insert_at, nop)
                insert_at += 1
                for r, nr in zip(so.results, nop.results):
                    mapping[r.id] = nr
            new_v = mapping[v.id]
            for o in top[insert_at:]:
                def subst(o):
                    o.operands = [new_v if (x is not None and x.id == v.id) else x for x in o.operands]
                    for r in o.regions:
                        for oo in r.ops:
                            subst(oo)
                subst(o)
            count += 1
    return count


def optimize(fn: ir.Function) -> RangeAnalysis:
    canonicalize(fn)
    for _ in range(4):
        changed = licm(fn)
        changed |= sink_broadcasts(fn)
        canonicalize(fn)
        if not changed:
            break
    ranges = RangeAnalysis(fn)
    canonicalize(fn, ranges)
    licm(fn)
    canonicalize(fn)
    remat_across_loops(fn)
    dce(fn)
    return RangeAnalysis(fn)
