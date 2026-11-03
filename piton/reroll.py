"""Reroll a large 1-D elementwise block into a loop over small chunks.

The QPU instruction cache holds roughly 500 instructions. A kernel such as
`exp` over BLOCK=256 unrolls into ~1000 straight-line instructions, so every
program streams its code through the cache (3.06% of fetches miss: one per
32-instruction line) and the misses stall the QPU. Measured on the Pi
(perf/icache_experiment.py): the same 256 elements as a loop over 64-element
chunks run exp 1.63x faster, with the miss rate down to 0.01%.

This pass rewrites the tile IR of kernels whose tensors are all one-
dimensional with the same length N, using only elementwise ops, loads and
stores (no reductions, dots, reshapes or loops):

    offs = base + arange(0, N); ...           for c in range(N // SUB):
    y = f(load(x + offs, mask)); store(...)  ->     offs = base + c*SUB + arange(0, SUB); ...

Scalar ops (program id, pointer arithmetic on scalars) stay outside the loop.
Kernels that load after a store are left alone, since chunking reorders their
memory accesses. compile() applies it only when the unrolled program would
not fit the instruction cache.
"""
from __future__ import annotations

from . import ir

SUB = 64                                  # 4 registers: measured best for exp (32: less overlap)
ICACHE_INSTRUCTIONS = 450                 # unrolled programs above this miss on every line

_TENSOR_OPS = {"tt.make_range", "tt.splat", "tt.addptr", "tt.load", "tt.store", "tt.bitcast"}


def _elementwise(op) -> bool:
    return op.name in _TENSOR_OPS or op.name.startswith("arith.") or op.name.startswith("math.")


def _tensor_length(fn: ir.Function):
    """N if every tensor in the kernel is 1-D of length N and rerolling is safe, else None."""
    n = None
    stored = False
    for op in fn.body.ops:
        if op.regions:
            return None
        values = [v for v in list(op.operands) + list(op.results) if v is not None]
        tensors = [v for v in values if v.is_tensor]
        if not tensors:
            continue
        if not _elementwise(op):
            return None
        for v in tensors:
            if len(v.shape) != 1 or (n is not None and v.shape[0] != n):
                return None
            n = v.shape[0]
        if op.name == "tt.make_range" and op.attrs.get("start") != 0:
            return None
        if op.name == "tt.load" and stored:
            return None                       # a load after a store might read what it wrote
        if op.name == "tt.store":
            stored = True
        if op.results and not op.results[0].is_tensor and any(v.is_tensor for v in op.operands if v is not None):
            return None                       # a tensor feeding a scalar (a reduction)
    return n


def applicable(fn: ir.Function, sub: int = SUB) -> bool:
    n = _tensor_length(fn)
    return n is not None and n > sub and n % sub == 0


def run(fn: ir.Function, sub: int = SUB) -> bool:
    """Rewrite `fn` in place with `sub`-element chunks; returns False (unchanged)
    when not applicable."""
    n = _tensor_length(fn)
    if n is None or n <= sub or n % sub:
        return False
    scalar_ops, tensor_ops = [], []
    for op in fn.body.ops:
        values = [v for v in list(op.operands) + list(op.results) if v is not None]
        (tensor_ops if any(v.is_tensor for v in values) else scalar_ops).append(op)
    ret = [op for op in scalar_ops if op.name == "tt.return"]
    scalar_ops = [op for op in scalar_ops if op.name != "tt.return"]

    for op in tensor_ops:                     # the originals are replaced by the loop body
        for v in op.operands:
            if v is not None and op in v.uses:
                v.uses.remove(op)
    fn.body.ops = []
    b = ir.Builder(fn.body)
    for op in scalar_ops:
        fn.body.append(op)

    def const(value):
        return b.create("arith.constant", [], [(ir.int32, ())], {"value": value}).result

    lb, ub, step = const(0), const(n // sub), const(1)
    body = ir.Block()
    chunk = ir.Value(ir.int32, (), name="chunk")
    body.args.append(chunk)
    loop = b.create("scf.for", [lb, ub, step], [], regions=[body])
    inner = ir.Builder(body)
    size = inner.create("arith.constant", [], [(ir.int32, ())], {"value": sub}).result
    start = inner.create("arith.muli", [chunk, size], [(ir.int32, ())]).result
    start_splat = inner.create("tt.splat", [start], [(ir.int32, (sub,))]).result
    mapping = {}
    for op in tensor_ops:
        operands = [mapping.get(v, v) if v is not None else None for v in op.operands]
        if op.name == "tt.make_range":
            r = inner.create("tt.make_range", [], [(ir.int32, (sub,))], {"start": 0, "end": sub}).result
            new = inner.create("arith.addi", [start_splat, r], [(ir.int32, (sub,))])
        else:
            new = inner.create(op.name, operands, [(v.dtype, (sub,) if v.is_tensor else v.shape)
                                                   for v in op.results], dict(op.attrs))
        for old, res in zip(op.results, new.results):
            mapping[old] = res
    inner.create("scf.yield", [])
    for op in ret:
        fn.body.append(op)
    return loop is not None
