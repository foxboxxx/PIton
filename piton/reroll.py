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

With `queue` (PITON_STORE_SLOTS=1) a store of at most one VPM slot (256
elements, backend/lower.py) whose address and mask do not depend on a load
stays outside the loop: the loop carries the N stored values in registers,
shifting each chunk in (`piton.shift_in`: N/16 - SUB/16 register moves per
chunk), and the full block is stored after the loop, one 1 KiB DMA instead
of N/SUB DMAs of SUB*4 bytes.
"""
from __future__ import annotations

from . import ir

SUB = 64                                  # 4 registers: measured best for exp (32: less overlap)
ICACHE_INSTRUCTIONS = 450                 # unrolled programs above this miss on every line
QUEUE_ELEMENTS = 256                      # `queue`: one store slot (16 VPM rows)
QUEUE_CHUNKS = 4                          # `queue`: at most this many chunks shifted through

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


def _queued_stores(tensor_ops, n: int, sub: int):
    """(stores, the ops computing their addresses and masks) for `queue`, or None."""
    stores = [op for op in tensor_ops if op.name == "tt.store"]
    if not stores or n > QUEUE_ELEMENTS or n // sub > QUEUE_CHUNKS:
        return None
    producer = {r: op for op in tensor_ops for r in op.results}
    if any(st.operands[1] not in producer for st in stores):
        return None                           # (the stored value is not a chunked tensor)
    needed = set()
    work = [v for st in stores for i, v in enumerate(st.operands) if i != 1 and v is not None]
    while work:
        op = producer.get(work.pop())
        if op is None or op in needed:
            continue
        if op.name == "tt.load":
            return None                       # the address or mask depends on loaded data
        needed.add(op)
        work.extend(v for v in op.operands if v is not None)
    return stores, [op for op in tensor_ops if op in needed or op in stores]


def run(fn: ir.Function, sub: int = SUB, queue: bool = False) -> bool:
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
    stores, after = (_queued_stores(tensor_ops, n, sub) if queue else None) or ([], [])

    for op in tensor_ops:                     # the originals are replaced by the loop body
        if op in after:
            continue                          # (moved after the loop, as they are)
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
    inits = []
    for st in stores:                         # the stored values, shifted in chunk by chunk
        dtype = st.operands[1].dtype
        zero = b.create("arith.constant", [], [(dtype, ())], {"value": 0.0 if dtype.is_float else 0})
        inits.append(b.create("tt.splat", [zero.result], [(dtype, (n,))]).result)
    body = ir.Block()
    chunk = ir.Value(ir.int32, (), name="chunk")
    body.args.append(chunk)
    body.args.extend(ir.Value(v.dtype, (n,), name="queue") for v in inits)
    loop = b.create("scf.for", [lb, ub, step] + inits, [(v.dtype, (n,)) for v in inits], regions=[body])
    inner = ir.Builder(body)
    size = inner.create("arith.constant", [], [(ir.int32, ())], {"value": sub}).result
    start = inner.create("arith.muli", [chunk, size], [(ir.int32, ())]).result
    start_splat = inner.create("tt.splat", [start], [(ir.int32, (sub,))]).result
    mapping = {}
    for op in tensor_ops:
        if op in stores:
            continue
        operands = [mapping.get(v, v) if v is not None else None for v in op.operands]
        if op.name == "tt.make_range":
            r = inner.create("tt.make_range", [], [(ir.int32, (sub,))], {"start": 0, "end": sub}).result
            new = inner.create("arith.addi", [start_splat, r], [(ir.int32, (sub,))])
        else:
            new = inner.create(op.name, operands, [(v.dtype, (sub,) if v.is_tensor else v.shape)
                                                   for v in op.results], dict(op.attrs))
        for old, res in zip(op.results, new.results):
            mapping[old] = res
    shifted = [inner.create("piton.shift_in", [q, mapping[st.operands[1]]], [(q.dtype, (n,))]).result
               for q, st in zip(body.args[1:], stores)]
    inner.create("scf.yield", shifted)
    for op in after:
        if op in stores:                      # store the loop's result instead
            st_val = op.operands[1]
            if op in st_val.uses:
                st_val.uses.remove(op)
            op.operands[1] = loop.results[stores.index(op)]
            op.operands[1].uses.append(op)
        fn.body.append(op)
    for op in ret:
        fn.body.append(op)
    return loop is not None
