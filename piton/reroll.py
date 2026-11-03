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

With `reductions` (PITON_REROLL_REDUCE=1 or "rows") the kernel may also
reduce such tensors to scalars (tl.sum / tl.max / tl.min) and splat the
results back, as softmax does. Each pass over the row becomes its own loop:

    m = max(x)                     for c: acc_m = fold(max, acc_m, chunk of x)
    e = exp(x - m); s = sum(e) ->  m = max(acc_m); for c: acc_s = fold(+, acc_s, chunk of e)
    store(e / s)                   s = sum(acc_s);  for c: store(chunk of e / s)

A loop recomputes everything its pass needs from the loads (here x is loaded
three times and exp runs twice) and carries a 16-lane partial: each chunk's
registers are combined into it one at a time, in the order the unrolled
reduction combines the row's registers, and its lanes are folded after the
loop exactly as before. The partial starts at a value whose combine with the
first register gives the next combine the same operand as that register:
+0 for float sums (+0 + x is x, except that -0, denormals and NaN come out
as +0, +0 and an infinity, which give the next add, with the row's second
register, the same result), the bit patterns below / above every value in
the fmax / fmin sign-magnitude order, and the integer identities. So the
results are bit-identical to the unrolled kernel's under the float semantics
measured on the Pi (emulator.py). Independent reductions share a loop;
scalar stores move after the last loop, so kernels that load after any
store, or store a tensor after a scalar, are left alone.

transposed_loads() (PITON_REROLL_REDUCE=1 or "tiles") handles the 2-D tile
that missed the cache, transpose: tl.trans of a loaded tile becomes a load
through the transposed pointers, which needs no lane permutation at all.
"""
from __future__ import annotations

from . import ir, passes

SUB = 64                                  # 4 registers: measured best for exp (32: less overlap)
ICACHE_INSTRUCTIONS = 450                 # unrolled programs above this miss on every line
# 16 lines of 32 instructions: programs up to ~540 instructions (scaled_relu,
# mandel_16x16x2) miss < 0.25 times per program on the Pi, and the misses of
# larger ones match a direct-mapped cache of this size
ICACHE_CAPACITY = 512
LANES = 16

_TENSOR_OPS = {"tt.make_range", "tt.splat", "tt.addptr", "tt.load", "tt.store", "tt.bitcast"}
_CARRIED_DTYPES = {"fp32", "int32", "uint32"}
_DIVISIONS = {"arith.divf", "arith.divsi", "arith.divui", "arith.remsi", "arith.remui"}


def _elementwise(op) -> bool:
    return op.name in _TENSOR_OPS or op.name.startswith("arith.") or op.name.startswith("math.")


def _is_reduction(op) -> bool:
    return (op.name == "tt.reduce" and op.attrs.get("kind") in ("sum", "max", "min")
            and len(op.operands) == 1 and not op.result.is_tensor)


def _is_tensor_op(op) -> bool:
    return any(v is not None and v.is_tensor for v in list(op.operands) + list(op.results))


def _memory_order_kept(fn: ir.Function) -> bool:
    """The reduction loops reload their inputs and scalar stores move after the
    last loop: safe when no load follows a store and no tensor store follows a
    scalar one (and no other op has side effects)."""
    stored = scalar_stored = False
    for op in fn.body.ops:
        tensor = _is_tensor_op(op)
        if op.name == "tt.load" and stored:
            return False
        if op.name == "tt.store":
            if tensor and scalar_stored:
                return False
            stored = True
            scalar_stored = scalar_stored or not tensor
        elif not tensor and op.name not in ("tt.load", "tt.return") and not passes.is_pure(op):
            return False
    return True


def _tensor_length(fn: ir.Function, reductions: bool = False):
    """N if every tensor in the kernel is 1-D of length N and rerolling is safe, else None.
    `reductions` also accepts reductions of these tensors to scalars."""
    n = None
    stored = False
    for op in fn.body.ops:
        if op.regions:
            return None
        values = [v for v in list(op.operands) + list(op.results) if v is not None]
        tensors = [v for v in values if v.is_tensor]
        if not tensors:
            continue
        if not _elementwise(op) and not (reductions and _is_reduction(op)):
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
        if op.results and not op.results[0].is_tensor and any(v.is_tensor for v in op.operands if v is not None) \
                and not (reductions and _is_reduction(op)):
            return None                       # a tensor feeding a scalar (a reduction)
    if reductions and has_reductions(fn):
        if n % LANES or not _memory_order_kept(fn):
            return None
    return n


def has_reductions(fn: ir.Function) -> bool:
    return any(_is_reduction(op) for op in fn.body.ops)


def applicable(fn: ir.Function, sub: int = SUB, reductions: bool = False) -> bool:
    n = _tensor_length(fn, reductions)
    return n is not None and n > sub and n % sub == 0


def run(fn: ir.Function, sub: int = SUB, reductions: bool = False, carry: bool = True) -> bool:
    """Rewrite `fn` in place with `sub`-element chunks; returns False (unchanged)
    when not applicable. `carry`: see _run_passes."""
    n = _tensor_length(fn, reductions)
    if n is None or n <= sub or n % sub:
        return False
    if reductions and has_reductions(fn):
        if sub % LANES:
            return False
        _run_passes(fn, n, sub, carry)
        return True
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


def transposed_loads(fn: ir.Function) -> int:
    """tl.trans(tl.load(p, mask, other)) -> tl.load(trans(p), trans(mask), trans(other)).

    Transposing a 16x16 tile moves every element across lanes (a masked
    rotate per element: transpose is ~1400 straight-line instructions, which
    miss the instruction cache on every line). Loading through the transposed
    pointers instead lets the TMU gather each output register directly, one
    address per lane; the transposes of the affine pointers and of the
    index-compare mask cost no instructions (Lowering.trans_val). Returns the
    number of loads rewritten."""
    uses = {}
    for op in fn.walk():
        for v in op.operands:
            if v is not None:
                uses[v.id] = uses.get(v.id, 0) + 1
    count = 0
    blocks = [fn.body] + [r for op in fn.walk() for r in op.regions]
    for block in blocks:
        for op in list(block.ops):
            ld = op.operands[0].op if op.name == "tt.trans" else None
            if ld is None or ld.name != "tt.load" or ld.parent is not block or uses.get(ld.result.id) != 1:
                continue
            m, n = ld.result.shape
            b = ir.Builder(ir.Block())
            operands = [b.create("tt.trans", [v], [(v.dtype, (n, m))], {"order": (1, 0), "load_operand": True}).result
                        for v in ld.operands]
            new = b.create("tt.load", operands, [(ld.result.dtype, (n, m))], dict(ld.attrs))
            for v in ld.operands:
                if ld in v.uses:
                    v.uses.remove(ld)
            at = block.ops.index(ld)
            block.ops[at:at + 1] = []
            for k, o in enumerate(b.block.ops):
                o.parent = block
                block.ops.insert(at + k, o)
            block.ops.remove(op)
            ir.replace_all_uses(op.result, new.result, fn.body)
            count += 1
    return count


def _identity(red: ir.Op, b: ir.Builder) -> ir.Value:
    """A 16-lane partial that the first combine of `red` returns unchanged."""
    dt, kind = red.result.dtype, red.attrs["kind"]
    if dt.is_float and kind != "sum":
        # fmax / fmin order by sign-magnitude bits: -NaN with every bit set is the
        # least value and +NaN the greatest (a python float would not keep the bits)
        bits = b.create("arith.constant", [], [(ir.int32, ())],
                        {"value": -1 if kind == "max" else 0x7FFFFFFF}).result
        init = b.create("arith.bitcast", [bits], [(dt, ())]).result
    else:
        if kind == "sum":
            value = 0.0 if dt.is_float else 0
        elif dt.kind == "uint":
            value = 0 if kind == "max" else 0xFFFFFFFF
        else:
            value = -2 ** 31 if kind == "max" else 2 ** 31 - 1
        init = b.create("arith.constant", [], [(dt, ())], {"value": value}).result
    return b.create("tt.splat", [init], [(dt, (LANES,))]).result


def _recomputed(op: ir.Op, memo: dict) -> bool:
    """Ops that every pass needing them recomputes instead of receiving them
    carried in registers: index arithmetic and masks (no load, math function
    or division among their tensor operands), and values that are not 32-bit
    numbers (masks, pointers), whose operands are carried instead."""
    if op not in memo:
        if op.result.dtype.name not in _CARRIED_DTYPES:
            memo[op] = True
        elif op.name == "tt.load" or op.name.startswith("math.") or op.name in _DIVISIONS:
            memo[op] = False
        else:
            memo[op] = all(v.op is None or _recomputed(v.op, memo)
                           for v in op.operands if v is not None and v.is_tensor)
    return memo[op]


def _run_passes(fn: ir.Function, n: int, sub: int, carry: bool = True) -> None:
    """Reroll a kernel with reductions: a loop per group of independent
    reductions, carrying their 16-lane partials, then a loop for the stores.

    With `carry`, a loaded or computed chunk that a later pass needs is
    computed once and kept in registers: a value spans n / sub loop-carried
    chunks ("slots"); each iteration reads slot 0 and shifts the slots down
    by one, appending its own chunk (or slot 0 again while a later pass
    still needs the value), so after a pass the slots hold the chunks in
    order again. A value whose last use is a pass hands its slots to a value
    that pass produces. Without `carry` every pass recomputes its chunks
    from the loads (fewer registers, more instructions)."""
    ops = list(fn.body.ops)
    order = {op: i for i, op in enumerate(ops)}
    tensor_ops = [op for op in ops if _is_tensor_op(op)]
    scalar_ops = [op for op in ops if not _is_tensor_op(op)]
    reductions = [op for op in tensor_ops if _is_reduction(op)]
    stores = [op for op in tensor_ops if op.name == "tt.store"]
    slots = n // sub

    deps = {}                                 # value id -> the reductions it depends on
    for op in ops:
        d = frozenset().union(*(deps.get(v.id, frozenset()) for v in op.operands if v is not None))
        for r in op.results:
            deps[r.id] = d | {op} if _is_reduction(op) else d
    groups = []
    for red in reductions:
        if groups and not deps[red.operands[0].id] & set(groups[-1]):
            groups[-1].append(red)            # independent of the group: same loop
        else:
            groups.append([red])
    passes_ = groups + ([stores] if stores else [])

    # which ops each pass computes, and which carried values it reads
    memo, computed_at, plan = {}, {}, []
    for p, roots in enumerate(passes_):
        need, reads = set(), set()
        stack = [v for op in roots for v in op.operands if v is not None and v.is_tensor]
        while stack:
            v = stack.pop()
            o = v.op
            if o is None or o in need or o in reads:
                continue
            if o in computed_at:
                reads.add(o)
                continue
            need.add(o)
            stack += [x for x in o.operands if x is not None and x.is_tensor]
        if carry:
            for o in need:
                if not _recomputed(o, memo):
                    computed_at[o] = p
        plan.append((roots, need, reads))
    last = {}
    for p, (_, _, reads) in enumerate(plan):
        for o in reads:
            last[o] = p

    for op in tensor_ops:                     # the originals are replaced by the loop bodies
        for v in op.operands:
            if v is not None and op in v.uses:
                v.uses.remove(op)
    fn.body.ops = []
    b = ir.Builder(fn.body)
    ready = {a.id for a in fn.args}
    pending = [op for op in scalar_ops if op.name not in ("tt.store", "tt.return")]

    def flush():
        """Emit the scalar ops whose operands are available (in program order)."""
        for op in list(pending):
            if all(v is None or v.id in ready for v in op.operands):
                fn.body.append(op)
                ready.update(r.id for r in op.results)
                pending.remove(op)

    def const(value, dtype=ir.int32, builder=b):
        return builder.create("arith.constant", [], [(dtype, ())], {"value": value}).result

    buffers = {}                              # carried op -> its slot values before the next loop

    def chunk_loop(p, roots, need, reads):
        """scf.for over the chunks: the ops of pass p, then each root (a
        reduction folds its chunk into its partial)."""
        reds = [op for op in roots if _is_reduction(op)]
        live = sorted((o for o in buffers if last[o] >= p), key=order.get)
        produced = sorted((o for o in computed_at if computed_at[o] == p and o in last), key=order.get)
        streams = []                          # (op whose slots come in or None, op shifted in or None)
        for o in live:
            streams.append([o, o if last[o] > p else None])
        for o in produced:
            free = next((s for s in streams if s[1] is None and last[s[0]] == p
                         and s[0].result.dtype == o.result.dtype), None)
            if free is not None:
                free[1] = o                   # takes over the slots of a value used for the last time
            else:
                streams.append([None, o])
        inits = [_identity(red, b) for red in reds]
        for old, new in streams:
            if old is not None:
                inits += buffers.pop(old)
            else:
                zero = const(0.0 if new.result.dtype.is_float else 0, new.result.dtype)
                inits += [b.create("tt.splat", [zero], [(new.result.dtype, (sub,))]).result] * slots
        lb, ub, step = const(0), const(slots), const(1)
        body = ir.Block()
        chunk = ir.Value(ir.int32, (), name="chunk")
        body.args.append(chunk)
        args = [ir.Value(v.dtype, v.shape) for v in inits]
        body.args += args
        loop = b.create("scf.for", [lb, ub, step] + inits, [(v.dtype, v.shape) for v in inits], regions=[body])
        inner = ir.Builder(body)
        size = inner.create("arith.constant", [], [(ir.int32, ())], {"value": sub}).result
        start = inner.create("arith.muli", [chunk, size], [(ir.int32, ())]).result
        start_splat = inner.create("tt.splat", [start], [(ir.int32, (sub,))]).result
        mapping = {}
        stream_args = args[len(reds):]
        for k, (old, new) in enumerate(streams):
            if old is not None and old in reads:
                mapping[old.result] = stream_args[k * slots]
        for op in tensor_ops:
            if op not in need and not (op in roots and op.name == "tt.store"):
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
        yields = []
        for red, acc in zip(reds, args):
            # (sub,) -> (sub / 16, 16), one row per register: tt.reduce with an
            # accumulator operand combines the rows into it in order
            x = mapping[red.operands[0]]
            rows = inner.create("tt.reshape", [x], [(x.dtype, (sub // LANES, LANES))]).result
            yields.append(inner.create("tt.reduce", [rows, acc], [(x.dtype, (LANES,))], dict(red.attrs)).result)
        for k, (old, new) in enumerate(streams):
            s = stream_args[k * slots:(k + 1) * slots]
            # slot j <- slot j + 1; the last slot gets this chunk of the value
            # shifted in (or slot 0 again: a full rotation; or nothing new)
            tail = s[0] if new is old else (mapping[new.result] if new is not None else s[-1])
            yields += s[1:] + [tail]
        inner.create("scf.yield", yields)
        results = loop.results[len(reds):]
        for k, (old, new) in enumerate(streams):
            if new is not None:
                buffers[new] = results[k * slots:(k + 1) * slots]
        return loop.results[:len(reds)]

    flush()
    for p, (roots, need, reads) in enumerate(plan):
        if roots is stores:
            chunk_loop(p, roots, need, reads)
            break
        for red, acc in zip(roots, chunk_loop(p, roots, need, reads)):
            # the partial's lanes are folded as the unrolled reduction folds its last register
            red.operands = [acc]
            acc.uses.append(red)
            fn.body.append(red)
            ready.add(red.result.id)
        flush()
    assert not pending, "internal: scalar ops left after rerolling"
    for op in scalar_ops:                     # scalar stores (in order), then the return
        if op.name in ("tt.store", "tt.return"):
            fn.body.append(op)
