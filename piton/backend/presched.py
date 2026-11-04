"""Pre-allocation scheduling: interleave independent dependence chains.

Lowering emits a tile's work one 16-lane register at a time, so an exp over a
[256] block is sixteen serial chains, one after another. Register allocation
then reuses the same accumulators for every chain, and the false dependences
that creates stop the post-allocation scheduler (sched.py) from overlapping
the chains: most instructions keep one ALU idle (perf/analyze.py: 80-100% of
unpaired neighbours in exp/log/sqrt are blocked by a dependence).

This pass reorders each block before allocation, by critical-path height, so
chains overlap while values are still virtual. Only pure ALU instructions and
constant loads on ordinary virtual registers move; everything with a side
effect or a fixed register (TMU, VPM, uniforms, r4/r5, flags-carrying
specials) keeps its relative order, and branches, semaphores and thread ends
are barriers. A soft register-pressure limit stops it from starting more
chains than the register file can hold; compile() also keeps the original
order whenever that allocates better or schedules shorter.

Flag groups (PITON_ISSUE=1)
---------------------------
There is one set of condition flags, and the special-value checks of the
math kernels set them several times per register (exp: 4, recip: 8), so
with every flag write ordered like a side effect the chains of a 64-element
chunk cannot overlap past their first compare. With `flags=True` a flag
write and the conditional instructions that read it (its group) move
together instead: flag-only instructions (`sub.setf -, a, b`) leave the
side-effect order, a group's writer is placed only once everything its
readers need besides the writer itself is placed (so the group closes right
after it), and no other writer is placed while a group is open. Groups of
independent chains then interleave; the last group of a block stays last
(its flags may be read by the branch or a successor), conditional reads of
flags set before the block stay ahead of its first writer, and a block with
a barrier before its end or an instruction that both reads and writes the
flags keeps the old order.
"""
from __future__ import annotations

from ..qpu import isa
from .mir import Function, MInst, VReg

# Live virtual registers above which the pass prefers instructions that free
# values over ones on the critical path (4 accumulators + 2 x 31 regfile
# entries, leaving room for values live across the block).
PRESSURE_LIMIT = 48


def _pure(ins: MInst) -> bool:
    if ins.kind == "ldi":
        return all(isinstance(op.dst, VReg) and op.dst.cls in ("any", "acc", "regfile") for op in ins.ops)
    if ins.kind != "alu" or ins.sig is not None:
        return False
    for op in ins.ops:
        if not isinstance(op.dst, VReg) or op.dst.cls not in ("any", "acc", "regfile"):
            return False
        for s in op.srcs:
            if isinstance(s, VReg):
                if s.cls not in ("any", "acc", "regfile"):
                    return False
            elif not hasattr(s, "code"):          # a physical register (Imm is fine)
                return False
    return True


def _flag_only(ins: MInst) -> bool:
    """A pure instruction except that ops may write only the flags (dst NOP)."""
    if ins.kind != "alu" or ins.sig is not None:
        return False
    for op in ins.ops:
        if isinstance(op.dst, VReg):
            if op.dst.cls not in ("any", "acc", "regfile"):
                return False
        elif not (op.dst == isa.NOP and op.setf):
            return False
        for s in op.srcs:
            if isinstance(s, VReg):
                if s.cls not in ("any", "acc", "regfile"):
                    return False
            elif not hasattr(s, "code"):
                return False
    return True


def _barrier(ins: MInst) -> bool:
    return ins.kind in ("branch", "sem") or ins.sig == "thrend"


def schedule_block(insts: list, flags: bool = False) -> list:
    n = len(insts)
    if n < 3:
        return insts
    if flags:
        order = _grouped_order(insts)
        if order is not None:
            return [insts[i] for i in order]
    succs = [set() for _ in range(n)]
    preds = [set() for _ in range(n)]

    def edge(a, b):
        if a != b and b not in succs[a]:
            succs[a].add(b)
            preds[b].add(a)

    _data_edges(insts, edge, [_pure(i) for i in insts], flag_edges=True)
    order = _list_schedule(insts, succs, preds)
    assert len(order) == n, "dependence cycle in pre-scheduling"
    return [insts[i] for i in order]


def _data_edges(insts, edge, pure, flag_edges):
    """Register, side-effect and barrier edges (and, with flag_edges, every
    flag write and read in program order)."""
    last_def, readers = {}, {}
    last_flag_write, flag_readers = None, []
    last_ordered, last_barrier = None, None
    for j, ins in enumerate(insts):
        for v in ins.uses():
            if v in last_def:
                edge(last_def[v], j)                  # RAW
        for v in ins.defs():
            if v in last_def:
                edge(last_def[v], j)                  # WAW
            for r in readers.get(v, ()):
                edge(r, j)                            # WAR
        if flag_edges:
            if ins.reads_flags and last_flag_write is not None:
                edge(last_flag_write, j)
            if ins.writes_flags:
                if last_flag_write is not None:
                    edge(last_flag_write, j)
                for r in flag_readers:
                    edge(r, j)
        if not pure[j] and last_ordered is not None:
            edge(last_ordered, j)                     # side effects keep their order
        if last_barrier is not None:
            edge(last_barrier, j)
        if _barrier(ins):
            for i in range(j):
                edge(i, j)
            last_barrier = j
        # bookkeeping after the edges for this instruction
        for v in ins.uses():
            readers.setdefault(v, []).append(j)
        for v in ins.defs():
            last_def[v] = j
            readers[v] = []
        if ins.writes_flags:
            last_flag_write, flag_readers = j, []
        if ins.reads_flags:
            flag_readers.append(j)
        if not pure[j]:
            last_ordered = j


def _list_schedule(insts, succs, preds, writer=None, live_in=()):
    """Critical-path list order (pressure-limited). With `writer` (instruction
    -> its group's reader set), no writer is taken while a group is open;
    returns a shorter list if that leaves nothing to take."""
    n = len(insts)
    height = [1] * n
    for i in range(n - 1, -1, -1):
        if succs[i]:
            height[i] = 1 + max(height[s] for s in succs[i])

    # values still read later (or live out of the block): uses remaining per vreg
    remaining = {}
    for ins in insts:
        for v in ins.uses():
            remaining[v] = remaining.get(v, 0) + 1
    live = set()
    waiting = [len(p) for p in preds]
    ready = [i for i in range(n) if waiting[i] == 0]
    order = []
    open_readers = set(live_in)
    while ready:
        cands = ready if not open_readers or writer is None else [i for i in ready if i not in writer]
        if not cands:
            break
        if len(live) >= PRESSURE_LIMIT:
            def key(i):
                frees = sum(1 for v in set(insts[i].uses()) if remaining.get(v, 0) == insts[i].uses().count(v))
                return (frees - len(insts[i].defs()), height[i], -i)
        else:
            def key(i):
                return (height[i], -i)
        best = max(cands, key=key)
        ready.remove(best)
        order.append(best)
        if writer is not None:
            if best in writer:
                open_readers = set(writer[best])
            open_readers.discard(best)
        ins = insts[best]
        for v in ins.uses():
            remaining[v] -= 1
            if remaining[v] == 0:
                live.discard(v)
        for v in ins.defs():
            if remaining.get(v, 0):
                live.add(v)
        for s in succs[best]:
            waiting[s] -= 1
            if waiting[s] == 0:
                ready.append(s)
    return order


def _grouped_order(insts):
    """The flag-group order (module docstring), or None to keep the old one."""
    n = len(insts)
    for j, ins in enumerate(insts):
        if ins.reads_flags and ins.writes_flags:
            return None
        if _barrier(ins) and j != n - 1:
            return None
    succs = [set() for _ in range(n)]
    preds = [set() for _ in range(n)]

    def edge(a, b):
        if a != b and b not in succs[a]:
            succs[a].add(b)
            preds[b].add(a)

    pure = [_pure(i) or _flag_only(i) for i in insts]
    _data_edges(insts, edge, pure, flag_edges=False)
    writer, live_in, cur = {}, [], None
    for j, ins in enumerate(insts):
        if _barrier(ins):
            continue                    # the last instruction: after everything anyway
        if ins.writes_flags:
            cur = j
            writer[j] = []
        elif ins.reads_flags:
            (writer[cur] if cur is not None else live_in).append(j)
    if not writer and not live_in:
        order = _list_schedule(insts, succs, preds)
        return order if len(order) == n else None
    for w, rs in writer.items():
        for r in rs:
            edge(w, r)
    for r in live_in:
        for w in writer:
            edge(r, w)
    if writer:
        last = max(writer)
        for w, rs in writer.items():
            if w != last:
                for x in [w] + rs:
                    edge(x, last)
    # a writer waits for whatever its readers need besides the writer itself
    for w, rs in writer.items():
        below = _descendants(w, succs, n)
        stack, seen = list(rs), set(rs)
        while stack:
            x = stack.pop()
            for p in list(preds[x]):
                if p == w:
                    continue
                if (below >> p) & 1:
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
                else:
                    edge(p, w)
    order = _list_schedule(insts, succs, preds, writer=writer, live_in=live_in)
    return order if len(order) == n else None


def _descendants(i, succs, n) -> int:
    """Bit set of the instructions reachable from i."""
    out, stack = 0, [i]
    while stack:
        x = stack.pop()
        for s in succs[x]:
            if not (out >> s) & 1:
                out |= 1 << s
                stack.append(s)
    return out


def run(fn: Function, flags: bool = False) -> int:
    """Reorder every block in place; returns the number of blocks changed.
    flags: let flag groups of independent chains interleave (PITON_ISSUE)."""
    changed = 0
    for b in fn.blocks:
        new = schedule_block(b.insts, flags)
        if any(x is not y for x, y in zip(new, b.insts)):
            b.insts = new
            changed += 1
    return changed
