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
"""
from __future__ import annotations

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


def _barrier(ins: MInst) -> bool:
    return ins.kind in ("branch", "sem") or ins.sig == "thrend"


def schedule_block(insts: list) -> list:
    n = len(insts)
    if n < 3:
        return insts
    succs = [set() for _ in range(n)]
    preds = [set() for _ in range(n)]

    def edge(a, b):
        if a != b and b not in succs[a]:
            succs[a].add(b)
            preds[b].add(a)

    last_def, readers = {}, {}
    last_flag_write, flag_readers = None, []
    last_ordered, last_barrier = None, None
    pure = [_pure(i) for i in insts]
    for j, ins in enumerate(insts):
        for v in ins.uses():
            if v in last_def:
                edge(last_def[v], j)                  # RAW
        for v in ins.defs():
            if v in last_def:
                edge(last_def[v], j)                  # WAW
            for r in readers.get(v, ()):
                edge(r, j)                            # WAR
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
    while ready:
        if len(live) >= PRESSURE_LIMIT:
            def key(i):
                frees = sum(1 for v in set(insts[i].uses()) if remaining.get(v, 0) == insts[i].uses().count(v))
                return (frees - len(insts[i].defs()), height[i], -i)
        else:
            def key(i):
                return (height[i], -i)
        best = max(ready, key=key)
        ready.remove(best)
        order.append(best)
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
    assert len(order) == n, "dependence cycle in pre-scheduling"
    return [insts[i] for i in order]


def run(fn: Function) -> int:
    """Reorder every block in place; returns the number of blocks changed."""
    changed = 0
    for b in fn.blocks:
        new = schedule_block(b.insts)
        if any(x is not y for x, y in zip(new, b.insts)):
            b.insts = new
            changed += 1
    return changed
