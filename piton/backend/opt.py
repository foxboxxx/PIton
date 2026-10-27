"""Machine-IR clean-ups run before register allocation."""

from __future__ import annotations

from collections import defaultdict

from ..qpu import isa
from .mir import MInst, VReg


def _has_side_effects(ins: MInst) -> bool:
    if ins.kind != "alu" or ins.sig is not None:
        return True
    for op in ins.ops:
        if op.setf or not isinstance(op.dst, VReg):
            return True
        if op.op == "r4def":
            return True
        for s in op.srcs:
            if isinstance(s, isa.Reg) and s.acc is None and s.raddr in (32, 48, 49, 50, 51):
                return True     # uniform / VPM / DMA wait / mutex reads advance hardware state
    return False


def _removable(ins: MInst) -> bool:
    if ins.kind == "ldi":
        return all(isinstance(op.dst, VReg) and not op.setf for op in ins.ops)
    return ins.kind == "alu" and not _has_side_effects(ins)


def dce(fn) -> int:
    removed = 0
    while True:
        used = set()
        for b in fn.blocks:
            for ins in b.insts:
                for v in ins.uses():
                    used.add(v)
        n = 0
        for b in fn.blocks:
            keep = []
            for ins in b.insts:
                if _removable(ins) and ins.defs() and all(d not in used for d in ins.defs()):
                    n += 1
                    continue
                keep.append(ins)
            b.insts = keep
        removed += n
        if not n:
            return removed


def coalesce_copies(fn) -> int:
    """`D: y = f(...); ...; mov x, y` with y used once -> `D: x = f(...)`."""
    defs = defaultdict(int)
    uses = defaultdict(int)
    for b in fn.blocks:
        for ins in b.insts:
            for d in ins.defs():
                defs[d] += 1
            for u in ins.uses():
                uses[u] += 1
    done = 0
    for b in fn.blocks:
        i = 0
        while i < len(b.insts):
            ins = b.insts[i]
            if ins.kind == "alu" and len(ins.ops) == 1 and ins.sig is None:
                op = ins.ops[0]
                if op.op == "mov" and op.cond == "always" and not op.setf and op.rot is None \
                        and isinstance(op.dst, VReg) and op.srcs and isinstance(op.srcs[0], VReg):
                    x, y = op.dst, op.srcs[0]
                    if y.cls in ("any", "regfile") and x.cls in ("any", "regfile") and \
                            defs[y] == 1 and uses[y] == 1 and x is not y:
                        j = _find_def(b, i, y)
                        if j is not None and _free_between(b, j, i, x):
                            d = b.insts[j]
                            for dop in d.ops:
                                if dop.dst is y and dop.cond == "always":
                                    dop.dst = x
                            del b.insts[i]
                            defs[x] += 1
                            done += 1
                            continue
            i += 1
    return done


def _find_def(b, i, y):
    for j in range(i - 1, -1, -1):
        ins = b.insts[j]
        if y in ins.defs():
            if ins.kind != "alu":
                return None
            return j
        if y in ins.uses():
            return None
    return None


def _free_between(b, j, i, x) -> bool:
    for k in range(j + 1, i):
        ins = b.insts[k]
        if x in ins.uses() or x in ins.defs():
            return False
    return True


def drop_self_moves(fn) -> int:
    """After allocation: delete `mov rX, rX`."""
    n = 0
    for b in fn.blocks:
        keep = []
        for ins in b.insts:
            if ins.kind == "alu" and len(ins.ops) == 1 and ins.sig is None:
                op = ins.ops[0]
                if op.op == "mov" and op.cond == "always" and not op.setf and op.rot is None \
                        and isinstance(op.dst, VReg) and op.srcs and isinstance(op.srcs[0], VReg) \
                        and op.dst.phys is not None and op.dst.phys == op.srcs[0].phys \
                        and op.dst.cls != "r5":
                    n += 1
                    continue
            keep.append(ins)
        b.insts = keep
    return n


def _loops(fn):
    """(header index, latch index) for every back edge, innermost first."""
    idx = {b.label: i for i, b in enumerate(fn.blocks)}
    out = []
    for i, b in enumerate(fn.blocks):
        if b.insts and b.insts[-1].kind == "branch":
            t = idx.get(b.insts[-1].target)
            if t is not None and t <= i:
                out.append((t, i))
    out.sort(key=lambda hl: hl[1] - hl[0])
    return out


def _pure_inst(ins: MInst) -> bool:
    if ins.kind == "ldi":
        return len(ins.ops) == 1 and isinstance(ins.ops[0].dst, VReg) and not ins.ops[0].setf \
            and ins.ops[0].cond == "always"
    if ins.kind != "alu" or ins.sig is not None or len(ins.ops) != 1:
        return False
    op = ins.ops[0]
    if op.setf or op.cond != "always" or not isinstance(op.dst, VReg) or op.dst.cls in ("r4", "r5", "acc"):
        return False
    if op.op == "r4def" or op.rot is not None:
        return False
    for s in op.srcs:
        if isinstance(s, isa.Reg) and s.acc is not None:
            return False
        if isinstance(s, isa.Reg) and s not in (isa.ELEM_NUM, isa.QPU_NUM):
            return False
    return True


def licm(fn) -> int:
    hoisted = 0
    for head, latch in _loops(fn):
        if head == 0:
            continue
        loop_blocks = fn.blocks[head:latch + 1]
        pre = fn.blocks[head - 1]
        changed = True
        while changed:
            changed = False
            defs_in = defaultdict(int)
            all_defs = defaultdict(int)
            for b in fn.blocks:
                for ins in b.insts:
                    for d in ins.defs():
                        all_defs[d] += 1
                    for op in ins.ops:
                        if op.cond != "always" and isinstance(op.dst, VReg):
                            all_defs[op.dst] += 1
            for b in loop_blocks:
                for ins in b.insts:
                    for d in ins.defs():
                        defs_in[d] += 1
                    for op in ins.ops:
                        if isinstance(op.dst, VReg):
                            defs_in[op.dst] += 1
            for b in loop_blocks:
                seen_uses = set()
                for ins in list(b.insts):
                    if _pure_inst(ins):
                        d = ins.ops[0].dst
                        srcs = ins.ops[0].srcs if ins.kind == "alu" else []
                        if all_defs[d] == 1 and d not in seen_uses and \
                                all(not isinstance(x, VReg) or defs_in[x] == 0 for x in srcs):
                            b.insts.remove(ins)
                            pos = len(pre.insts)
                            if pre.insts and pre.insts[-1].kind == "branch":
                                pos -= 1
                                if pos > 0 and pre.insts[pos - 1].writes_flags:
                                    pos -= 1
                            pre.insts.insert(pos, ins)
                            defs_in[d] = 0
                            hoisted += 1
                            changed = True
                            continue
                    for u in ins.uses():
                        seen_uses.add(u)
    return hoisted


def run(fn):
    stats = {}
    stats["licm"] = licm(fn)
    stats["dce"] = dce(fn)
    stats["coalesced"] = coalesce_copies(fn)
    stats["dce2"] = dce(fn)
    return stats
