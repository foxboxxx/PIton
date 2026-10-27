"""Register allocation for the QPU.

The QPU has four general accumulators (r0-r3, no read-after-write delay),
and two 32-entry register files A and B.  An instruction has one read port
per register file, and a small immediate (or a mul-output rotation)
occupies the B port.  So the two register operands of an operation must sit
in *different* files unless one of them is an accumulator.

Algorithm:
  1. precise liveness -> live segments per virtual register
  2. accumulators: required ones (rotation sources) and short-lived
     temporaries, first-fit over r0-r3
  3. remaining values: A/B file assignment as a 2-colouring of the
     "must differ" graph (one edge per pair of register operands of an
     instruction), each connected component oriented to balance pressure
  4. first-fit register assignment inside each file
  5. every instruction is re-checked for encodability; operands that still
     conflict get a copy into a fresh accumulator temporary and the whole
     allocation is redone.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..qpu import isa
from ..qpu.isa import Imm, Reg
from .mir import MInst, MOp, VReg, alu

RESERVED = {14}          # thread-end writes W/Z into address 14 of both files
NUM_ACC = 4


class RegAllocError(Exception):
    pass


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _positions(fn):
    pos = {}
    p = 0
    bounds = {}
    for b in fn.blocks:
        start = p
        for ins in b.insts:
            pos[id(ins)] = p
            p += 2
        bounds[b.label] = (start, max(start, p - 1))
    return pos, bounds


def _inst_defs(ins: MInst):
    return [op.dst for op in ins.ops if isinstance(op.dst, VReg) and op.cond == "always"]


def _inst_uses(ins: MInst):
    out = []
    for op in ins.ops:
        out += [s for s in op.srcs if isinstance(s, VReg)]
        if op.cond != "always" and isinstance(op.dst, VReg):
            out.append(op.dst)
    return out


def liveness(fn):
    fn.compute_cfg()
    gen, kill = {}, {}
    for b in fn.blocks:
        g, k = set(), set()
        for ins in b.insts:
            for u in _inst_uses(ins):
                if u not in k:
                    g.add(u)
            for d in _inst_defs(ins):
                k.add(d)
        gen[b.label], kill[b.label] = g, k
    live_in = {b.label: set() for b in fn.blocks}
    live_out = {b.label: set() for b in fn.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(fn.blocks):
            out = set()
            for s in b.succs:
                out |= live_in[s]
            inn = gen[b.label] | (out - kill[b.label])
            if out != live_out[b.label] or inn != live_in[b.label]:
                live_out[b.label], live_in[b.label] = out, inn
                changed = True
    return live_in, live_out


def live_segments(fn):
    """vreg -> list of inclusive (start, end) position ranges."""
    pos, bounds = _positions(fn)
    live_in, live_out = liveness(fn)
    segs = defaultdict(list)
    for b in fn.blocks:
        start, end = bounds[b.label]
        live = set(live_out[b.label])
        cur_end = {v: end for v in live}
        for ins in reversed(b.insts):
            p = pos[id(ins)]
            for d in _inst_defs(ins):
                if d in live:
                    segs[d].append((p + 1, cur_end[d]))
                    live.discard(d)
                else:
                    segs[d].append((p + 1, p + 1))
            for u in _inst_uses(ins):
                if u not in live:
                    live.add(u)
                    cur_end[u] = p
        for v in live:
            segs[v].append((start, cur_end[v]))
    for v in segs:
        segs[v] = _merge(segs[v])
    return segs, pos


def _merge(ss):
    ss = sorted(ss)
    out = []
    for s, e in ss:
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _overlap(a, b) -> bool:
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] < b[j][0]:
            i += 1
        elif b[j][1] < a[i][0]:
            j += 1
        else:
            return True
    return False


# ---------------------------------------------------------------------------
# Port constraints
# ---------------------------------------------------------------------------


def _special_ports(ins: MInst):
    """(fixedA, fixedB, flexible) port usage of non-vreg operands."""
    fixed_a = fixed_b = False
    flex = set()
    imm_vals = set()
    for op in ins.ops:
        if op.rot is not None and op.rot != 0:
            fixed_b = True
        for s in op.srcs:
            if isinstance(s, Imm):
                imm_vals.add(s.value)
            elif isinstance(s, Reg) and s.acc is None and s.rfile is not None:
                if s.rfile == "A":
                    fixed_a = True
                elif s.rfile == "B":
                    fixed_b = True
                else:
                    flex.add(s.raddr)
    if imm_vals:
        fixed_b = True
    return fixed_a, fixed_b, flex


def _vreg_operands(ins: MInst):
    seen = []
    for op in ins.ops:
        for s in op.srcs:
            if isinstance(s, VReg) and s.cls not in ("r4", "r5") and s not in seen:
                seen.append(s)
    return seen


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


class Allocator:
    def __init__(self, fn, verbose=False):
        self.fn = fn
        self.verbose = verbose
        self.copies_inserted = 0

    def run(self, max_rounds=8):
        for _ in range(max_rounds):
            self.allocate()
            bad = self.check()
            if not bad:
                return
            for blk, ins, operand in bad:
                self.insert_copy(blk, ins, operand)
        raise RegAllocError("could not satisfy register-file port constraints")

    # -- one allocation round ------------------------------------------------

    def allocate(self):
        fn = self.fn
        segs, pos = live_segments(fn)
        self.segs = segs
        vregs = list(segs)
        for v in vregs:
            v.phys = None
        # fixed classes
        for v in vregs:
            if v.cls == "r4":
                v.phys = isa.R4
            elif v.cls == "r5":
                v.phys = isa.R5

        # constraints
        differ = defaultdict(set)
        not_file = defaultdict(set)    # vreg -> {'A','B'} files it must avoid
        must_acc = set(v for v in vregs if v.cls == "acc")
        copy_pairs = []
        uses = defaultdict(int)
        for b in fn.blocks:
            for ins in b.insts:
                ops = [v for v in _vreg_operands(ins) if v.cls != "acc"]
                for v in ops:
                    uses[v] += 1
                fa, fb, flex = _special_ports(ins)
                free = (not fa) + (not fb) - len(flex)
                if free <= 0:
                    must_acc.update(ops)
                    continue
                if len(ops) > free:
                    for v in ops[free:]:
                        must_acc.add(v)
                    ops = ops[:free]
                for v in ops:
                    if fa:
                        not_file[v].add("A")
                    if fb:
                        not_file[v].add("B")
                if len(ops) == 2:
                    a, c = ops
                    differ[a].add(c)
                    differ[c].add(a)
                # two regfile destinations in one (atomic) instruction
                dsts = [op.dst for op in ins.ops if isinstance(op.dst, VReg) and op.dst.cls in ("any", "regfile")]
                if len(dsts) == 2 and dsts[0] is not dsts[1]:
                    differ[dsts[0]].add(dsts[1])
                    differ[dsts[1]].add(dsts[0])
                for op in ins.ops:
                    if op.op == "mov" and isinstance(op.dst, VReg) and op.srcs and isinstance(op.srcs[0], VReg) \
                            and op.cond == "always" and op.rot is None:
                        copy_pairs.append((op.dst, op.srcs[0]))
        self.must_acc = must_acc

        # -- accumulators ----------------------------------------------------
        acc_owner = [[] for _ in range(NUM_ACC)]    # list of segment lists

        def acc_fits(v, i):
            return all(not _overlap(segs[v], s) for s in acc_owner[i])

        required = [v for v in vregs if v in must_acc and v.phys is None]
        required.sort(key=lambda v: segs[v][0][0])
        for v in required:
            if v.cls == "regfile":
                v.cls = "any"
            placed = False
            pref = [v.hint] if isinstance(v.hint, int) else []
            for i in pref + list(range(NUM_ACC)):
                if acc_fits(v, i):
                    v.phys = isa.ACC[i]
                    acc_owner[i].append(segs[v])
                    placed = True
                    break
            if not placed:
                start = segs[v][0][0]
                holders = []
                for i in range(NUM_ACC):
                    for w in required:
                        if w.phys is isa.ACC[i] and _overlap(segs[w], [(start, start)]):
                            holders.append(f"r{i}:{w.name}{w.id}[{w.cls}] segs={segs[w][:3]}")
                raise RegAllocError(f"out of accumulators for {v!r} at position {start} "
                                    f"(held by {', '.join(holders)})")

        def length(v):
            s = segs[v]
            return s[-1][1] - s[0][0]

        cand = [v for v in vregs if v.phys is None and v.cls == "any" and length(v) <= 12]
        cand.sort(key=lambda v: (length(v), segs[v][0][0]))
        for v in cand:
            for i in range(NUM_ACC):
                if acc_fits(v, i):
                    v.phys = isa.ACC[i]
                    acc_owner[i].append(segs[v])
                    break

        # -- register files ----------------------------------------------------
        rest = [v for v in vregs if v.phys is None]
        nodes = set(rest)
        npos = max((s[-1][1] for s in segs.values()), default=0) + 2
        comps = []
        color = {}
        conflict = set()
        for v in sorted(rest, key=lambda x: segs[x][0][0]):
            if v in color:
                continue
            comp = [v]
            color[v] = 0
            stack = [v]
            while stack:
                x = stack.pop()
                for y in differ[x]:
                    if y not in nodes:
                        continue
                    if y not in color:
                        color[y] = 1 - color[x]
                        comp.append(y)
                        stack.append(y)
                    elif color[y] == color[x]:
                        conflict.add((x, y))
            comps.append(comp)

        def coverage(vs):
            arr = np.zeros(npos + 1, dtype=np.int32)
            for v in vs:
                for s, e in segs[v]:
                    arr[s] += 1
                    arr[e + 1] -= 1
            return np.cumsum(arr)

        press = {"A": np.zeros(npos + 1, dtype=np.int32), "B": np.zeros(npos + 1, dtype=np.int32)}
        comps.sort(key=lambda c: -sum(length(v) + 1 for v in c))
        file_of = {}
        for comp in comps:
            side = [[v for v in comp if color[v] == 0], [v for v in comp if color[v] == 1]]
            cov = [coverage(side[0]), coverage(side[1])]
            options = []
            for orient in (("A", "B"), ("B", "A")):
                bad = sum(1 for k in (0, 1) for v in side[k] if orient[k] in not_file[v])
                pa = press["A"] + (cov[0] if orient[0] == "A" else cov[1])
                pb = press["B"] + (cov[1] if orient[1] == "B" else cov[0])
                options.append((bad, max(pa.max(), pb.max()), int(pa.sum() + pb.sum()), orient))
            options.sort(key=lambda t: t[:3])
            orient = options[0][3]
            for k in (0, 1):
                for v in side[k]:
                    f = orient[k]
                    if f in not_file[v]:
                        other = "B" if f == "A" else "A"
                        if other not in not_file[v]:
                            f = other
                    file_of[v] = f
            for k in (0, 1):
                press[orient[k]] += cov[k]

        regs_owner = {"A": [[] for _ in range(32)], "B": [[] for _ in range(32)]}
        vregs_in = {"A": [[] for _ in range(32)], "B": [[] for _ in range(32)]}
        phys = {"A": isa.RA, "B": isa.RB}

        def evict_for(v, f):
            """Free a register of file f for v by moving the values overlapping it
            there into free registers of the other file (when they may live there:
            read-port conflicts this creates are fixed by the copy pass)."""
            g = "B" if f == "A" else "A"
            for r in range(32):
                if r in RESERVED:
                    continue
                blockers = [u for u in vregs_in[f][r] if _overlap(segs[u], segs[v])]
                if not blockers or any(g in not_file[u] for u in blockers):
                    continue
                taken = defaultdict(list)
                moves = []
                for u in blockers:
                    for r2 in range(32):
                        if r2 in RESERVED:
                            continue
                        if all(not _overlap(segs[u], x) for x in regs_owner[g][r2] + taken[r2]):
                            taken[r2].append(segs[u])
                            moves.append((u, r2))
                            break
                    else:
                        break
                if len(moves) != len(blockers):
                    continue
                for u, r2 in moves:
                    regs_owner[f][r].remove(segs[u])
                    vregs_in[f][r].remove(u)
                    regs_owner[g][r2].append(segs[u])
                    vregs_in[g][r2].append(u)
                    u.phys = phys[g][r2]
                regs_owner[f][r].append(segs[v])
                vregs_in[f][r].append(v)
                v.phys = phys[f][r]
                return True
            return False

        hint_map = defaultdict(list)
        for d, s in copy_pairs:
            hint_map[d].append(s)
            hint_map[s].append(d)
        order = sorted(rest, key=lambda v: segs[v][0][0])
        for v in order:
            f = file_of[v]
            placed = False
            for attempt in (f, "B" if f == "A" else "A"):
                if attempt in not_file[v] and attempt != f:
                    continue
                owners = regs_owner[attempt]
                prefs = []
                for p in hint_map[v]:
                    if p.phys is not None and p.phys.is_regfile and p.phys.rfile == attempt:
                        prefs.append(p.phys.raddr)
                for r in prefs + list(range(32)):
                    if r in RESERVED:
                        continue
                    if all(not _overlap(segs[v], s) for s in owners[r]):
                        owners[r].append(segs[v])
                        vregs_in[attempt][r].append(v)
                        v.phys = isa.RA[r] if attempt == "A" else isa.RB[r]
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                placed = any(evict_for(v, attempt) for attempt in (f, "B" if f == "A" else "A")
                             if attempt not in not_file[v])
            if not placed:
                live = int(max(press["A"].max(), press["B"].max()))
                raise RegAllocError(f"out of registers (peak pressure ~{live} per file); "
                                    f"reduce block sizes")
        # vregs that never got a segment (unused defs)
        for b in fn.blocks:
            for ins in b.insts:
                for op in ins.ops:
                    for x in [op.dst] + list(op.srcs):
                        if isinstance(x, VReg) and x.phys is None:
                            x.phys = isa.ACC[0] if x.cls in ("any", "acc") else x.phys

    # -- verification ------------------------------------------------------------

    def check(self):
        bad = []
        for b in self.fn.blocks:
            for ins in b.insts:
                if ins.kind != "alu" or not ins.ops:
                    continue
                problem = self._port_problem(ins)
                if problem is not None:
                    bad.append((b, ins, problem))
        return bad

    def _port_problem(self, ins: MInst):
        """Return an operand to copy into an accumulator, or None if encodable."""
        fa, fb, flex = _special_ports(ins)
        a_addrs = set()
        b_addrs = set()
        for op in ins.ops:
            for s in op.srcs:
                if isinstance(s, Reg) and s.acc is None and s.rfile == "A":
                    a_addrs.add(s.raddr)
                elif isinstance(s, Reg) and s.acc is None and s.rfile == "B":
                    b_addrs.add(s.raddr)
        b_taken_by_imm = fb and not b_addrs
        culprit = None
        for v in _vreg_operands(ins):
            p = v.phys
            if p is None or p.acc is not None:
                continue
            (a_addrs if p.rfile == "A" else b_addrs).add(p.raddr)
            culprit = v
            if len(a_addrs) > 1 or len(b_addrs) > 1 or (p.rfile == "B" and b_taken_by_imm):
                return v
        ports_used = (1 if a_addrs else 0) + (1 if (b_addrs or b_taken_by_imm) else 0)
        if len(flex) > 2 - ports_used:
            return culprit
        dsts = [op.dst.phys for op in ins.ops if isinstance(op.dst, VReg) and op.dst.phys is not None
                and op.dst.phys.is_regfile]
        if len(dsts) == 2 and dsts[0].rfile == dsts[1].rfile and dsts[0] != dsts[1]:
            return ("dst", ins.ops[1].dst)
        return None

    def insert_copy(self, blk, ins, operand):
        if isinstance(operand, tuple):
            # destination conflict inside an atomic pair: write via a temp
            _, v = operand
            tmp = VReg("cpy", cls="acc", is_float=v.is_float)
            for op in ins.ops:
                if op.dst is v:
                    op.dst = tmp
            idx = blk.insts.index(ins)
            blk.insts.insert(idx + 1, alu("mov", v, tmp))
            self.copies_inserted += 1
            return
        tmp = VReg("cpy", cls="acc", is_float=operand.is_float)
        for op in ins.ops:
            op.srcs = [tmp if s is operand else s for s in op.srcs]
        idx = blk.insts.index(ins)
        blk.insts.insert(idx, alu("mov", tmp, operand))
        self.copies_inserted += 1


def allocate(fn, verbose=False) -> Allocator:
    a = Allocator(fn, verbose)
    a.run()
    return a


def pressure_report(fn, top=40) -> str:
    """Human-readable list of values live at the point of peak pressure."""
    segs, pos = live_segments(fn)
    npos = max((s[-1][1] for s in segs.values()), default=0) + 2
    cnt = np.zeros(npos + 1, dtype=np.int32)
    for v, ss in segs.items():
        if v.cls in ("r4", "r5"):
            continue
        for s, e in ss:
            cnt[s] += 1
            cnt[e + 1] -= 1
    cnt = np.cumsum(cnt)
    p = int(cnt.argmax())
    live = [v for v, ss in segs.items() if v.cls not in ("r4", "r5") and any(s <= p <= e for s, e in ss)]
    groups = defaultdict(int)
    for v in live:
        groups[v.name] += 1
    where = None
    for b in fn.blocks:
        for ins in b.insts:
            if pos[id(ins)] <= p:
                where = (b.label, ins)
    lines = [f"peak {cnt[p]} live values at position {p} ({where[0]}: {where[1]!r})"]
    for name, n in sorted(groups.items(), key=lambda t: -t[1])[:top]:
        lines.append(f"  {n:3d} x {name}")
    return "\n".join(lines)
