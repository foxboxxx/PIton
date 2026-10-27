"""Instruction scheduling and bundling.

Post-register-allocation list scheduler.  Each basic block's MInsts form a
dependence DAG whose edge latencies encode the QPU pipeline rules
(reference guide p37, plus what tools/probes.py measured on hardware):

  regfile RAW            2   (no forwarding from the register files)
  accumulator RAW        1   (r5 too: probe 'broadcast')
  rotation of an acc     2   (guide: "must not immediately follow a write")
  rotation by r5         2
  ldtmu -> r4 read       1
  SFU write -> r4 read   3   (and no other r4 writer in between)
  vr_setup -> vpm read   4
  unif_addr -> unif      3
  WAR                    0   (an instruction reads before it writes)
  WAW / ordering chains  1

Ready operations are packed greedily into instructions: one add-unit op,
one mul-unit op, one signal, subject to read ports / write swap (checked by
actually encoding the candidate instruction) and the one-peripheral-access-
per-instruction rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..qpu import isa
from ..qpu.asm import Program
from ..qpu.isa import AluOp, EncodeError, Imm, Instr, Reg
from .mir import MInst, MOp, VReg

LAT_REGFILE = 2
LAT_ACC = 1
LAT_ROT = 2
LAT_SFU = 3
LAT_VPM_READ = 4
LAT_UNIF_ADDR = 3
# Soft latency used only for priorities: keep TMU requests far from their reads.
SOFT_TMU = 10


def _phys(x):
    if isinstance(x, VReg):
        if x.phys is None:
            raise RuntimeError(f"unallocated {x}")
        return x.phys
    return x


def to_alu(op: MOp) -> AluOp:
    dst = op.dst
    if isinstance(dst, VReg) and dst.cls == "r5":
        dst = isa.R5QUAD if dst.hint == "quad" else isa.R5REP
    else:
        dst = _phys(dst)
    srcs = [_phys(s) for s in op.srcs]
    a = srcs[0] if srcs else isa.NOP
    b = srcs[1] if len(srcs) > 1 else a
    return AluOp(op.op, dst, a, b, cond=op.cond, rot=op.rot)


# ---------------------------------------------------------------------------
# Resource model (shared by the DAG builder and the cross-block checker)
# ---------------------------------------------------------------------------


def _reg_key(r: Reg):
    if r.acc is not None and r.acc < 4:
        return ("acc", r.acc)
    if r.acc == 4:
        return ("r4",)
    if r.acc == 5:
        return ("r5",)
    if r.is_regfile:
        return (r.rfile, r.raddr)
    return None


def _write_key(r: Reg):
    if r.waddr is None or r == isa.NOP or r.waddr == 39:
        return []
    if r.waddr < 32:
        return [(r.wfile, r.waddr)]
    if 32 <= r.waddr <= 35:
        return [("acc", r.waddr - 32)]
    if r.waddr == 37:
        return [("r5",)]
    if r.waddr in (56, 57, 58, 59):
        return [("tmu0",), ("mem",), ("periph",)]
    if r.waddr in (60, 61, 62, 63):
        return [("tmu1",), ("mem",), ("periph",)]
    if r.waddr in (52, 53, 54, 55):
        return [("r4",), ("sfu",), ("periph",)]
    if r.waddr == 48:
        return [("vpm",)]
    if r.waddr == 49:
        return [("vpm",), ("vrsetup",) if r.wfile == "A" else ("vpm",)]
    if r.waddr == 50:
        return [("vpm",), ("mem",)]
    if r.waddr == 40:
        return [("unif",), ("unifaddr",)]
    if r.waddr == 51:
        return [("mutex",), ("vpm",), ("mem",)]
    if r.waddr == 38:
        return [("host",)]
    if r.waddr == 36:
        return [("tmu0",), ("tmu1",)]
    return [("io", r.waddr)]


def _read_keys(r: Reg):
    k = _reg_key(r)
    if k is not None:
        return [k]
    if r.rfile is None:
        return []
    if r.raddr == 32:
        return [("unif",)]
    if r.raddr == 48:
        return [("vpm",), ("vpmread",)]
    if r.raddr in (49, 50):
        return [("vpm",), ("mem",)]
    if r.raddr == 51:
        return [("mutex",), ("vpm",), ("mem",), ("periph",)]
    return []


@dataclass
class Res:
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    rot_srcs: set = field(default_factory=set)   # keys read through a rotation
    rot_r5: bool = False
    sfu: bool = False                            # writes r4 through the SFU
    tmu_req: bool = False
    tmu_recv: bool = False
    vr_setup: bool = False
    vpm_read: bool = False
    unif_addr: bool = False
    unif_read: bool = False
    setf: bool = False
    reads_flags: bool = False
    periph: int = 0


def resources_of(alu_ops, sig=None, kind="alu", setf=False, cond_any=False) -> Res:
    r = Res()
    for op in alu_ops:
        for s in op.srcs() if op.op != "nop" else []:
            if isinstance(s, Imm):
                continue
            for k in _read_keys(s):
                if k == ("periph",):
                    r.periph += 1
                    continue
                r.reads.add(k)
                if k in (("unif",), ("vpm",), ("mem",), ("mutex",), ("vpmread",)):
                    # reading these advances hardware state: keep them ordered
                    r.writes.add(k)
                if k == ("unif",):
                    r.unif_read = True
                if k == ("vpmread",):
                    r.vpm_read = True
            if op.rot is not None and _reg_key(s) is not None:
                r.rot_srcs.add(_reg_key(s))
        if op.rot == "r5":
            r.rot_r5 = True
            r.reads.add(("r5",))
        if op.cond not in ("always", "never"):
            r.reads_flags = True
            k = _reg_key(op.dst) if op.dst.waddr is not None and op.dst.waddr < 36 else None
            if op.dst.waddr is not None and op.dst.waddr < 32:
                r.reads.add((op.dst.wfile, op.dst.waddr))
            elif k is not None:
                r.reads.add(k)
        for k in _write_key(op.dst):
            if k == ("periph",):
                r.periph += 1
                continue
            r.writes.add(k)
        if op.dst in isa.SFU_REGS:
            r.sfu = True
        if op.dst in isa.TMU_S:
            r.tmu_req = True
        if op.dst is isa.VR_SETUP:
            r.vr_setup = True
        if op.dst is isa.UNIF_ADDR:
            r.unif_addr = True
    if sig in ("ldtmu0", "ldtmu1"):
        unit = ("tmu0",) if sig == "ldtmu0" else ("tmu1",)
        r.writes.add(unit)
        r.writes.add(("r4",))
        r.tmu_recv = True
        r.periph += 1
    if kind == "sem":
        r.writes.add(("sem",))
        r.periph += 1
    if setf:
        r.setf = True
        r.writes.add(("flags",))
    if r.reads_flags or cond_any:
        r.reads.add(("flags",))
    return r


def raw_latency(prod: Res, cons: Res, key) -> int:
    if key[0] in ("A", "B"):
        return LAT_REGFILE
    if key[0] == "acc":
        return LAT_ROT if key in cons.rot_srcs else LAT_ACC
    if key == ("r5",):
        return LAT_ROT if cons.rot_r5 else LAT_ACC
    if key == ("r4",):
        return LAT_SFU if prod.sfu else 1
    if key == ("vpm",) and prod.vr_setup and cons.vpm_read:
        return LAT_VPM_READ
    if key == ("unif",) and prod.unif_addr and cons.unif_read:
        return LAT_UNIF_ADDR
    return 1


def waw_latency(prev: Res, nxt: Res, key) -> int:
    if key == ("r4",) and prev.sfu:
        return LAT_SFU
    if key == ("unif",) and prev.unif_read and nxt.unif_addr:
        return LAT_UNIF_ADDR
    if key == ("vpm",) and prev.vr_setup and nxt.vpm_read:
        return LAT_VPM_READ
    if key == ("unif",) and prev.unif_addr and nxt.unif_read:
        return LAT_UNIF_ADDR
    return 1


def required_distance(prev: Res, nxt: Res) -> int:
    """Minimum instruction distance between two instructions (for checks)."""
    d = 0
    for k in prev.writes & nxt.reads:
        d = max(d, raw_latency(prev, nxt, k))
    for k in prev.writes & nxt.writes:
        if k in (("r4",),) or k[0] in ("vpm", "unif"):
            d = max(d, waw_latency(prev, nxt, k))
    if prev.sfu and (("r4",) in nxt.reads or ("r4",) in nxt.writes):
        d = max(d, LAT_SFU)
    return d


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Node:
    idx: int
    minst: MInst
    alu_ops: list
    res: Res
    mops: list = field(default_factory=list)      # MOps matching alu_ops
    preds: list = field(default_factory=list)     # (node, latency)
    succs: list = field(default_factory=list)
    prio: int = 0
    cycle: Optional[int] = None
    soft_succs: list = field(default_factory=list)


def _node_for(i, m: MInst) -> Node:
    if m.kind == "ldi":
        ops = [AluOp("mov", to_alu(op).dst, isa.NOP, cond=op.cond) for op in m.ops]
        res = resources_of(ops, None, "ldi", setf=any(o.setf for o in m.ops))
    elif m.kind == "sem":
        ops = []
        res = resources_of([], None, "sem")
    else:
        mops = [op for op in m.ops if op.op != "r4def"]
        ops = [to_alu(op) for op in mops]
        res = resources_of(ops, m.sig, "alu", setf=any(o.setf for o in mops))
        return Node(i, m, ops, res, mops)
    return Node(i, m, ops, res, list(m.ops))


def build_dag(minsts) -> list:
    nodes = [_node_for(i, m) for i, m in enumerate(minsts)]
    last_w = {}
    readers = {}
    last_tmu_req = {}
    for n in nodes:
        r = n.res
        for k in r.reads:
            w = last_w.get(k)
            if w is not None:
                _edge(w, n, raw_latency(w.res, r, k))
            readers.setdefault(k, []).append(n)
        for k in r.writes:
            for rd in readers.get(k, []):
                if rd is not n:
                    lat = 0
                    if k == ("flags",):
                        lat = 1
                    _edge(rd, n, lat)
            w = last_w.get(k)
            if w is not None and w is not n:
                _edge(w, n, waw_latency(w.res, r, k))
            last_w[k] = n
            readers[k] = []
        # TMU soft latency for priorities
        for unit in (("tmu0",), ("tmu1",)):
            if unit in r.writes and r.tmu_req:
                last_tmu_req.setdefault(unit, []).append(n)
            if unit in r.writes and r.tmu_recv and last_tmu_req.get(unit):
                req = last_tmu_req[unit].pop(0)
                req.soft_succs.append(n)
    # priorities: longest path to the end (soft TMU latencies included)
    for n in reversed(nodes):
        p = 1
        for s, lat in n.succs:
            p = max(p, s.prio + max(lat, 1))
        for s in n.soft_succs:
            p = max(p, s.prio + SOFT_TMU)
        n.prio = p
    return nodes


def _edge(a: Node, b: Node, lat: int):
    a.succs.append((b, lat))
    b.preds.append((a, lat))


# ---------------------------------------------------------------------------
# Bundling
# ---------------------------------------------------------------------------


class Bundle:
    def __init__(self):
        self.kind = "alu"
        self.add = None
        self.mul = None
        self.sig = "none"
        self.setf = False
        self.setf_unit = None
        self.imm = 0
        self.ldi_mode = 0
        self.acquire = False
        self.sema = 0
        self.nodes = []
        self.periph = 0
        self.unif = False
        self.has_cond = False
        self.comment = []

    @property
    def empty(self):
        return not self.nodes

    def to_instr(self) -> Instr:
        if self.kind == "ldi":
            return Instr(kind="ldi", add=self.add, mul=self.mul, imm=self.imm, ldi_mode=self.ldi_mode,
                         setf=self.setf, comment="; ".join(c for c in self.comment if c))
        if self.kind == "sem":
            return Instr(kind="sem", acquire=self.acquire, sema=self.sema)
        return Instr(kind="alu", add=self.add, mul=self.mul, sig=self.sig, setf=self.setf,
                     comment="; ".join(c for c in self.comment if c))

    def try_add(self, n: Node) -> bool:
        m = n.minst
        if self.kind != "alu":
            return False
        if m.kind in ("ldi", "sem"):
            if not self.empty:
                return False
            self.kind = m.kind
            if m.kind == "ldi":
                ops = n.alu_ops
                self.add = ops[0] if ops else None
                self.mul = ops[1] if len(ops) > 1 else None
                self.imm = m.imm
                self.ldi_mode = m.ldi_mode
                self.setf = any(o.setf for o in m.ops)
            else:
                self.acquire, self.sema = m.acquire, m.sema
            self.nodes.append(n)
            self.comment.append(m.comment)
            return True
        r = n.res
        if self.periph + r.periph > 1:
            return False
        if r.unif_read and self.unif:
            return False
        # keep uniform-stream traffic out of instructions that touch the TMU/SFU
        if (r.unif_read or r.unif_addr) and self.periph:
            return False
        if r.periph and self.unif:
            return False
        if m.sig and self.sig != "none":
            return False
        mop_setf = [op.setf for op in n.mops]
        if any(mop_setf) and (self.setf or self.has_cond):
            return False
        if self.setf and any(op.cond not in ("always", "never") for op in n.mops):
            return False
        # place ops on units
        saved = (self.add, self.mul, self.sig, self.setf, self.setf_unit)
        placement = []
        free_add, free_mul = self.add is None, self.mul is None
        for op, aop in zip(n.mops, n.alu_ops):
            unit = op.unit
            if unit == "any":
                unit = "add" if free_add else ("mul" if free_mul else None)
            if unit == "add" and free_add:
                free_add = False
            elif unit == "mul" and free_mul:
                free_mul = False
            else:
                return False
            placement.append((unit, op, aop))
        dsts = set()
        for x in (self.add, self.mul):
            if x is not None and x.dst != isa.NOP:
                dsts.add((x.dst.waddr, x.dst.wfile if x.dst.waddr < 32 else ""))
        for unit, op, aop in placement:
            key = (aop.dst.waddr, aop.dst.wfile if aop.dst.waddr is not None and aop.dst.waddr < 32 else "")
            if aop.dst != isa.NOP and key in dsts:
                return False
            if aop.dst != isa.NOP:
                dsts.add(key)
            if unit == "add":
                self.add = aop
            else:
                self.mul = aop
            if op.setf:
                self.setf = True
                self.setf_unit = unit
        if m.sig:
            self.sig = m.sig
        # flags come from the add ALU unless it is a nop
        if self.setf and self.setf_unit == "mul" and self.add is not None:
            self.add, self.mul, self.sig, self.setf, self.setf_unit = saved
            return False
        try:
            isa.encode(isa.canonicalize(self.to_instr()))
        except EncodeError:
            self.add, self.mul, self.sig, self.setf, self.setf_unit = saved
            return False
        self.nodes.append(n)
        self.periph += r.periph
        self.unif = self.unif or r.unif_read or r.unif_addr
        self.has_cond = self.has_cond or any(op.cond not in ("always", "never") for op in n.mops)
        if m.comment:
            self.comment.append(m.comment)
        return True


def schedule_block(minsts, terminator: Optional[MInst] = None):
    """Returns list[Instr] for one block (terminator branch excluded)."""
    nodes = build_dag(minsts)
    remaining = set(range(len(nodes)))
    earliest = {n.idx: 0 for n in nodes}
    npred = {n.idx: len(n.preds) for n in nodes}
    out = []
    cycle = 0
    ready = [n for n in nodes if not n.preds]
    while remaining:
        bundle = Bundle()
        progress = True
        while progress:
            progress = False
            cands = sorted((n for n in ready if earliest[n.idx] <= cycle),
                           key=lambda n: (-n.prio, n.idx))
            for n in cands:
                if bundle.try_add(n):
                    n.cycle = cycle
                    ready.remove(n)
                    remaining.discard(n.idx)
                    for s, lat in n.succs:
                        earliest[s.idx] = max(earliest[s.idx], cycle + lat)
                        npred[s.idx] -= 1
                        if npred[s.idx] == 0:
                            ready.append(s)
                    progress = True
                    break
        if bundle.empty:
            stuck = [n for n in ready if earliest[n.idx] <= cycle]
            if stuck:
                raise RuntimeError(f"cannot encode {stuck[0].minst!r} ({[str(a) for a in stuck[0].alu_ops]})")
        out.append(bundle.to_instr() if not bundle.empty else Instr.nop())
        cycle += 1
        if len(out) > 100000:
            raise RuntimeError("scheduler did not converge")
    # drop trailing pure nops produced by latency waits at the very end
    while out and _is_nop(out[-1]):
        out.pop()
    return out


def _is_nop(ins: Instr) -> bool:
    return ins.kind == "alu" and ins.add is None and ins.mul is None and ins.sig == "none"


def instr_res(ins: Instr) -> Res:
    if ins.kind == "branch":
        r = Res()
        if ins.bcond != "always":
            r.reads.add(("flags",))
        return r
    if ins.kind == "sem":
        return resources_of([], None, "sem")
    ops = [o for o in (ins.add, ins.mul) if o is not None]
    if ins.kind == "ldi":
        ops = [AluOp("mov", o.dst, isa.NOP, cond=o.cond) for o in ops]
    return resources_of(ops, ins.sig if ins.kind == "alu" else None, ins.kind, setf=ins.setf)


def fill_delay_slots(body: list, branch: Instr) -> list:
    """Move up to three trailing instructions into the branch delay slots."""
    k = 0
    while k < 3 and k < len(body):
        cand = body[len(body) - 1 - k]
        r = instr_res(cand)
        if r.setf or cand.kind == "branch" or cand.sig == "thrend":
            break
        k += 1
    moved = body[len(body) - k:] if k else []
    head = body[:len(body) - k]
    slots = moved + [Instr.nop() for _ in range(3 - len(moved))]
    return head + [branch] + slots


def fix_block_boundaries(blocks_instrs: list, preds: dict, order: list) -> None:
    """Insert nops at block heads so no hazard crosses a block boundary.

    blocks_instrs: label -> list[Instr] (including branches and delay slots)
    """
    def tails(lbl, k, depth=0):
        """The possible last-k instruction sequences executed up to the end of
        block lbl: short (or empty) blocks continue into their predecessors."""
        ins = blocks_instrs[lbl]
        ps = preds.get(lbl, [])
        if len(ins) >= k or not ps or depth > 8:
            return [ins[-k:]]
        out = []
        for p in ps:
            for t in tails(p, k - len(ins), depth + 1):
                out.append(t + ins)
        return out

    changed = True
    while changed:
        changed = False
        for label in order:
            head = blocks_instrs[label]
            for tail in [t for p in preds.get(label, []) for t in tails(p, 4)]:
                need = 0
                for i, t in enumerate(tail):
                    tr = instr_res(t)
                    dist_tail = len(tail) - i      # distance to head[0]
                    for j, h in enumerate(head[:4]):
                        req = required_distance(tr, instr_res(h))
                        if req > dist_tail + j:
                            need = max(need, req - (dist_tail + j))
                if need:
                    blocks_instrs[label] = [Instr.nop("boundary hazard") for _ in range(need)] + head
                    head = blocks_instrs[label]
                    changed = True


def schedule_function(fn, name: str) -> Program:
    """Schedule an allocated mir.Function into a Program."""
    fn.compute_cfg()
    preds = fn.preds()
    per_block = {}
    order = [b.label for b in fn.blocks]
    for b in fn.blocks:
        insts = list(b.insts)
        for ins in insts[:-1]:
            if ins.kind == "branch":
                raise RuntimeError(f"branch in the middle of block {b.label}")
        term = None
        if insts and insts[-1].kind == "branch":
            term = insts.pop()
        ends = insts and insts[-1].sig == "thrend"
        end_inst = insts.pop() if ends else None
        body = schedule_block(insts)
        if term is not None:
            br = Instr(kind="branch", bcond=term.bcond, target=term.target, rel=True,
                       add=AluOp("mov", isa.NOP, isa.NOP), mul=AluOp("mov", isa.NOP, isa.NOP))
            body = fill_delay_slots(body, br)
        if end_inst is not None:
            body += [Instr(kind="alu", sig="thrend", comment="end of program"), Instr.nop(), Instr.nop()]
        per_block[b.label] = body
    fix_block_boundaries(per_block, preds, order)
    prog = Program(name)
    for label in order:
        prog.labels[label] = len(prog.instrs)
        prog.instrs.extend(per_block[label])
    return prog
