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

Pairing (PITON_PAIR=1)
----------------------
The greedy packing fills the add ALU with the highest-priority op that
fits, so a register move that could run on either ALU often takes the add
ALU while an add-only op of the same cycle waits for the next instruction,
and in blocks with far more add-only than mul-only work (or the reverse)
the critical-path order runs out of ops for the busy ALU at the end. With
the switch:
  * a move on the add ALU steps over to the idle mul ALU (as `v8min x, x`,
    bit-exact) when an add-only op needs the add ALU (Bundle.try_add);
  * each block is scheduled with several priority functions (PAIR_MODES:
    the plain one, the move, and the move with a bonus of 1, 2 or 5 cycles
    for ops of the ALU with more single-unit ops left) and the shortest
    schedule is kept (the earlier one on ties); a loop that streams through
    the TMU keeps the one whose iterations wait least for their loads
    instead (tmu_loop_cost: the bonus can move a request closer to the load
    that receives it, and matmul's loop is bound by those waits);
  * a dot product's multiply-add pairs stay atomic, except that the add of
    the last product before each A broadcast (lower.py op_tt_dot, MInst
    `split`) may also be scheduled on its own: the broadcast's rotation
    leaves the add ALU idle, and that add is the one op ready for it; both
    forms are scheduled and the shorter kept (allocation still sees the
    pair, so the joint form always encodes);
  * the pairs those schedules missed for the read ports go to the
    port-aware allocation (regalloc.PortHints), each weighted by the share
    of schedules that missed it, so its next attempt does not depend on
    which schedule happened to be kept; a second port-aware search starts
    from the pairs only the kept schedules missed (compiler._build: either
    greedy search can end in the shorter loop);
  * a hazard that only the way into a loop has (a value written just
    before the loop and read at its head) is padded at the end of the
    preheader, not at the loop head where every iteration ran the nop
    (fix_block_boundaries).
Every schedule is built from the same dependence DAG and Bundle checks, so
all latency and hazard rules hold for each.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Optional

from ..qpu import isa
from ..qpu.asm import Program
from ..qpu.isa import AluOp, EncodeError, Imm, Instr, Reg
from .lower import TMU_DEPTH
from .mir import MInst, MOp, VReg

LAT_REGFILE = 2
LAT_ACC = 1
LAT_ROT = 2
LAT_SFU = 3
LAT_VPM_READ = 4
LAT_UNIF_ADDR = 3
# Soft latency used only for priorities: keep TMU requests far from their reads.
SOFT_TMU = 10
# TMU latencies (instructions) for tmu_loop_cost: the emulator's fixed-latency
# model reproduces the TMU stall of perf/results/pi-2026-11-03-baseline.json
# at about 60 (dense_relu, 25%), 70 (row_sum, 43%) and 90-100 (matmul, 26-35%)
TMU_LATENCIES = (60, 90, 120)
# Special destinations a register move may write from either ALU (PITON_PAIR;
# the mul ALU spells it `v8min x, x`), besides accumulators and register
# files. Not a discarded result or a move of an immediate, which vc4asm
# assembles to other bits on the mul ALU; VPM, DMA, uniform-address and mutex
# writes keep the ALU the plain packing gives them.
_EITHER_DSTS = frozenset([isa.R5QUAD, isa.R5REP, isa.TMU0_S, isa.TMU1_S] + list(isa.SFU_REGS))


def pair_enabled() -> bool:
    """PITON_PAIR=1: the pairing schedules (module docstring); unset or 0: off."""
    return os.environ.get("PITON_PAIR", "0") not in ("", "0")


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
        # a store slot's semaphore fences the VPM writes, DMAs and vw_wait
        # (lower.py slot_acquire / slot_release): nothing crosses it
        r.writes.update((("sem",), ("vpm",), ("mem",), ("mutex",)))
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


def _split_pairs(minsts) -> list:
    """One MInst per op of each atomic MInst marked `split` (PITON_PAIR)."""
    out = []
    for m in minsts:
        if m.split and m.atomic:
            out += [replace(m, ops=[op], atomic=False, split=False) for op in m.ops]
        else:
            out.append(m)
    return out


def build_dag(minsts, split: bool = False) -> list:
    """split: the ops of atomic MInsts marked `split` get a node each (PITON_PAIR)."""
    if split:
        minsts = _split_pairs(minsts)
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


def _either_move(n: Node) -> bool:
    """Is n a single register move that may run on either ALU (PITON_PAIR)?"""
    if n.minst.kind != "alu" or len(n.mops) != 1:
        return False
    op, aop = n.mops[0], n.alu_ops[0]
    if op.unit != "any" or op.op != "mov" or op.setf or aop.rot is not None:
        return False
    if any(isinstance(s, Imm) for s in aop.srcs()):
        return False
    d = aop.dst
    return d.waddr is not None and (d.waddr < 36 or d in _EITHER_DSTS)


class Bundle:
    def __init__(self, swap: bool = False):
        self.swap = swap            # PITON_PAIR: a move may leave the add ALU for the mul ALU
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
        self.rejected = []          # nodes refused only because the instruction did not encode

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
        if self.swap and self.mul is None and self.add is not None and len(n.mops) == 1 \
                and n.mops[0].unit == "add":
            # the move holding the add ALU steps over to the idle mul ALU
            holder = [x for x in self.nodes if x.alu_ops and x.alu_ops[0] is self.add]
            if holder and _either_move(holder[0]):
                self.add, self.mul = None, self.add
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
                self.add, self.mul = saved[:2]
                return False
            placement.append((unit, op, aop))
        dsts = set()
        for x in (self.add, self.mul):
            if x is not None and x.dst != isa.NOP:
                dsts.add((x.dst.waddr, x.dst.wfile if x.dst.waddr < 32 else ""))
        for unit, op, aop in placement:
            key = (aop.dst.waddr, aop.dst.wfile if aop.dst.waddr is not None and aop.dst.waddr < 32 else "")
            if aop.dst != isa.NOP and key in dsts:
                self.add, self.mul = saved[:2]
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
            if n not in self.rejected:
                self.rejected.append(n)
            return False
        self.nodes.append(n)
        self.periph += r.periph
        self.unif = self.unif or r.unif_read or r.unif_addr
        self.has_cond = self.has_cond or any(op.cond not in ("always", "never") for op in n.mops)
        if m.comment:
            self.comment.append(m.comment)
        return True


def _missed_pair(bundle: Bundle, ready: list):
    """Ready single-op nodes that only the encoding (read ports, write swap,
    ...) kept out of a bundle that still has the ALU they need free, each as
    the [(unit, MOp)] list the instruction would have held."""
    if bundle.kind != "alu" or bundle.empty or (bundle.add is None) == (bundle.mul is None):
        return []
    free = "add" if bundle.add is None else "mul"
    held = []
    for node in bundle.nodes:
        for mop, aop in zip(node.mops, node.alu_ops):
            held.append(("add" if aop is bundle.add else "mul", mop))
    out = []
    for n in bundle.rejected:
        if n in ready and len(n.mops) == 1 and n.mops[0].unit in (free, "any"):
            out.append(held + [(free, n.mops[0])])
    return out


def schedule_block(minsts, terminator: Optional[MInst] = None, missed: Optional[list] = None,
                   weight: float = 1, kept: Optional[list] = None, cost=None):
    """Returns list[Instr] for one block (terminator branch excluded).

    missed: if given, collects (weight, [(unit, MOp)]) for every instruction
    left with an idle ALU that a ready operation could have filled but for
    the encoding (register allocation reads these, regalloc.PortHints).
    With PITON_PAIR the shortest of the PAIR_MODES schedules (module
    docstring; the later ones only while none reaches _lower_bound), and
    `missed` gets the pairs of all those built, each at weight / their
    number; `kept`, if given, gets the pairs of the schedule returned, at
    full weight (compiler.py's second port-aware search). cost (PITON_PAIR):
    every schedule is built and the one of lowest (cost(schedule), length)
    kept, the earlier on ties (schedule_function: a loop's TMU waits)."""
    if not pair_enabled():
        return _list_schedule(build_dag(minsts), missed, weight)
    best, best_key, best_pairs, pairs, built = None, None, None, [], 0
    collect = missed is not None or kept is not None
    for split in (False, True) if any(m.split for m in minsts) else (False,):
        nodes = build_dag(minsts, split)
        bound = -1 if cost is not None else _lower_bound(nodes)
        for swap, balance in PAIR_MODES:
            mine = [] if collect else None
            out = _list_schedule(nodes, mine, weight, swap, balance)
            built += 1
            if collect:
                pairs += mine
            key = (cost(out), len(out)) if cost is not None else len(out)
            if best is None or key < best_key:
                best, best_key, best_pairs = out, key, mine
            if len(best) <= bound:
                break
        if len(best) <= bound:
            break
    if missed is not None:
        missed.extend((w / built, p) for w, p in pairs)
    if kept is not None:
        kept.extend(best_pairs)
    return best


# PITON_PAIR: (move to the mul ALU, balance bonus) of the schedules
# schedule_block compares, in order of preference
PAIR_MODES = ((False, 0), (True, 0), (True, 1), (True, 2), (True, 5))


def _lower_bound(nodes) -> int:
    """No schedule of the block is shorter: its critical path, and the ALU
    ops, peripheral accesses and load immediates each instruction can hold."""
    start = {}
    add = mul = either = periph = full = 0
    for n in nodes:
        start[n.idx] = max((start[p.idx] + lat for p, lat in n.preds), default=0)
        if n.minst.kind in ("ldi", "sem"):
            full += 1
            continue
        for op in n.mops:
            add += op.unit == "add"
            mul += op.unit == "mul"
            either += op.unit == "any"
        periph += n.res.periph
    alu = max(add, mul, -(-(add + mul + either) // 2), periph)
    return max(max(start.values(), default=-1) + 1, full + alu)


def _list_schedule(nodes, missed, weight, swap=False, balance=0):
    """Cycle-by-cycle list schedule of build_dag's nodes (schedule_block).
    swap: Bundle's move to the mul ALU (PITON_PAIR); balance: priority bonus
    of an op for the ALU with more single-unit ops left (PITON_PAIR)."""
    pair = swap or balance
    for n in nodes:
        n.cycle = None
    remaining = set(range(len(nodes)))
    earliest = {n.idx: 0 for n in nodes}
    npred = {n.idx: len(n.preds) for n in nodes}
    out = []
    cycle = 0
    ready = [n for n in nodes if not n.preds]
    left = {"add": 0, "mul": 0}             # single-unit ops not yet scheduled (balance)
    if balance:
        for n in nodes:
            if len(n.mops) == 1 and n.mops[0].unit in left:
                left[n.mops[0].unit] += 1

    def key(n):
        if balance and len(n.mops) == 1:
            u = n.mops[0].unit
            if u in left and left[u] > left["mul" if u == "add" else "add"]:
                return (-n.prio - balance, n.idx)
        return (-n.prio, n.idx)
    while remaining:
        bundle = Bundle(swap)
        refused = set()     # (PITON_PAIR) an op a bundle refused it refuses with more ops too
        progress = True
        while progress:
            progress = False
            cands = sorted((n for n in ready if earliest[n.idx] <= cycle and n.idx not in refused), key=key)
            for n in cands:
                if pair and (bundle.kind != "alu" or n.alu_ops and bundle.add is not None and bundle.mul is not None):
                    refused.add(n.idx)          # (no ALU left for it: what try_add would find)
                    continue
                if bundle.try_add(n):
                    n.cycle = cycle
                    ready.remove(n)
                    remaining.discard(n.idx)
                    if balance and len(n.mops) == 1 and n.mops[0].unit in left:
                        left[n.mops[0].unit] -= 1
                    for s, lat in n.succs:
                        earliest[s.idx] = max(earliest[s.idx], cycle + lat)
                        npred[s.idx] -= 1
                        if npred[s.idx] == 0:
                            ready.append(s)
                    progress = True
                    break
                if pair:
                    refused.add(n.idx)
        if bundle.empty:
            stuck = [n for n in ready if earliest[n.idx] <= cycle]
            if stuck:
                raise RuntimeError(f"cannot encode {stuck[0].minst!r} ({[str(a) for a in stuck[0].alu_ops]})")
        if missed is not None:
            pairs = _missed_pair(bundle, ready)
            missed.extend((weight / len(pairs), p) for p in pairs)
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
    # The resource model describes the immediate, non-link branches emitted by
    # schedule_function. Keep indirect targets and link writes outside it.
    if branch.breg is not None or any(op is not None and op.dst != isa.NOP
                                     for op in (branch.add, branch.mul)):
        return list(body) + [branch] + [Instr.nop() for _ in range(3)]
    k = 0
    while k < 3 and k < len(body):
        cand = body[len(body) - 1 - k]
        r = instr_res(cand)
        if r.setf or cand.kind in ("branch", "sem") or cand.sig == "thrend":
            break
        k += 1
    moved = body[len(body) - k:] if k else []
    head = body[:len(body) - k]
    slots = moved + [Instr.nop() for _ in range(3 - len(moved))]
    current = head + [branch] + slots
    if k == 3:
        return current

    # A final compare often prevents the trailing-only path from filling any
    # slots. Try small groups of independent arithmetic before that compare.
    # Moving a group together also handles a multiply whose add follows it.
    resources = [instr_res(ins) for ins in current]
    branch_pos = len(head)
    candidates = [i for i in range(max(0, branch_pos - 8), branch_pos)
                  if _pure_delay_alu(current[i]) and not _is_nop(current[i])]
    for count in range(3 - k, 0, -1):
        # At most 92 groups: choose up to three of the last eight instructions.
        for group in reversed(list(combinations(candidates, count))):
            selected = set(group)
            if any(not _commute(resources[i], resources[j])
                   for i in group for j in range(i + 1, branch_pos + 1) if j not in selected):
                continue
            proposed = [ins for i, ins in enumerate(current[:branch_pos + 1]) if i not in selected]
            proposed += [current[i] for i in group] + moved
            proposed += [Instr.nop() for _ in range(3 - k - count)]
            checked = _repair_delay_latencies(proposed)
            if checked is not None:
                return checked
    return current


def _pure_delay_alu(ins: Instr) -> bool:
    """Arithmetic that can move without touching flags or peripheral state."""
    if ins.kind != "alu" or ins.sig not in (None, "none") or ins.setf:
        return False
    r = instr_res(ins)
    return not (r.reads_flags or r.setf or r.periph) and \
        all(key[0] in ("A", "B", "acc", "r4", "r5") for key in r.reads | r.writes)


def _commute(a: Res, b: Res) -> bool:
    """Swapping the instructions preserves RAW, WAR and WAW dependencies."""
    return not (a.writes & (b.reads | b.writes) or a.reads & b.writes)


def _repair_delay_latencies(instructions: list) -> Optional[list]:
    """Validate the whole block, with at most six small producer hoists.

    Removing an instruction may close a required latency gap. In particular,
    moving the final add into a slot can leave a counter increment immediately
    before its compare. Hoist a pure producer across independent instructions
    to restore that gap; abandon the candidate if this cannot be proved safe.
    """
    result = list(instructions)
    for _ in range(6):
        resources = [instr_res(ins) for ins in result]
        violation = None
        for i, r in enumerate(resources):
            # Four cycles is the largest hard latency in the resource model.
            for j in range(i + 1, min(i + 5, len(resources))):
                distance = required_distance(r, resources[j])
                if distance > j - i:
                    violation = (i, distance - (j - i))
                    break
            if violation is not None:
                break
        if violation is None:
            return result
        i, shift = violation
        if not _pure_delay_alu(result[i]) or i < shift:
            return None
        if any(result[j].kind == "branch" or not _commute(resources[i], resources[j])
               for j in range(i - shift, i)):
            return None
        producer = result.pop(i)
        result.insert(i - shift, producer)
    return None


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

    def needed(tail, head):
        """Nops between tail and head for every latency across them."""
        need = 0
        for i, t in enumerate(tail):
            tr = instr_res(t)
            dist_tail = len(tail) - i      # distance to head[0]
            for j, h in enumerate(head[:4]):
                req = required_distance(tr, instr_res(h))
                if req > dist_tail + j:
                    need = max(need, req - (dist_tail + j))
        return need

    pair = pair_enabled()
    succs = {}
    for lbl, ps in preds.items():
        for p in ps:
            succs.setdefault(p, set()).add(lbl)
    changed = True
    while changed:
        changed = False
        for label in order:
            head = blocks_instrs[label]
            ps = preds.get(label, [])
            if pair and len(ps) > 1:
                # PITON_PAIR: a hazard only on the way in from blocks that fall
                # through into this one and nowhere else (a loop's preheader) is
                # padded at their end, so the other predecessors (the loop's back
                # edge) do not run the nops
                needs = {p: max(needed(t, head) for t in tails(p, 4)) for p in ps}
                late = [p for p in ps if needs[p]]
                if late and len(late) < len(ps) and all(
                        succs.get(p) == {label} and order.index(p) + 1 == order.index(label)
                        and not any(ins.kind == "branch" for ins in blocks_instrs[p][-4:]) for p in late):
                    for p in late:
                        blocks_instrs[p] = blocks_instrs[p] + [Instr.nop("boundary hazard")
                                                               for _ in range(needs[p])]
                    changed = True
                    continue
            for tail in [t for p in ps for t in tails(p, 4)]:
                need = needed(tail, head)
                if need:
                    blocks_instrs[label] = [Instr.nop("boundary hazard") for _ in range(need)] + head
                    head = blocks_instrs[label]
                    changed = True


def _tmu_events(ins: Instr):
    """(TMU unit requested, TMU unit received) by one instruction, None for neither."""
    req = None
    for op in (ins.add, ins.mul):
        if op is not None and op.dst in isa.TMU_S:
            req = isa.TMU_S.index(op.dst)
    recv = None
    if ins.kind == "alu" and ins.sig in ("ldtmu0", "ldtmu1"):
        recv = int(ins.sig[-1])
    return req, recv


def tmu_loop_period(body: list, latency: int, iters: int = 8) -> float:
    """Instructions per iteration, waits included, of a loop body that runs
    again and again with each TMU result ready `latency` instructions after its
    request (the emulator's TimingModel).  The FIFOs start as full as the body
    allows: a LoopStream keeps them full across the back edge."""
    events = [_tmu_events(ins) for ins in body]
    fifo = (deque(), deque())
    for u in (0, 1):
        level = peak = 0
        for req, recv in events:
            level += (req == u) - (recv == u)
            peak = max(peak, level)
        fifo[u].extend([0] * max(0, TMU_DEPTH - peak))
    t = 0
    ends = []
    for _ in range(iters):
        for req, recv in events:
            if recv is not None and fifo[recv]:
                t = max(t, fifo[recv].popleft())
            if req is not None:
                fifo[req].append(t + latency)
            t += 1
        ends.append(t)
    half = iters // 2
    return (ends[-1] - ends[half - 1]) / (iters - half)


def tmu_loop_cost(body: list) -> float:
    """tmu_loop_period averaged over TMU_LATENCIES."""
    return sum(tmu_loop_period(body, lat) for lat in TMU_LATENCIES) / len(TMU_LATENCIES)


def schedule_function(fn, name: str, missed: Optional[list] = None, kept: Optional[list] = None) -> Program:
    """Schedule an allocated mir.Function into a Program (missed, kept: see
    schedule_block; weighted by 100^loop depth, as compiler.py weighs blocks)."""
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
        if term is not None:
            br = Instr(kind="branch", bcond=term.bcond, target=term.target, rel=True,
                       add=AluOp("mov", isa.NOP, isa.NOP), mul=AluOp("mov", isa.NOP, isa.NOP))
        cost = None
        if pair_enabled() and term is not None and term.target == b.label \
                and any(op.dst in isa.TMU_S for m in insts for op in m.ops):
            # (PITON_PAIR) a loop that streams through the TMU keeps the schedule
            # that waits least for its loads (matmul_8x64x4: the shortest one
            # waits longer at every latency from 24 to 120)
            def cost(body, br=br):
                return tmu_loop_cost(fill_delay_slots(body, br))
        body = schedule_block(insts, missed=missed, weight=100 ** b.loop_depth, kept=kept, cost=cost)
        if term is not None:
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
