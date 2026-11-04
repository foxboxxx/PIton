"""Compiler driver: Triton kernel -> tile IR -> machine IR -> QPU program."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Optional

from . import frontend, ir, passes, reroll
from .backend import opt, presched
from .backend.lower import Lowering
from .backend.regalloc import PortHints, RegAllocError, allocate, clone_function
from .backend.sched import schedule_function
from .qpu.asm import Program

PORT_ROUNDS = 4         # port-aware allocation attempts per build (PITON_PORT_ALLOC)
PRESCHED_BIASES = [(3.0,), (3.0, 1.0, 0.5)]     # PITON_ISSUE: presched.py _cycle_order biases per build


@dataclass
class CompiledKernel:
    name: str
    program: Program
    uniforms: list                  # ABI: [(kind, name)], read in this order
    signature: dict
    constants: dict
    asm: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def words(self):
        return self.program.words()

    def __repr__(self):
        return (f"CompiledKernel({self.name}, {len(self.program.instrs)} instructions, "
                f"uniforms={[n for _, n in self.uniforms]})")


def _pyfn(kernel):
    if hasattr(kernel, "fn") and callable(kernel.fn):
        return kernel.fn
    if callable(kernel):
        return kernel
    raise TypeError("expected a @jit function")


def _weighted_length(prog: Program, mf) -> int:
    """Instruction count with each block weighted by 100^loop depth."""
    depth = {b.label: b.loop_depth for b in mf.blocks}
    starts = sorted((idx, label) for label, idx in prog.labels.items())
    bounds = [idx for idx, _ in starts] + [len(prog.instrs)]
    total, current = bounds[0], 0           # code before the first label: depth 0
    for (idx, label), end in zip(starts, bounds[1:]):
        current = depth.get(label, current)
        total += (end - idx) * 100 ** current
    return total


def _block_lengths(prog: Program) -> dict:
    starts = sorted((idx, label) for label, idx in prog.labels.items())
    bounds = [idx for idx, _ in starts] + [len(prog.instrs)]
    out = {None: bounds[0]}
    for (idx, label), end in zip(starts, bounds[1:]):
        out[label] = end - idx
    return out


def _likely_blocks(mf) -> set:
    """Labels on the path that predicts forward conditional branches not
    taken and backward ones taken: partial-store paths of ragged tails,
    skipped chunks and the slow division path are off it."""
    index = {b.label: i for i, b in enumerate(mf.blocks)}
    likely, stack = set(), [mf.blocks[0].label] if mf.blocks else []
    while stack:
        label = stack.pop()
        if label in likely:
            continue
        likely.add(label)
        i = index[label]
        last = mf.blocks[i].insts[-1] if mf.blocks[i].insts else None
        if last is not None and last.sig == "thrend":
            continue
        if last is not None and last.kind == "branch":
            if last.bcond == "always" or index[last.target] <= i:
                stack.append(last.target)
            if last.bcond == "always":
                continue
        if i + 1 < len(mf.blocks):
            stack.append(mf.blocks[i + 1].label)
    return likely


def _port_better(new: Program, new_mf, old: Program, old_mf) -> bool:
    """Is `new` shorter, weighting blocks by 100^loop depth and those off the
    likely path at 1/1000 (plain 100^depth cannot tell a block every program
    runs from a ragged-tail path: copy traded one instruction off each unused
    partial-store block for three on its main path)."""
    depth = {b.label: b.loop_depth for b in old_mf.blocks}
    likely = _likely_blocks(old_mf)

    def cost(prog):
        return sum(n * 100 ** depth.get(label, 0) * (1000 if label in likely or label is None else 1)
                   for label, n in _block_lengths(prog).items())
    return cost(new) < cost(old)


def _port_aware(original, name: str, missed: list, copy_of: dict, rounds: int = PORT_ROUNDS):
    """Port-aware allocations of `original` (regalloc.py, "Port-aware file
    choice"), starting from the pairs the scheduler could not form with the
    plain allocation (`missed`, `copy_of`): each round allocates a fresh
    clone, schedules it, and adds the pairs it still missed for the next.
    Yields (program, mir, allocator, self moves dropped); stops when an
    attempt fails to allocate or schedule (the plain allocation then stays)
    or the scheduler misses nothing new."""
    hints = PortHints()
    hints.add(missed, copy_of)
    for _ in range(rounds):
        mf = clone_function(original)
        try:
            alloc = allocate(mf, hints=hints)
            dropped = opt.drop_self_moves(mf)
            missed = []
            prog = schedule_function(mf, name, missed=missed)
        except (RegAllocError, RuntimeError):
            return
        yield prog, mf, alloc, dropped
        if not hints.add(missed, alloc.copy_of):
            return


def _build(tile, ranges, options: dict, interleave: bool, alphas=None):
    """Lower, allocate and schedule; returns (program, mir, lowering, stats, mir dumps)."""
    port_alloc = os.environ.get("PITON_PORT_ALLOC", "1") != "0"     # PITON_PORT_ALLOC=0: plain allocation only
    issue = options.get("issue", False)
    for cache_dot_a in (True, False):
        lw = Lowering(tile, ranges, dict(options, cache_dot_a=cache_dot_a))
        mf = lw.run()
        stats = dict(lw.stats)
        stats.update(opt.run(mf, issue=issue))
        stats["presched_blocks"] = presched.run(mf, flags=issue, alphas=alphas) if interleave else 0
        mir = mf.dump()
        original = clone_function(mf) if port_alloc else None
        try:
            alloc = allocate(mf)
        except RegAllocError:
            # Values outside the dot may already occupy rotation accumulators.
            # Rebuild fresh MIR without the optional cache before rejecting a
            # tile that the original allocation strategy could support.
            if not cache_dot_a or not stats.get("dot_a_cache_registers", 0):
                raise
            continue
        stats["dot_a_cache_fallback"] = int(not cache_dot_a)
        break
    stats["copies_inserted"] = alloc.copies_inserted
    stats["self_moves_dropped"] = opt.drop_self_moves(mf)
    mir_alloc = mf.dump()
    missed = [] if original is not None else None
    prog = schedule_function(mf, tile.name, missed=missed)
    if original is not None:
        # The plain allocation stays unless a port-aware one schedules shorter.
        stats["port_alloc"] = 0
        for attempt, (p, f, a, dropped) in enumerate(_port_aware(original, tile.name, missed, alloc.copy_of), 1):
            if _port_better(p, f, prog, mf):
                prog, mf, mir_alloc = p, f, f.dump()
                stats.update(port_alloc=attempt, copies_inserted=a.copies_inserted, self_moves_dropped=dropped)
    if issue:
        # the chosen allocation again, each block also with either-ALU ops on the
        # mul ALU (sched.py prefer_mul; never longer)
        prog = schedule_function(mf, tile.name, prefer_mul=True)
    return prog, mf, lw, stats, mir, mir_alloc


def _compile_tile(fn, signature, constants, name, pid_max, options, chunk: int = 0):
    """Tile IR (rerolled into `chunk`-element pieces if nonzero) through
    scheduling; picks the better of the plain and pre-scheduled builds."""
    tile = frontend.generate(fn, signature, constants, name=name)
    asm = {"ttir": tile.dump()}
    if options.get("issue"):
        passes.drop_dead_other(tile)          # (PITON_ISSUE: masked lanes no store reads)
    if chunk:
        if options.get("store_slots"):
            reroll.run(tile, chunk, queue=True)     # (the block's store after the loop)
        else:
            reroll.run(tile, chunk)
        asm["ttir_rerolled"] = tile.dump()
    ranges = passes.optimize(tile, pid_max)
    asm["ttir_opt"] = tile.dump()
    # Interleaving independent chains before allocation (presched.py) usually
    # shortens the schedule but needs more registers; keep whichever version
    # allocates and runs shorter, weighting loop bodies by their depth.
    try:
        prog, mf, lw, stats, asm["mir"], asm["mir_alloc"] = _build(tile, ranges, options, interleave=False)
    except RegAllocError:
        if not options.get("store_slots"):
            raise
        # the store slot's registers (or a rerolled loop's store queue) did not
        # fit: the private VPM rows need fewer
        return _compile_tile(fn, signature, constants, name, pid_max, dict(options, store_slots=False), chunk)
    stats["presched"] = 0
    if os.environ.get("PITON_PRESCHED", "1") != "0":       # PITON_PRESCHED=0: A/B comparisons
        # PITON_ISSUE: flag groups ordered by the cycle model (presched.py), close to
        # program order and also the best of three biases per block
        sets = PRESCHED_BIASES if options.get("issue") else [None]
        variants = [(alphas, options) for alphas in sets]
        if stats.get("loop_lanes"):
            # and with lane registers per column block in loops (lower.py "Issue")
            variants += [(alphas, dict(options, loop_lanes=True)) for alphas in sets]
        for k, (alphas, opts) in enumerate(variants, 1):
            try:
                candidate = _build(tile, ranges, opts, interleave=True, alphas=alphas)
            except RegAllocError:
                continue
            if _weighted_length(candidate[0], candidate[1]) < _weighted_length(prog, mf):
                prog, mf, lw, stats, asm["mir"], asm["mir_alloc"] = candidate
                stats["presched"] = k
    stats["rerolled"] = chunk
    return tile, prog, mf, lw, stats, asm


def compile(kernel, signature: dict, constants: Optional[dict] = None, num_stages: int = 2,
            name: Optional[str] = None, verify: bool = True, store_mutex: bool = True,
            grid=None) -> CompiledKernel:
    """Compile a Triton kernel for the VideoCore IV QPU.

    signature  -- {arg: "*fp32" | "*i32" | "i32" | "fp32"} for runtime arguments
    constants  -- {arg: value} for tl.constexpr arguments and specialised scalars
    num_stages -- 1 disables software pipelining of loop loads
    store_mutex -- serialise store sequences across QPUs with the hardware mutex
    grid        -- largest launch grid the kernel will run with (enables range-based
                   optimisations of program-id arithmetic); None: any grid
    """
    fn = _pyfn(kernel)
    constants = dict(constants or {})
    pid_max = None
    if grid is not None:
        g = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        pid_max = tuple(max(1, int(x)) - 1 for x in g) + (0,) * (3 - len(g))
    options = {"num_stages": num_stages, "store_mutex": store_mutex,
               # stores through shared 16-row VPM slots (backend/lower.py); PITON_STORE_SLOTS=0: private rows
               "store_slots": os.environ.get("PITON_STORE_SLOTS", "1") not in ("", "0"),
               # issue efficiency (presched.py flag groups, lower.py "Issue"); PITON_ISSUE=1: on
               "issue": os.environ.get("PITON_ISSUE", "0") not in ("", "0")}
    use_reroll = os.environ.get("PITON_REROLL", "1") != "0"
    sub = reroll.SUB
    try:
        tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options)
    except RegAllocError:
        # A large block does not fit the registers as straight-line code;
        # as a loop over chunks only one chunk is live at a time.
        if not (use_reroll and reroll.applicable(frontend.generate(fn, signature, constants, name=name), sub)):
            raise
        tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options, sub)
        sub //= 2
    # A straight-line program larger than the instruction cache stalls on
    # every cache line (reroll.py); a loop over 64-element chunks fits, and
    # heavier bodies (exact division) get smaller chunks until theirs does.
    while (len(prog.instrs) > reroll.ICACHE_INSTRUCTIONS and sub >= 16 and use_reroll
           and reroll.applicable(frontend.generate(fn, signature, constants, name=name), sub)):
        rerolled = _compile_tile(fn, signature, constants, name, pid_max, options, sub)
        if len(rerolled[1].instrs) >= len(prog.instrs):
            break
        tile, prog, mf, lw, stats, asm = rerolled
        sub //= 2
    prog.header = [f"{tile.name}: compiled by PIton (Triton -> VideoCore IV QPU)",
                   "uniforms: " + ", ".join(n for _, n in lw.uniforms),
                   "constants: " + ", ".join(f"{k}={v}" for k, v in constants.items())]
    asm["qasm"] = prog.qasm()
    stats["instructions"] = len(prog.instrs)
    ck = CompiledKernel(tile.name, prog, list(lw.uniforms), dict(signature), constants, asm, stats)
    if verify:
        prog.words()     # every instruction must encode
    return ck
