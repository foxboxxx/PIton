"""Compiler driver: Triton kernel -> tile IR -> machine IR -> QPU program."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Optional

from . import frontend, ir, passes, reroll
from .backend import opt, presched
from .backend.lower import Lowering, fast_idiv_enabled
from .backend.regalloc import PortHints, RegAllocError, allocate, clone_function
from .backend.sched import schedule_function
from .qpu.asm import Program

PORT_ROUNDS = 4         # port-aware allocation attempts per build (PITON_PORT_ALLOC)


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


def _misses_icache(tile, prog: Program, mf, stats: dict) -> bool:
    """Does `prog` stream through the instruction cache (reroll.py)?  Any
    program over ICACHE_INSTRUCTIONS, except (PITON_FAST_IDIV) a rerolled one
    with branch-free divisions whose stores are queued (reroll.py `queue`):
    the stores run once per program, after the loop, so only the loop has to
    stay cached, and a smaller chunk shrinks only the loop or loses the queue
    (idiv: 692 instructions with a 268-instruction loop and two 1 KiB DMAs;
    in 32-element chunks 16 DMAs of 128 bytes).  Kept to the switch, so that
    builds without it are unchanged (idiv's chunks with the branching division
    do not fit the queue's registers anyway)."""
    if len(prog.instrs) <= reroll.ICACHE_INSTRUCTIONS:
        return False
    fast = any(stats.get(k) for k in ("idiv_one_step", "idiv_short", "idiv_full"))
    if not (fast and stats.get("rerolled") and any(o.name == "piton.shift_in" for o in tile.walk())):
        return True
    depth = {b.label: b.loop_depth for b in mf.blocks}
    return sum(n for label, n in _block_lengths(prog).items() if depth.get(label, 0)) > reroll.ICACHE_INSTRUCTIONS


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


def _build(tile, ranges, options: dict, interleave: bool):
    """Lower, allocate and schedule; returns (program, mir, lowering, stats, mir dumps)."""
    port_alloc = os.environ.get("PITON_PORT_ALLOC", "1") != "0"     # PITON_PORT_ALLOC=0: plain allocation only
    for cache_dot_a in (True, False):
        lw = Lowering(tile, ranges, dict(options, cache_dot_a=cache_dot_a))
        mf = lw.run()
        stats = dict(lw.stats)
        stats.update(opt.run(mf))
        stats["presched_blocks"] = presched.run(mf) if interleave else 0
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
    return prog, mf, lw, stats, mir, mir_alloc


def _compile_tile(fn, signature, constants, name, pid_max, options, chunk: int = 0,
                  reductions: bool = False, carry: bool = True, transposed: bool = False):
    """Tile IR (rerolled into `chunk`-element pieces if nonzero; `reductions`,
    `carry`: reroll.run; `transposed`: reroll.transposed_loads) through
    scheduling; picks the better of the plain and pre-scheduled builds."""
    tile = frontend.generate(fn, signature, constants, name=name)
    asm = {"ttir": tile.dump()}
    if transposed:
        reroll.transposed_loads(tile)
    if chunk:
        # with store slots, a block's store of one slot or less waits after the loop
        reroll.run(tile, chunk, reductions, carry, queue=bool(options.get("store_slots")))
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
        return _compile_tile(fn, signature, constants, name, pid_max, dict(options, store_slots=False), chunk,
                             reductions, carry, transposed)
    stats["presched"] = 0
    candidate = None
    if os.environ.get("PITON_PRESCHED", "1") != "0":       # PITON_PRESCHED=0: A/B comparisons
        try:
            candidate = _build(tile, ranges, options, interleave=True)
        except RegAllocError:
            pass
    if candidate is not None and _weighted_length(candidate[0], candidate[1]) < _weighted_length(prog, mf):
        prog, mf, lw, stats, asm["mir"], asm["mir_alloc"] = candidate
        stats["presched"] = 1
    stats["rerolled"] = chunk
    if chunk and reductions:
        stats["reroll_carry"] = int(carry)
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
               "store_slots": os.environ.get("PITON_STORE_SLOTS", "1") not in ("", "0")}
    try:
        tile, prog, lw, stats, asm = _compile_program(fn, signature, constants, name, pid_max, options)
    except RegAllocError:
        # PITON_FAST_IDIV: the straight-line division keeps a whole tile's operands,
        # both results and the block's cached lane offsets and constants live at
        # once; a tile that only fits the registers with the branching division
        # (e.g. offs // N and offs % N of a 16x16 tile) gets that one
        if not fast_idiv_enabled():
            raise
        tile, prog, lw, stats, asm = _compile_program(fn, signature, constants, name, pid_max,
                                                      dict(options, fast_idiv=False))
        stats["fast_idiv_fallback"] = 1
    prog.header = [f"{tile.name}: compiled by PIton (Triton -> VideoCore IV QPU)",
                   "uniforms: " + ", ".join(n for _, n in lw.uniforms),
                   "constants: " + ", ".join(f"{k}={v}" for k, v in constants.items())]
    asm["qasm"] = prog.qasm()
    stats["instructions"] = len(prog.instrs)
    ck = CompiledKernel(tile.name, prog, list(lw.uniforms), dict(signature), constants, asm, stats)
    if verify:
        prog.words()     # every instruction must encode
    return ck


def _compile_program(fn, signature, constants, name, pid_max, options):
    """compile() up to the scheduled program: straight-line, else rerolled."""
    # PITON_REROLL_REDUCE=1 (default off) also rerolls 1-D kernels with row
    # reductions (softmax: one loop per pass, reroll.py) and loads transposed
    # 2-D tiles through transposed pointers (transpose); "rows" / "tiles"
    # enable only one of the two.
    mode = os.environ.get("PITON_REROLL_REDUCE", "0")
    rows, tiles = mode not in ("", "0", "tiles"), mode not in ("", "0", "rows")
    use_reroll = os.environ.get("PITON_REROLL", "1") != "0"

    def generate():
        return frontend.generate(fn, signature, constants, name=name)
    multipass = rows and use_reroll and reroll.applicable(generate(), reroll.SUB // 4, True) \
        and reroll.has_reductions(generate())
    sub = reroll.SUB
    try:
        tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options)
    except RegAllocError as e:
        if use_reroll and reroll.applicable(generate(), sub):
            # A large block does not fit the registers as straight-line code;
            # as a loop over chunks only one chunk is live at a time.
            tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options, sub)
            sub //= 2
        elif multipass or tiles and reroll.transposed_loads(generate()):
            # a row or tile too large for the registers: chunked passes, or loads
            # that need no permutation temporaries, may still fit
            prog, unrolled_error = None, e
        else:
            raise
    if (tiles and (prog is None or len(prog.instrs) > reroll.ICACHE_INSTRUCTIONS)
            and reroll.transposed_loads(generate())):
        try:
            candidate = _compile_tile(fn, signature, constants, name, pid_max, options, transposed=True)
        except RegAllocError:
            candidate = None
        if candidate is not None and (prog is None or len(candidate[1].instrs) < len(prog.instrs)):
            tile, prog, mf, lw, stats, asm = candidate
    if multipass:
        if prog is None or len(prog.instrs) > reroll.ICACHE_INSTRUCTIONS:
            # Every chunk halving adds loop iterations and carried-slot moves to
            # each pass: keep the largest chunks whose loops fit the cache, and
            # recompute instead of carrying only if carrying runs out of registers.
            best = None
            for carry in (True, False):
                for sub in (reroll.SUB, reroll.SUB // 2, reroll.SUB // 4):
                    if not reroll.applicable(generate(), sub, True):
                        continue
                    try:
                        cand = _compile_tile(fn, signature, constants, name, pid_max, options, sub, True, carry)
                    except RegAllocError:
                        continue
                    if best is None or len(best[1].instrs) > reroll.ICACHE_CAPACITY and \
                            len(cand[1].instrs) < len(best[1].instrs):
                        best = cand
                    if len(cand[1].instrs) <= reroll.ICACHE_CAPACITY:
                        break
                if best is not None and len(best[1].instrs) <= reroll.ICACHE_CAPACITY:
                    break
            if best is not None and (prog is None or len(best[1].instrs) < len(prog.instrs)):
                tile, prog, mf, lw, stats, asm = best
    elif prog is not None:
        # A straight-line program larger than the instruction cache stalls on
        # every cache line (reroll.py); a loop over 64-element chunks fits, and
        # heavier bodies (exact division) get smaller chunks until theirs does.
        while (_misses_icache(tile, prog, mf, stats) and sub >= 16 and use_reroll
               and reroll.applicable(generate(), sub)):
            rerolled = _compile_tile(fn, signature, constants, name, pid_max, options, sub)
            if len(rerolled[1].instrs) >= len(prog.instrs):
                break
            tile, prog, mf, lw, stats, asm = rerolled
            sub //= 2
    if prog is None:
        raise unrolled_error
    return tile, prog, lw, stats, asm
