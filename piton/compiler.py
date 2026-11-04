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
    # logical program ids per hardware program along axis 0 (PITON_PERSISTENT):
    # launch ceil(grid[0] / coarsen) programs; uniform "pid0" is the hardware
    # program's index, "nprog0" the logical grid
    coarsen: int = 1
    # PITON_COARSEN_RUNTIME: the program reads its ids per hardware program
    # from uniform "coarsen0" (1 ... coarsen), and each launch chooses them
    # (runtime/launcher.py launch_coarsen): (grid, hardware programs) from
    # which to coarsen, as _coarsening applies them to a known grid
    # (_coarsen_limits). None: every launch runs `coarsen` ids (compile-time or
    # forced factor).
    coarsen_launch: Optional[tuple] = None

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


# PITON_PERSISTENT=1 coarsens known 1-D grids of at least COARSEN_MIN_GRID
# programs (21 per QPU) by the largest power of two up to COARSEN_MAX (short
# program ids) or COARSEN_MAX_LONG that leaves COARSEN_MIN_PROGRAMS hardware
# programs. The Pi BLOCK sweeps (perf/results/tune-2026-11-03-*.json) ran
# 128-512 programs about equally fast and 64 (5 per QPU) 3-9% slower: the last
# programs leave QPUs idle. With 256 programs of ~90 us (idiv at 4x BLOCK)
# that tail cost 2% more than the per-program start saved, so ids that run
# more than COARSEN_SHORT instructions, or run-time loops, get at most 2.
COARSEN_MIN_GRID = 256
COARSEN_MIN_PROGRAMS = 128
# Coarsening lets an id's store DMAs drain while the next id runs, so what
# decides it is what the ids store, not what they load (Pi, 2026-11-04, one
# build launched with each k: perf/coarsen_sweep.py,
# perf/results/coarsen-launch-*.jsonl). At 0.48 bytes stored per executed
# instruction the digits' first layer (dense_relu) gains 1.13x on its full
# grid and from 64 ids up (1.12x at 64 with k=2, 0.96x at 32); add (2.8) from
# 128 up (1.21x at k=4 on 32 hardware programs). At 0.26 the second layer
# loses everywhere (0.988x on its full grid, 0.76-0.96x below), matmul 8x64x8
# at 0.14 too (0.79-0.93x below 128 ids). PITON_COARSEN_STORES=1 coarsens only
# kernels storing COARSEN_MIN_STORE_BYTES; with PITON_COARSEN_RUNTIME those
# use COARSEN_LAUNCH_MIN_GRID / _PROGRAMS (every launch, and the compile time
# with the largest grid), the rest COARSEN_MIN_GRID / _PROGRAMS.
COARSEN_MIN_STORE_BYTES = 0.4      # bytes stored per executed instruction
COARSEN_LAUNCH_MIN_GRID = 64
COARSEN_LAUNCH_MIN_PROGRAMS = 32
COARSEN_MAX = 4
COARSEN_MAX_LONG = 2
COARSEN_SHORT = 1500
# Fewer hardware programs leave a less even last round on the 12 QPUs, which
# costs compute-bound kernels 2-6% (Pi, 2026-11-03: recip 0.975x, rsqrt 0.966x,
# idiv 0.940x at 1.9-2.7 bytes per instruction) while kernels that wait on
# memory gain (add/mul/fma 1.21-1.29x, dense_relu 1.12x at 8.2-8.9).
COARSEN_MIN_BYTES = 4              # bytes loaded and stored per executed instruction
# A store through a shared slot waits for its DMAs before it releases the slot,
# where a store to private rows lets the program run on. That pays when stores
# are a large part of the work (copy 5.4 bytes stored per instruction: 1.48x)
# and costs when they are rare (Pi, 2026-11-03, matmul 8x64x8: 256^3 at 0.17
# bytes per instruction 1.05x faster with slots, 512^3 at 0.08 even, 1024^3 at
# 0.04 0.95x).
SLOTS_MIN_STORE_BYTES = 0.1        # bytes stored per executed instruction


def coarsen_stores_enabled() -> bool:
    """PITON_COARSEN_STORES=1: coarsen only kernels whose ids store at least
    COARSEN_MIN_STORE_BYTES per executed instruction."""
    return os.environ.get("PITON_COARSEN_STORES", "0") not in ("", "0")


def coarsen_runtime_enabled() -> bool:
    """PITON_COARSEN_RUNTIME=1: a coarsened program reads its ids per hardware
    program from a uniform, so one build launches small grids with one id per
    program (Lowering.begin_programs, CompiledKernel.coarsen_launch)."""
    return os.environ.get("PITON_COARSEN_RUNTIME", "0") not in ("", "0")


def _id_length(tile, prog: Program, mf, upper: bool = True) -> Optional[int]:
    """Instructions one program id executes at most: code outside loops once,
    code inside them times the trip count of every loop (None when a loop runs
    a run-time count). (With store slots the store after a rerolled loop, with
    its four slot copies of acquire and release, is ~40% of the program.)
    upper=False: inside code times the largest single trip count instead, close
    for loops one after another (softmax's passes) and low for nested ones."""
    trips = 1
    for op in tile.walk():
        if op.name != "scf.for":
            continue
        lb, ub, step = (v.op.attrs["value"] if v.op is not None and v.op.name == "arith.constant" else None
                        for v in op.operands[:3])
        if lb is None or ub is None or not step or step < 0:
            return None
        count = max(1, -(-(ub - lb) // step))
        trips = trips * count if upper else max(trips, count)
    depth = {b.label: b.loop_depth for b in mf.blocks}
    starts = sorted((idx, label) for label, idx in prog.labels.items())
    bounds = [idx for idx, _ in starts] + [len(prog.instrs)]
    n, current = bounds[0], 0               # code before the first label: depth 0
    for (idx, label), end in zip(starts, bounds[1:]):
        current = depth.get(label, current)
        n += (end - idx) * (trips if current else 1)
    return n


def _id_bytes(tile, kinds=("tt.load", "tt.store")) -> Optional[int]:
    """Bytes one program id loads and stores (`kinds`): every tensor load and
    store, times the trip counts of its loops (None when a loop runs a run-time count)."""
    def walk(block, trips):
        n = 0
        for op in block.ops:
            if op.name in kinds:
                n += trips * 4 * max(1, (op.result if op.name == "tt.load" else op.operands[1]).numel)
            inner = trips
            if op.name == "scf.for":
                lb, ub, step = (v.op.attrs["value"] if v.op is not None and v.op.name == "arith.constant" else None
                                for v in op.operands[:3])
                if lb is None or ub is None or not step or step < 0:
                    return None
                inner = trips * max(1, -(-(ub - lb) // step))
            for r in op.regions:
                m = walk(r, inner)
                if m is None:
                    return None
                n += m
        return n
    return walk(tile.body, 1)


def _stores_densely(tile, length) -> bool:
    stored = _id_bytes(tile, ("tt.store",))
    return length is not None and stored is not None and stored >= COARSEN_MIN_STORE_BYTES * length


def _coarsen_limits(tile, id_length) -> tuple:
    """(grid, hardware programs) from which a kernel is coarsened: with
    PITON_COARSEN_RUNTIME, COARSEN_LAUNCH_MIN_* for ids that store densely."""
    if coarsen_runtime_enabled() and _stores_densely(tile, id_length()):
        return COARSEN_LAUNCH_MIN_GRID, COARSEN_LAUNCH_MIN_PROGRAMS
    return COARSEN_MIN_GRID, COARSEN_MIN_PROGRAMS


def _coarsening(tile, pid_max, id_length) -> int:
    """Logical program ids per hardware program along axis 0 (PITON_PERSISTENT):
    unset or 1 -> automatic, 0 -> 1 (one hardware program per id), k >= 2 -> k
    for every grid. Kernels that never read tl.program_id(0) are not coarsened.
    id_length() is _id_length of the plain build."""
    env = os.environ.get("PITON_PERSISTENT", "1")
    try:
        want = int(env)
    except ValueError:
        raise ValueError(f"PITON_PERSISTENT={env!r}: expected 0, 1 or a factor >= 2") from None
    if want <= 0 or not any(op.name == "tt.get_program_id" and op.attrs["axis"] == 0
                            for op in tile.walk()):
        return 1
    bound = None if pid_max is None else pid_max[0] + 1
    if want >= 2:
        return want if bound is None else max(1, min(want, bound))
    if bound is None or any(pid_max[1:]):
        return 1
    min_grid, min_programs = _coarsen_limits(tile, id_length)
    if bound < min_grid:
        return 1
    length = id_length()
    if length is not None and (_id_bytes(tile) or 0) < COARSEN_MIN_BYTES * length:
        return 1
    if coarsen_stores_enabled() and not _stores_densely(tile, length):
        return 1
    k = COARSEN_MAX if length is not None and length <= COARSEN_SHORT else COARSEN_MAX_LONG
    while k > 1 and bound // k < min_programs:
        k //= 2
    return k


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
               # stores through shared 16-row VPM slots (backend/lower.py) where they are
               # frequent (SLOTS_MIN_STORE_BYTES); PITON_STORE_SLOTS=0: private rows,
               # =all: slots for every kernel
               "store_slots": os.environ.get("PITON_STORE_SLOTS", "1") not in ("", "0"),
               "store_slots_all": os.environ.get("PITON_STORE_SLOTS", "1") == "all"}
    if os.environ.get("PITON_MANDEL_FAST", "1") not in ("", "0"):
        # `if tl.max(x) > c` branches on the lanes of x, and a loop guarded by
        # such an if exits when it fails (examples/mandelbrot.py)
        options["fast_branches"] = True
    if coarsen_runtime_enabled():
        options["coarsen_runtime"] = True       # Lowering.begin_programs: the factor is a uniform
    try:
        tile, prog, lw, stats, asm, coarsen = _compile_program(fn, signature, constants, name, pid_max, options)
    except RegAllocError:
        # PITON_FAST_IDIV: the straight-line division keeps a whole tile's operands,
        # both results and the block's cached lane offsets and constants live at
        # once; a tile that only fits the registers with the branching division
        # (e.g. offs // N and offs % N of a 16x16 tile) gets that one
        if not fast_idiv_enabled():
            raise
        tile, prog, lw, stats, asm, coarsen = _compile_program(fn, signature, constants, name, pid_max,
                                                               dict(options, fast_idiv=False))
        stats["fast_idiv_fallback"] = 1
    prog.header = [f"{tile.name}: compiled by PIton (Triton -> VideoCore IV QPU)",
                   "uniforms: " + ", ".join(n for _, n in lw.uniforms),
                   "constants: " + ", ".join(f"{k}={v}" for k, v in constants.items())]
    launch = None
    if coarsen > 1 and options.get("coarsen_runtime") and int(os.environ.get("PITON_PERSISTENT", "1")) < 2:
        # (a forced factor, PITON_PERSISTENT=k checked by _coarsening, runs k ids in every launch)
        launch = (stats["coarsen_min_grid"], stats["coarsen_min_programs"])
        prog.header.append(f"coarsened: up to {coarsen} program ids (axis 0) per hardware program, "
                           f"chosen per launch (uniform coarsen0)")
    elif coarsen > 1:
        prog.header.append(f"coarsened: {coarsen} program ids (axis 0) per hardware program")
    asm["qasm"] = prog.qasm()
    stats["instructions"] = len(prog.instrs)
    ck = CompiledKernel(tile.name, prog, list(lw.uniforms), dict(signature), constants, asm, stats,
                        coarsen, launch)
    if verify:
        prog.words()     # every instruction must encode
    return ck


def _compile_program(fn, signature, constants, name, pid_max, options):
    """compile() up to the scheduled program: straight-line, else rerolled;
    with PITON_PERSISTENT, coarsened (returns the ids per hardware program too)."""
    # PITON_REROLL_REDUCE: "rows" (the default) also rerolls 1-D kernels with
    # row reductions (softmax: one loop per pass, reroll.py); "tiles" loads
    # transposed 2-D tiles through transposed pointers (transpose; off by
    # default: 0.71x on the Pi, each request's lanes span 16 rows); "1" both,
    # "0" neither.
    mode = os.environ.get("PITON_REROLL_REDUCE", "rows")
    rows, tiles = mode not in ("", "0", "tiles"), mode not in ("", "0", "rows")
    use_reroll = os.environ.get("PITON_REROLL", "1") != "0"

    def generate():
        return frontend.generate(fn, signature, constants, name=name)
    multipass = rows and use_reroll and reroll.applicable(generate(), reroll.SUB // 4, True) \
        and reroll.has_reductions(generate())
    sub = reroll.SUB
    try:
        tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options)
        variant = ((), {})          # _compile_tile's arguments for the chosen build (coarsening rebuilds it)
    except RegAllocError as e:
        if use_reroll and reroll.applicable(generate(), sub):
            # A large block does not fit the registers as straight-line code;
            # as a loop over chunks only one chunk is live at a time.
            tile, prog, mf, lw, stats, asm = _compile_tile(fn, signature, constants, name, pid_max, options, sub)
            variant = ((sub,), {})
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
            variant = ((), {"transposed": True})
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
                        cand_variant = ((sub, True, carry), {})
                    except RegAllocError:
                        continue
                    if best is None or len(best[1].instrs) > reroll.ICACHE_CAPACITY and \
                            len(cand[1].instrs) < len(best[1].instrs):
                        best, best_variant = cand, cand_variant
                    if len(cand[1].instrs) <= reroll.ICACHE_CAPACITY:
                        break
                if best is not None and len(best[1].instrs) <= reroll.ICACHE_CAPACITY:
                    break
            if best is not None and (prog is None or len(best[1].instrs) < len(prog.instrs)):
                tile, prog, mf, lw, stats, asm = best
                variant = best_variant
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
            variant = ((sub,), {})
            sub //= 2
    if prog is None:
        raise unrolled_error
    # Each program pays a start and a final wait for its last store DMA, so with
    # many short programs one hardware program runs several program ids in a loop
    # (Lowering.begin_programs), chunked like the plain build. The kernel's
    # uniforms stay in registers or are re-read for every id, whichever is
    # shorter. Constants hoisted out of that loop stay live through every id;
    # without hoisting a kernel needs only ~3 registers more than its plain
    # build. If no version allocates, the plain build stays; so does a plain
    # build with store slots whose coarsened versions allocate only without
    # them. (With slots every store waits for its DMAs before it releases the
    # slot, so only the start is saved; the private rows' last DMA of an id
    # still runs on into the next.)
    if stats.get("store_slots") and not options.get("store_slots_all"):
        length = _id_length(tile, prog, mf, upper=False)
        stored = _id_bytes(tile, ("tt.store",))
        if length and stored is not None and stored < SLOTS_MIN_STORE_BYTES * length:
            return _compile_program(fn, signature, constants, name, pid_max, dict(options, store_slots=False))
    coarsen = _coarsening(tile, pid_max, lambda: _id_length(tile, prog, mf))
    if coarsen > 1:
        limits = _coarsen_limits(tile, lambda: _id_length(tile, prog, mf)) if options.get("coarsen_runtime") else None
        best = None
        for hoist in (True, False):
            for reload in (False, True):
                opts = dict(options, coarsen=coarsen, coarsen_reload=reload, coarsen_hoist=hoist)
                try:
                    built = _compile_tile(fn, signature, constants, name, pid_max, opts, *variant[0], **variant[1])
                except RegAllocError:
                    continue
                if stats.get("store_slots") and not built[4].get("store_slots"):
                    # it allocated only with the private VPM rows: the plain
                    # build's 1 KiB slot DMAs gain more than coarsening
                    continue
                if best is None or _weighted_length(built[1], built[2]) < _weighted_length(best[1], best[2]):
                    best = built
            if best is not None:
                break
        if best is None:
            coarsen = 1
        else:
            tile, prog, mf, lw, stats, asm = best
            if limits:
                stats["coarsen_min_grid"], stats["coarsen_min_programs"] = limits
    return tile, prog, lw, stats, asm, coarsen
