"""Compiler driver: Triton kernel -> tile IR -> machine IR -> QPU program."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Optional

from . import frontend, ir, passes, reroll
from .backend import opt, presched
from .backend.lower import Lowering
from .backend.regalloc import RegAllocError, allocate
from .backend.sched import schedule_function
from .qpu.asm import Program


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


def _build(tile, ranges, options: dict, interleave: bool):
    """Lower, allocate and schedule; returns (program, mir, lowering, stats, mir dumps)."""
    for cache_dot_a in (True, False):
        lw = Lowering(tile, ranges, dict(options, cache_dot_a=cache_dot_a))
        mf = lw.run()
        stats = dict(lw.stats)
        stats.update(opt.run(mf))
        stats["presched_blocks"] = presched.run(mf) if interleave else 0
        mir = mf.dump()
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
    prog = schedule_function(mf, tile.name)
    return prog, mf, lw, stats, mir, mir_alloc


def _compile_tile(fn, signature, constants, name, pid_max, options, chunk: int = 0):
    """Tile IR (rerolled into `chunk`-element pieces if nonzero) through
    scheduling; picks the better of the plain and pre-scheduled builds."""
    tile = frontend.generate(fn, signature, constants, name=name)
    asm = {"ttir": tile.dump()}
    if chunk:
        reroll.run(tile, chunk)
        asm["ttir_rerolled"] = tile.dump()
    ranges = passes.optimize(tile, pid_max)
    asm["ttir_opt"] = tile.dump()
    # Interleaving independent chains before allocation (presched.py) usually
    # shortens the schedule but needs more registers; keep whichever version
    # allocates and runs shorter, weighting loop bodies by their depth.
    prog, mf, lw, stats, asm["mir"], asm["mir_alloc"] = _build(tile, ranges, options, interleave=False)
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
    options = {"num_stages": num_stages, "store_mutex": store_mutex}
    if os.environ.get("PITON_MANDEL_FAST", "0") not in ("", "0"):
        # `if tl.max(x) > c` branches on the lanes of x, and a loop guarded by
        # such an if exits when it fails (examples/mandelbrot.py)
        options["fast_branches"] = True
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
