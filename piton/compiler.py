"""Compiler driver: Triton kernel -> tile IR -> machine IR -> QPU program."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional

from . import frontend, ir, passes
from .backend import opt
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
    tile = frontend.generate(fn, signature, constants, name=name)
    asm = {"ttir": tile.dump()}
    pid_max = None
    if grid is not None:
        g = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        pid_max = tuple(max(1, int(x)) - 1 for x in g) + (0,) * (3 - len(g))
    ranges = passes.optimize(tile, pid_max)
    asm["ttir_opt"] = tile.dump()
    for cache_dot_a in (True, False):
        lw = Lowering(tile, ranges, {"num_stages": num_stages, "store_mutex": store_mutex,
                                     "cache_dot_a": cache_dot_a})
        mf = lw.run()
        stats = dict(lw.stats)
        stats.update(opt.run(mf))
        asm["mir"] = mf.dump()
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
    asm["mir_alloc"] = mf.dump()
    prog = schedule_function(mf, tile.name)
    prog.header = [f"{tile.name}: compiled by PIton (Triton -> VideoCore IV QPU)",
                   "uniforms: " + ", ".join(n for _, n in lw.uniforms),
                   "constants: " + ", ".join(f"{k}={v}" for k, v in constants.items())]
    asm["qasm"] = prog.qasm()
    stats["instructions"] = len(prog.instrs)
    ck = CompiledKernel(tile.name, prog, list(lw.uniforms), dict(signature), constants, asm, stats)
    if verify:
        prog.words()     # every instruction must encode
    return ck
