"""Compiler driver: Triton kernel -> tile IR -> machine IR -> QPU program."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional

from . import frontend, ir, passes
from .backend import opt
from .backend.lower import Lowering
from .backend.regalloc import allocate
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
            name: Optional[str] = None, verify: bool = True) -> CompiledKernel:
    """Compile a Triton kernel for the VideoCore IV QPU.

    signature  -- {arg: "*fp32" | "*i32" | "i32" | "fp32"} for runtime arguments
    constants  -- {arg: value} for tl.constexpr arguments and specialised scalars
    num_stages -- 1 disables software pipelining of loop loads
    """
    fn = _pyfn(kernel)
    constants = dict(constants or {})
    tile = frontend.generate(fn, signature, constants, name=name)
    asm = {"ttir": tile.dump()}
    ranges = passes.optimize(tile)
    asm["ttir_opt"] = tile.dump()
    lw = Lowering(tile, ranges, {"num_stages": num_stages})
    mf = lw.run()
    stats = dict(lw.stats)
    stats.update(opt.run(mf))
    asm["mir"] = mf.dump()
    alloc = allocate(mf)
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
