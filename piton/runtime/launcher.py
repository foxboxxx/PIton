"""`kernel[grid](*args)` for PIton kernels.

Arguments follow Triton's conventions: numpy arrays become pointers (the
array is copied in, and copied back out after the launch), Python ints and
floats become scalars, and `tl.constexpr` parameters are compile-time
constants.  Kernels run on the emulator by default, or on a Raspberry Pi
with `device="pi"`.
"""

from __future__ import annotations

import inspect
from typing import Optional

import numpy as np

from ..compiler import CompiledKernel, compile
from ..language.core import constexpr
from ..qpu import isa
from ..qpu.asm import Program
from ..qpu.emulator import V3D
from ..qpu.isa import f2u

_DTYPE_SIG = {np.dtype(np.float32): "*fp32", np.dtype(np.int32): "*i32", np.dtype(np.uint32): "*u32"}


def _sig_of(value):
    if isinstance(value, np.ndarray):
        if value.dtype not in _DTYPE_SIG:
            raise TypeError(f"unsupported array dtype {value.dtype} (use float32/int32)")
        return _DTYPE_SIG[value.dtype]
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, (int, np.integer)):
        return "i32"
    if isinstance(value, (float, np.floating)):
        return "fp32"
    raise TypeError(f"unsupported kernel argument {type(value).__name__}")


def bind(fn, args, kwargs, do_not_specialize=()):
    """Split launch arguments into (signature, constants, runtime values)."""
    params = list(inspect.signature(fn).parameters.values())
    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    signature, constants, values = {}, {}, {}
    for p in params:
        if p.name not in bound.arguments:
            if p.default is inspect.Parameter.empty:
                raise TypeError(f"missing kernel argument '{p.name}'")
            constants[p.name] = p.default.value if isinstance(p.default, constexpr) else p.default
            continue
        v = bound.arguments[p.name]
        ann = p.annotation
        if ann is constexpr or (isinstance(ann, str) and "constexpr" in ann):
            constants[p.name] = v
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v == 1 \
                and p.name not in do_not_specialize:
            constants[p.name] = 1        # like Triton: integers equal to 1 are specialised
        else:
            signature[p.name] = _sig_of(v)
            values[p.name] = v
    return signature, constants, values


def uniforms_for(ck: CompiledKernel, values: dict, addrs: dict, pid, grid, zero_addr, scratch_addr,
                 self_addr=0, coarsen=None):
    """The uniform words of hardware program `pid` of a launch on `grid` (the
    logical grid); coarsen: the launch's ids per hardware program (uniform
    coarsen0; default launch_coarsen)."""
    out = []
    for kind, name in ck.uniforms:
        if kind == "arg":
            v = values[name]
            if isinstance(v, np.ndarray):
                out.append(addrs[name])
            elif isinstance(v, (float, np.floating)):
                out.append(f2u(float(v)))
            else:
                out.append(int(v) & 0xFFFFFFFF)
        elif kind == "pid":
            out.append(pid[int(name[3:])])
        elif kind == "nprog":
            out.append(grid[int(name[5:])])
        elif kind == "coarsen":
            out.append(launch_coarsen(ck, grid) if coarsen is None else coarsen)
        elif kind == "zero":
            out.append(zero_addr)
        elif kind == "scratch":
            out.append(scratch_addr)
        elif kind == "self":
            out.append(self_addr)
        else:
            raise ValueError(kind)
    return out


def launch_coarsen(ck: CompiledKernel, grid) -> int:
    """Program ids per hardware program along axis 0 for a launch of `ck` on
    `grid`: ck.coarsen, unless each launch chooses them (PITON_COARSEN_RUNTIME,
    ck.coarsen_launch; the program reads them from uniform coarsen0): then, as
    compiler._coarsening chooses for a known grid, ck.coarsen for at least
    COARSEN_MIN_GRID launched programs, halved until COARSEN_MIN_PROGRAMS
    hardware programs remain, and 1 (the plain program's ids, one hardware
    program each) for smaller launches."""
    if ck.coarsen_launch is None:
        return ck.coarsen
    min_grid, min_programs = ck.coarsen_launch
    gx = (list(grid) + [1])[0]
    if gx < min_grid:
        return 1
    k = ck.coarsen
    while k > 1 and gx // k < min_programs:
        k //= 2
    return k


def program_ids(grid, coarsen: int = 1):
    """(program ids, logical grid) for every hardware program of a launch.

    A kernel compiled with coarsen=k (ck.coarsen, PITON_PERSISTENT) runs k
    consecutive axis-0 ids per hardware program: there are ceil(gx / k) of them
    along axis 0, each given its own index as pid0 and the logical grid. (With
    PITON_COARSEN_RUNTIME the launch's factor is launch_coarsen(ck, grid).)"""
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    hx = -(-gx // coarsen)
    for z in range(gz):
        for y in range(gy):
            for x in range(hx):
                yield (x, y, z), (gx, gy, gz)


def slot_semaphores(words) -> tuple:
    """Semaphores a program acquires.  PIton code acquires only the semaphores
    of its store slots (backend/lower.py, PITON_STORE_SLOTS), and each of them
    must be 1 (slot free) while the program runs."""
    return tuple(sorted({ins.sema for ins in map(isa.decode, words) if ins.kind == "sem" and ins.acquire}))


def semaphore_programs(sems) -> tuple:
    """(init, fini) machine words for a launch of a kernel with store slots.

    The hardware semaphores keep their counts across programs and launches
    and are 0 after a reset.  `init` releases each slot semaphore once and
    runs to completion before the kernel's programs are queued; every kernel
    program releases what it acquires; `fini`, queued after all of them have
    finished, acquires each once more.  So a launch leaves the semaphores as
    it found them (0), and neither program can wait: init only releases
    (counts 0 -> 1) and fini finds every slot free (1 -> 0).
    """
    def prog(name, acquire):
        code = [isa.Instr(kind="sem", acquire=acquire, sema=s) for s in sems]
        code += [isa.Instr(kind="alu", sig="thrend"), isa.Instr.nop(), isa.Instr.nop()]
        return Program(name, code).words()
    return prog("slots_init", False), prog("slots_fini", True)


def run_emulator(ck: CompiledKernel, grid, values: dict, num_qpus: int = 12, strict: bool = True,
                 v3d: Optional[V3D] = None, coarsen: Optional[int] = None):
    """coarsen: ids per hardware program, 1 ... ck.coarsen, for a kernel that
    reads them from uniform coarsen0 (PITON_COARSEN_RUNTIME; default
    launch_coarsen)."""
    v = v3d or V3D(num_qpus=num_qpus, strict=strict)
    words = ck.words
    sems = slot_semaphores(words)
    addrs = {}
    for name, val in values.items():
        if isinstance(val, np.ndarray):
            addrs[name] = v.mem.alloc(max(val.nbytes, 4), 64)
            v.mem.write_array(addrs[name], val)
    zero = v.mem.alloc(64, 64)
    scratch = v.mem.alloc(64, 64)
    unifs = []
    k = launch_coarsen(ck, grid) if coarsen is None else coarsen
    if k != ck.coarsen and not (("coarsen", "coarsen0") in ck.uniforms and 1 <= k <= ck.coarsen):
        raise ValueError(f"{ck.name} runs {ck.coarsen} program ids per hardware program, not {k}")
    for pid, g in program_ids(grid, k):
        a = v.mem.alloc(4 * max(len(ck.uniforms), 1), 16)
        u = uniforms_for(ck, values, addrs, pid, g, zero, scratch, a, k)
        if u:
            v.mem.write_words(a, np.array(u, dtype=np.uint32))
        unifs.append(a)
    if sems:
        # as piton_launch_slots (pi/runtime/piton_rt.c): init, the kernel, fini
        init, fini = semaphore_programs(sems)
        before, counts = list(v.sems), v.pc_counts
        v.pc_counts = None
        v.run(v.load(init), [zero])
        v.pc_counts = counts
    stats = v.run(v.load(words), unifs)
    if sems:
        expected = [c + (i in sems) for i, c in enumerate(before)]
        if v.sems != expected:
            raise RuntimeError(f"store slot semaphores unbalanced after the kernel: {v.sems}, "
                               f"expected {expected}")
        v.pc_counts = None
        v.run(v.load(fini), [zero])
        v.pc_counts = counts
    for name, val in values.items():
        if isinstance(val, np.ndarray):
            val[...] = v.mem.read_array(addrs[name], val.shape, val.dtype)
    if not np.array_equal(v.mem.read_array(zero, (16,), np.uint32), np.zeros(16, np.uint32)):
        raise RuntimeError("kernel wrote to the zero page")
    return stats


class Launcher:
    def __init__(self, jitfn, grid, extra_constants=None):
        self.jitfn = jitfn
        self.grid = grid
        self.extra_constants = dict(extra_constants or {})

    def __call__(self, *args, device: str = "emu", num_stages: int = 2, **kwargs):
        fn = self.jitfn.fn
        launch_opts = {k: kwargs.pop(k) for k in list(kwargs) if k in ("num_warps", "num_ctas")}
        signature, constants, values = bind(fn, args, kwargs, getattr(self.jitfn, "do_not_specialize", ()))
        constants.update(self.extra_constants)
        grid = self.grid(dict(constants, **values)) if callable(self.grid) else self.grid
        grid = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        # compiled for the grid rounded up to a power of two (program-id ranges)
        bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)
        key = (tuple(sorted(signature.items())), tuple(sorted((k, repr(v)) for k, v in constants.items())),
               num_stages, bound)
        ck = self.jitfn._cache.get(key)
        if ck is None:
            ck = compile(self.jitfn, signature, constants, num_stages=num_stages, grid=bound)
            self.jitfn._cache[key] = ck
        if device == "emu":
            ck.last_stats = run_emulator(ck, grid, values)
        elif device == "pi":
            from .pi import run_on_pi
            ck.last_stats = run_on_pi(ck, grid, values)
        else:
            raise ValueError(f"unknown device {device!r}")
        return ck
