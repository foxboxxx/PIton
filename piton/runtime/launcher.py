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
                 self_addr=0):
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
        elif kind == "zero":
            out.append(zero_addr)
        elif kind == "scratch":
            out.append(scratch_addr)
        elif kind == "self":
            out.append(self_addr)
        else:
            raise ValueError(kind)
    return out


def program_ids(grid):
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                yield (x, y, z), (gx, gy, gz)


def run_emulator(ck: CompiledKernel, grid, values: dict, num_qpus: int = 12, strict: bool = True,
                 v3d: Optional[V3D] = None):
    v = v3d or V3D(num_qpus=num_qpus, strict=strict)
    addrs = {}
    for name, val in values.items():
        if isinstance(val, np.ndarray):
            addrs[name] = v.mem.alloc(max(val.nbytes, 4), 64)
            v.mem.write_array(addrs[name], val)
    zero = v.mem.alloc(64, 64)
    scratch = v.mem.alloc(64, 64)
    unifs = []
    for pid, g in program_ids(grid):
        a = v.mem.alloc(4 * max(len(ck.uniforms), 1), 16)
        u = uniforms_for(ck, values, addrs, pid, g, zero, scratch, a)
        if u:
            v.mem.write_words(a, np.array(u, dtype=np.uint32))
        unifs.append(a)
    stats = v.run(v.load(ck.words), unifs)
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
        key = (tuple(sorted(signature.items())), tuple(sorted((k, repr(v)) for k, v in constants.items())),
               num_stages)
        ck = self.jitfn._cache.get(key)
        if ck is None:
            ck = compile(self.jitfn, signature, constants, num_stages=num_stages)
            self.jitfn._cache[key] = ck
        grid = self.grid(dict(constants, **values)) if callable(self.grid) else self.grid
        grid = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        if device == "emu":
            ck.last_stats = run_emulator(ck, grid, values)
        elif device == "pi":
            from .pi import run_on_pi
            ck.last_stats = run_on_pi(ck, grid, values)
        else:
            raise ValueError(f"unknown device {device!r}")
        return ck
