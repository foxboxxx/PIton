"""`@piton.jit`: a Triton-style decorator around a kernel function."""

from __future__ import annotations

import inspect


def cdiv(a, b):
    return (a + b - 1) // b


def next_power_of_2(n):
    return 1 << (int(n) - 1).bit_length()


class JITFunction:
    _piton_jit = True

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.src = inspect.getsource(fn)
        self.arg_names = list(inspect.signature(fn).parameters)
        self._cache = {}

    def __call__(self, *args, **kwargs):
        raise RuntimeError("@piton.jit kernels are launched with kernel[grid](...)")

    def __getitem__(self, grid):
        from .runtime.launcher import Launcher
        return Launcher(self, grid)

    def __repr__(self):
        return f"JITFunction({self.__name__})"


def jit(fn=None, **_kwargs):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)
