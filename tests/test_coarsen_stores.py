"""PITON_COARSEN_STORES: coarsen only kernels whose ids store densely.

Coarsening lets an id's store DMAs drain while the next id runs; with the
switch on, compiler._coarsening also needs COARSEN_MIN_STORE_BYTES stored per
executed instruction. That keeps the elementwise kernels and the digits' first
layer coarsened and builds the second and third layers, the matmul and the
run-time loop of row_sum (whose length is unknown) plain. The plain builds are
the ones every other test checks; here only the choice is.
"""
import os
import sys

import pytest

from piton import compiler

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.dirname(__file__))
from kernels import add_kernel, matmul_kernel, row_sum_kernel  # noqa: E402
from primitive_kernels import copy_kernel  # noqa: E402
from test_coarsen_runtime import _digits_layer  # noqa: E402
from test_persistent import ADD_SIG, COPY_SIG, MATMUL_SIG, ROW_SIG  # noqa: E402

MATMUL = dict(K=256, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8,
              GROUP_SIZE_M=4)


def _builds():
    return {"add": compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, grid=(1024,)),
            "copy": compiler.compile(copy_kernel, COPY_SIG, {"BLOCK": 256}, grid=(1024,)),
            "dense1": _digits_layer(1), "dense2": _digits_layer(2), "dense3": _digits_layer(3, (256,)),
            "matmul": compiler.compile(matmul_kernel, MATMUL_SIG, MATMUL, grid=(512,)),
            "row_sum": compiler.compile(row_sum_kernel, ROW_SIG, {"BLOCK_M": 4, "BLOCK_N": 64}, grid=(1024,))}


@pytest.fixture
def stores(monkeypatch):
    monkeypatch.setenv("PITON_PERSISTENT", "1")
    monkeypatch.setenv("PITON_COARSEN_RUNTIME", "0")

    def set_switch(value):
        if value is None:
            monkeypatch.delenv("PITON_COARSEN_STORES", raising=False)
        else:
            monkeypatch.setenv("PITON_COARSEN_STORES", value)
    return set_switch


def test_switch(stores):
    """"" and "0" coarsen by E1's rule alone (every build the same code); "1"
    and unset (on by default) leave the sparse stores plain."""
    stores("0")
    base = _builds()
    assert {n: ck.coarsen for n, ck in base.items()} == \
        {"add": 4, "copy": 4, "dense1": 2, "dense2": 2, "dense3": 2, "matmul": 2, "row_sum": 2}
    stores("")
    assert all(list(ck.words) == list(base[n].words) and ck.uniforms == base[n].uniforms
               for n, ck in _builds().items())
    stores(None)
    default = _builds()
    stores("1")
    on = _builds()
    assert all(list(ck.words) == list(default[n].words) for n, ck in on.items())
    assert {n: ck.coarsen for n, ck in on.items()} == \
        {"add": 4, "copy": 4, "dense1": 2, "dense2": 1, "dense3": 1, "matmul": 1, "row_sum": 1}
    for n in ("add", "copy", "dense1"):
        assert list(on[n].words) == list(base[n].words) and on[n].uniforms == base[n].uniforms, n


def test_plain_builds_are_the_uncoarsened_ones(stores, monkeypatch):
    """A kernel the switch leaves alone is exactly PITON_PERSISTENT=0's build."""
    stores("1")
    on = _builds()
    monkeypatch.setenv("PITON_PERSISTENT", "0")
    stores(None)
    plain = _builds()
    for n in ("dense2", "dense3", "matmul", "row_sum"):
        assert list(on[n].words) == list(plain[n].words) and on[n].uniforms == plain[n].uniforms, n


def test_with_the_launch_choosing_k(stores, monkeypatch):
    """With PITON_COARSEN_RUNTIME too: the dense stores are coarsened from
    COARSEN_LAUNCH_MIN_GRID ids, the others not at all."""
    stores("1")
    monkeypatch.setenv("PITON_COARSEN_RUNTIME", "1")
    on = _builds()
    launch = (compiler.COARSEN_LAUNCH_MIN_GRID, compiler.COARSEN_LAUNCH_MIN_PROGRAMS)
    assert {n: (ck.coarsen, ck.coarsen_launch) for n, ck in on.items()} == \
        {"add": (4, launch), "copy": (4, launch), "dense1": (2, launch), "dense2": (1, None),
         "dense3": (1, None), "matmul": (1, None), "row_sum": (1, None)}
    assert _digits_layer(3).coarsen == 1 and _digits_layer(1, (64,)).coarsen == 2


def test_store_density_of_the_measured_kernels(stores):
    """The bytes stored per executed instruction behind COARSEN_MIN_STORE_BYTES
    (Pi sweep 2026-11-04): the first layer and add above it, the second and
    third layers and the matmul below."""
    stores(None)
    seen = {}
    real = compiler._coarsening

    def spy(tile, pid_max, id_length):
        seen["last"] = compiler._id_bytes(tile, ("tt.store",)) / id_length()
        return real(tile, pid_max, id_length)
    try:
        compiler._coarsening = spy
        density = {}
        for name, build in (("add", lambda: compiler.compile(add_kernel, ADD_SIG, {"BLOCK_SIZE": 256}, grid=(1024,))),
                            ("dense1", lambda: _digits_layer(1)), ("dense2", lambda: _digits_layer(2)),
                            ("dense3", lambda: _digits_layer(3)),
                            ("matmul", lambda: compiler.compile(matmul_kernel, MATMUL_SIG, MATMUL, grid=(512,)))):
            build()
            density[name] = seen["last"]
    finally:
        compiler._coarsening = real
    assert density["add"] > density["dense1"] > compiler.COARSEN_MIN_STORE_BYTES
    assert compiler.COARSEN_MIN_STORE_BYTES > density["dense3"] > density["dense2"] > density["matmul"]
