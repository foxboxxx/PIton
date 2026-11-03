"""Emulator tests for the kernels benchmarked by pi/examples/primitives.

The benchmark itself needs a Raspberry Pi; these cover the kernels it compiles
and the manifest that drives its build, so a broken primitive shows up in
`pytest -q` rather than only on the board.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
import primitive_kernels as P  # noqa: E402

rng = np.random.default_rng(7)


def test_copy():
    n = 1024
    x = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, np.float32)
    P.copy_kernel[(piton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    np.testing.assert_array_equal(out, x)


def test_fma():
    n = 1024
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, np.float32)
    P.fma_kernel[(piton.cdiv(n, 256),)](x, y, out, n, 2.5, BLOCK=256)
    # x * alpha + y cancels for operands of opposite sign, which turns a 1-ulp
    # difference in the product into a large *relative* error on a near-zero
    # result, so the bound that means anything here is the absolute one.
    np.testing.assert_allclose(out, x * np.float32(2.5) + y, rtol=1e-5, atol=1e-6)


def test_log():
    n = 1024
    x = (np.abs(rng.standard_normal(n)) * 7.0 + 0.25).astype(np.float32)
    out = np.zeros(n, np.float32)
    P.log_kernel[(piton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    np.testing.assert_allclose(out, np.log(x), rtol=2e-5, atol=1e-6)


def test_idiv_matches_c_semantics():
    """Truncation toward zero, with the remainder taking the dividend's sign."""
    n = 1024
    a = rng.integers(-10 ** 9, 10 ** 9, n, dtype=np.int32)
    b = rng.integers(1, 1 << 16, n, dtype=np.int32) * rng.choice([-1, 1], n).astype(np.int32)
    q = np.zeros(n, np.int32)
    r = np.zeros(n, np.int32)
    P.idiv_kernel[(piton.cdiv(n, 256),)](a, b, q, r, n, BLOCK=256)
    want_q = np.trunc(a.astype(np.float64) / b.astype(np.float64)).astype(np.int32)
    np.testing.assert_array_equal(q, want_q)
    np.testing.assert_array_equal(r, a - want_q * b)


def test_row_max():
    rows, cols = 256, 256
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    out = np.zeros(rows, np.float32)
    P.row_max_kernel[(rows // 4,)](x, out, cols, cols, BLOCK_M=4, BLOCK_N=64)
    np.testing.assert_array_equal(out, x.max(axis=1))


@pytest.mark.parametrize("rows,cols", [(256, 256), (64, 128)])
def test_transpose(rows, cols):
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    out = np.zeros((cols, rows), np.float32)
    grid = (piton.cdiv(rows, 16) * piton.cdiv(cols, 16),)
    P.transpose_kernel[grid](x, out, rows, cols, cols, rows, BLOCK_M=16, BLOCK_N=16)
    np.testing.assert_array_equal(out, x.T)


def test_benchmark_manifest_compiles_every_op():
    """Every entry in the benchmark's manifest must compile and name a real
    kernel, so `make` in pi/examples/primitives cannot fail at the first step."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pi", "examples",
                                    "primitives"))
    import build_kernels  # noqa: E402

    from piton import compiler
    from piton.tools.compile import _load_kernel

    assert len(build_kernels.OPS) >= 16
    names = [name for name, *_ in build_kernels.OPS]
    assert len(set(names)) == len(names), "duplicate output name in the manifest"
    for name, path, kernel, sig, cons, grid in build_kernels.OPS:
        fn = _load_kernel(os.path.join(build_kernels.ROOT, path), kernel)
        ck = compiler.compile(fn, dict(sig), dict(cons), name=name, grid=(grid,))
        assert ck.words, name
        # the generated launcher reads the uniforms in this order
        assert [n for _, n in ck.uniforms]
