import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
from kernels import add_kernel  # noqa: E402


@pytest.mark.parametrize("n,block", [(64, 64), (1000, 64), (4096, 256), (37, 16), (16, 16)])
def test_vector_add(n, block):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.full(n + 16, 7.0, dtype=np.float32)   # guard band checks masking
    grid = lambda meta: (piton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block)
    np.testing.assert_allclose(out[:n], x + y, rtol=1e-6, atol=1e-6)
    assert np.all(out[n:] == 7.0)
