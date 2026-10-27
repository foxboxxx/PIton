"""Hardware tests: run compiled kernels on an attached Raspberry Pi.

Skipped unless a Pi, the bootloader and the course libpi are available.
Results must match the emulator bit for bit (the emulator models the
QPU's round-toward-zero float arithmetic).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
from piton.runtime.pi import pi_available  # noqa: E402

pytestmark = pytest.mark.skipif(not pi_available() or os.environ.get("PITON_HW") != "1",
                                reason="set PITON_HW=1 with a Pi attached")


def _both(kernel, grid, make_args, **const):
    emu = make_args()
    hw = make_args()
    kernel[grid](*emu, **const)
    kernel[grid](*hw, device="pi", **const)
    return emu, hw


def test_vector_add_hw():
    from kernels import add_kernel
    n = 1000
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    def args():
        return [x.copy(), y.copy(), np.full(n + 16, 7.0, np.float32), n]
    emu, hw = _both(add_kernel, (piton.cdiv(n, 64),), args, BLOCK_SIZE=64)
    np.testing.assert_array_equal(hw[2], emu[2])
    np.testing.assert_allclose(hw[2][:n], x + y, rtol=1e-6)
    assert np.all(hw[2][n:] == 7.0)
