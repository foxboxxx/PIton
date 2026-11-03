import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from mandelbrot import TARGET, escape_counts, make_palette, reference_counts, render  # noqa: E402

W, H, ITERS = 64, 24, 64
STEP = np.float32(3.2 / W)
X0 = np.float32(-0.6 - 1.6)
Y0 = np.float32(-float(STEP) * (H / 2 - 0.5))   # no row exactly on the real axis


def test_counts_match_float64_reference():
    got = escape_counts(X0, Y0, STEP, W, H, ITERS)
    ref = reference_counts(X0, Y0, STEP, W, H, ITERS)
    # Round-toward-zero arithmetic moves a few boundary pixels by some iterations.
    assert (got != ref).mean() < 0.02
    assert got.min() >= 1 and got.max() == ITERS


@pytest.mark.parametrize("block,unroll,tile_h", [(16, 8, 1), (32, 4, 1), (64, 8, 1), (16, 16, 2)])
def test_early_exit_unroll_and_tiles_do_not_change_counts(block, unroll, tile_h):
    # The plain loop (UNROLL=1 still skips, but only after whole blocks escape)
    # and every chunked or tiled configuration must agree exactly.
    base = escape_counts(X0, Y0, STEP, W, H, ITERS, block=16, unroll=1)
    np.testing.assert_array_equal(escape_counts(X0, Y0, STEP, W, H, ITERS, block, unroll, tile_h), base)


def test_several_rows_per_program():
    base = escape_counts(X0, Y0, STEP, W, H, ITERS, block=16, unroll=8)
    np.testing.assert_array_equal(escape_counts(X0, Y0, STEP, W, H, ITERS, 16, 8, rows=8), base)


def test_zoomed_view_and_partial_last_block():
    width, height, iters = 40, 8, 128   # 40 is not a multiple of the block
    step = np.float32(3.5 / 100 / 640)
    x0 = np.float32(TARGET[0] - width / 2 * float(step))
    y0 = np.float32(TARGET[1] - height / 2 * float(step))
    got = escape_counts(x0, y0, step, width, height, iters, block=16, unroll=8)
    ref = reference_counts(x0, y0, step, width, height, iters)
    # Nearly every pixel here is on the boundary, where fp32 rounding matters.
    assert (got != ref).mean() < 0.15
    image, palette = render(x0, y0, step, width, height, iters, block=16, unroll=8)
    np.testing.assert_array_equal(image, palette[np.where(got == iters, 0, got)])
    assert palette[0] == 0 and np.all(palette[1:] != 0)
