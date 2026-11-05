"""Image filters as tiled 2-D Triton stencils: Gaussian blur and Sobel edges.

Every kernel takes a float32 grayscale image (row stride `src_stride`) and
writes one float32 output image. Program (i, j) of the 2-D grid
(cdiv(W, BLOCK_W), cdiv(H, ROWS)) covers a BLOCK_W-wide strip of ROWS rows,
in BLOCK_H x BLOCK_W tiles (a runtime loop, so the code stays small and a
VGA image needs a few hundred programs instead of thousands). Each tap of the
stencil is one tl.load of the whole tile, shifted by (dy, dx).

Borders: BORDER=CLAMP (the default) reads the nearest edge pixel (OpenCV's
BORDER_REPLICATE, scipy's mode="nearest"): the row and column indices are
clamped, so the loads need no mask. BORDER=ZERO reads 0 outside the image
with masked loads (`other=0.0`). Stores are masked to the image in both.

    blur_kernel        binomial Gaussian, 3x3 ([1 2 1]/4 per axis) or 5x5
                       ([1 4 6 4 1]/16), every tap a 2-D load: (2R+1)^2 loads
    blur_pass_kernel   one axis of the same blur; two launches (rows, then
                       columns) make the separable blur: 2(2R+1) loads
    blur_rolling_kernel  the 5x5 blur in one pass with the column filter
                       rolling down the strip: each row is loaded (5 taps) and
                       filtered horizontally once, and the last four filtered
                       rows are carried in registers by the loop
    sobel_kernel       Sobel gradient magnitude sqrt(gx^2 + gy^2), 8 loads
    blur_sobel_kernel  3x3 blur fused into Sobel: one 5x5 stencil (24 loads)
                       whose weights are the two filters convolved

The weights are small integers and the image holds integers 0..255, so every
blur sum is an exact float32 integer and the power-of-two normalisation is
exact too: the blurs give the same bits in any summation order, on the QPU
(which rounds toward zero) and in numpy. So do the separable and rolling
versions, and the fused blur+Sobel matches blur then Sobel bit for bit away
from the 2-pixel border (at the border, clamping the 5x5 stencil is not the
same as clamping each 3x3 pass). Only the Sobel square root differs from
numpy (PIton's sqrt is ~1 ulp, on the ALUs).

    .venv/bin/python examples/image_filters.py     # emulator previews in build/
"""
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402

CLAMP = 0
ZERO = 1


@triton.jit
def clamp(x, lo, hi):
    return tl.minimum(tl.maximum(x, lo), hi)


@triton.jit
def tap(src_ptr, rows, cols, H, W, stride, BORDER: tl.constexpr):
    """The tile at rows x cols (1-D index vectors, possibly outside the image)."""
    if BORDER == 0:
        # clamp the row *offset*, not the row: (rows * stride) stays affine in
        # the program id, so only the min/max are per element
        row_off = clamp(rows * stride, 0, (H - 1) * stride)
        c = clamp(cols, 0, W - 1)
        x = tl.load(src_ptr + row_off[:, None] + c[None, :])
    else:
        inside = ((rows >= 0) & (rows < H))[:, None] & ((cols >= 0) & (cols < W))[None, :]
        x = tl.load(src_ptr + rows[:, None] * stride + cols[None, :], mask=inside, other=0.0)
    return x


@triton.jit
def blur_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, ROWS: tl.constexpr,
                RADIUS: tl.constexpr, BORDER: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    for r in range(0, ROWS, BLOCK_H):
        rows = tl.program_id(1) * ROWS + r + tl.arange(0, BLOCK_H)
        for i in tl.static_range(2 * RADIUS + 1):
            wi = 1                                   # binomial(2 * RADIUS, i)
            for k in tl.static_range(i):
                wi = wi * (2 * RADIUS - k) // (k + 1)
            for j in tl.static_range(2 * RADIUS + 1):
                wj = 1
                for k in tl.static_range(j):
                    wj = wj * (2 * RADIUS - k) // (k + 1)
                x = tap(src_ptr, rows + (i - RADIUS), cols + (j - RADIUS), H, W, src_stride, BORDER)
                # x * 1 and 0 + x are not no-ops on the QPU (-0 and denormals
                # become +0), so the compiler keeps them: leave them out here
                if j == 0:
                    row_sum = x
                else:
                    row_sum += x * wj
            if i == 0:
                acc = row_sum
            else:
                acc += row_sum * wi
        out = acc * (1.0 / (1 << (4 * RADIUS)))      # / (sum of weights)^2, exact
        mask = (rows[:, None] < H) & (cols[None, :] < W)
        tl.store(dst_ptr + rows[:, None] * dst_stride + cols[None, :], out, mask=mask)


@triton.jit
def blur_pass_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                     BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, ROWS: tl.constexpr,
                     RADIUS: tl.constexpr, BORDER: tl.constexpr, AXIS: tl.constexpr):
    """The blur along one axis: AXIS=1 filters each row, AXIS=0 each column."""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    for r in range(0, ROWS, BLOCK_H):
        rows = tl.program_id(1) * ROWS + r + tl.arange(0, BLOCK_H)
        for j in tl.static_range(2 * RADIUS + 1):
            wj = 1
            for k in tl.static_range(j):
                wj = wj * (2 * RADIUS - k) // (k + 1)
            if AXIS == 1:
                x = tap(src_ptr, rows, cols + (j - RADIUS), H, W, src_stride, BORDER)
            else:
                x = tap(src_ptr, rows + (j - RADIUS), cols, H, W, src_stride, BORDER)
            if j == 0:
                acc = x
            else:
                acc += x * wj
        out = acc * (1.0 / (1 << (2 * RADIUS)))
        mask = (rows[:, None] < H) & (cols[None, :] < W)
        tl.store(dst_ptr + rows[:, None] * dst_stride + cols[None, :], out, mask=mask)


@triton.jit
def blur_row(src_ptr, row_off, cols, W):
    """[1 4 6 4 1] along the image row at offset row_off, columns clamped."""
    acc = tl.load(src_ptr + row_off + clamp(cols - 2, 0, W - 1))
    acc += tl.load(src_ptr + row_off + clamp(cols - 1, 0, W - 1)) * 4
    acc += tl.load(src_ptr + row_off + clamp(cols, 0, W - 1)) * 6
    acc += tl.load(src_ptr + row_off + clamp(cols + 1, 0, W - 1)) * 4
    acc += tl.load(src_ptr + row_off + clamp(cols + 2, 0, W - 1))
    return acc


@triton.jit
def blur_rolling_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                        BLOCK_W: tl.constexpr, ROWS: tl.constexpr):
    """5x5 binomial blur (clamped borders), one image row per loop iteration.

    Output row y needs the horizontally filtered rows y-2 ... y+2. Four of them
    were filtered by the previous iterations and are carried in registers (h0
    ... h3), so each iteration loads and filters one new row: 5 loads per pixel
    instead of 25 (blur_kernel) or 10 plus a round trip through memory
    (blur_pass_kernel twice)."""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    y0 = tl.program_id(1) * ROWS
    last = (H - 1) * src_stride                      # offset of the last row
    h0 = blur_row(src_ptr, clamp((y0 - 2) * src_stride, 0, last), cols, W)
    h1 = blur_row(src_ptr, clamp((y0 - 1) * src_stride, 0, last), cols, W)
    h2 = blur_row(src_ptr, tl.minimum(y0 * src_stride, last), cols, W)
    h3 = blur_row(src_ptr, tl.minimum((y0 + 1) * src_stride, last), cols, W)
    src_off = (y0 + 2) * src_stride                  # row y + 2
    dst_off = y0 * dst_stride
    for y in range(y0, tl.minimum(y0 + ROWS, H)):
        h4 = blur_row(src_ptr, tl.minimum(src_off, last), cols, W)
        out = (h0 + h1 * 4 + h2 * 6 + h3 * 4 + h4) * (1.0 / 256)
        tl.store(dst_ptr + dst_off + cols, out, mask=cols < W)
        h0, h1, h2, h3 = h1, h2, h3, h4
        src_off += src_stride
        dst_off += dst_stride


@triton.jit
def sobel_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                 BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, ROWS: tl.constexpr,
                 BORDER: tl.constexpr):
    """|grad| = sqrt(gx^2 + gy^2); gx = [-1 0 1] across, smoothed [1 2 1] down."""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    for r in range(0, ROWS, BLOCK_H):
        rows = tl.program_id(1) * ROWS + r + tl.arange(0, BLOCK_H)
        gx = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        gy = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                wx = (1, 2, 1)[i] * (-1, 0, 1)[j]
                wy = (-1, 0, 1)[i] * (1, 2, 1)[j]
                if wx != 0 or wy != 0:               # the centre has weight 0 in both
                    x = tap(src_ptr, rows + (i - 1), cols + (j - 1), H, W, src_stride, BORDER)
                    if wx != 0:
                        gx += x * wx
                    if wy != 0:
                        gy += x * wy
        mag = tl.sqrt(gx * gx + gy * gy)
        mask = (rows[:, None] < H) & (cols[None, :] < W)
        tl.store(dst_ptr + rows[:, None] * dst_stride + cols[None, :], mag, mask=mask)


@triton.jit
def blur_sobel_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                      BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, ROWS: tl.constexpr):
    """Sobel magnitude of the 3x3-blurred image in one pass (clamped borders).

    [1 2 1] * [1 2 1] = [1 4 6 4 1] (smoothing) and [1 2 1] * [-1 0 1] =
    [-1 -2 0 2 1] (derivative), so gx = sum_i s_i sum_j d_j x[i, j] and gy
    the same with s and d swapped: a 5x5 stencil, 24 loads (the centre is 0).
    (With BORDER=ZERO's masks the 24 taps do not fit the registers.)"""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    for r in range(0, ROWS, BLOCK_H):
        rows = tl.program_id(1) * ROWS + r + tl.arange(0, BLOCK_H)
        gx = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        gy = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        for i in tl.static_range(5):
            for j in tl.static_range(5):
                wx = (1, 4, 6, 4, 1)[i] * (-1, -2, 0, 2, 1)[j]
                wy = (-1, -2, 0, 2, 1)[i] * (1, 4, 6, 4, 1)[j]
                if wx != 0 or wy != 0:
                    x = tap(src_ptr, rows + (i - 2), cols + (j - 2), H, W, src_stride, 0)
                    if wx != 0:
                        gx += x * wx
                    if wy != 0:
                        gy += x * wy
        gx = gx * (1.0 / 16)       # the blur's normalisation: exact
        gy = gy * (1.0 / 16)
        mag = tl.sqrt(gx * gx + gy * gy)
        mask = (rows[:, None] < H) & (cols[None, :] < W)
        tl.store(dst_ptr + rows[:, None] * dst_stride + cols[None, :], mag, mask=mask)


@triton.jit
def row_taps3(src_ptr, row_off, cols, W):
    """One clamped image row filtered across: ([1 2 1] smooth, [-1 0 1] derivative)."""
    left = tl.load(src_ptr + row_off + clamp(cols - 1, 0, W - 1))
    mid = tl.load(src_ptr + row_off + clamp(cols, 0, W - 1))
    right = tl.load(src_ptr + row_off + clamp(cols + 1, 0, W - 1))
    return left + mid * 2 + right, right - left


@triton.jit
def row_taps5(src_ptr, row_off, cols, W):
    """The same for the fused 5x5 stencil: [1 4 6 4 1] and [-1 -2 0 2 1]."""
    x0 = tl.load(src_ptr + row_off + clamp(cols - 2, 0, W - 1))
    x1 = tl.load(src_ptr + row_off + clamp(cols - 1, 0, W - 1))
    x2 = tl.load(src_ptr + row_off + clamp(cols, 0, W - 1))
    x3 = tl.load(src_ptr + row_off + clamp(cols + 1, 0, W - 1))
    x4 = tl.load(src_ptr + row_off + clamp(cols + 2, 0, W - 1))
    return x0 + x1 * 4 + x2 * 6 + x3 * 4 + x4, (x4 - x0) + (x3 - x1) * 2


@triton.jit
def sobel_rolling_kernel(src_ptr, dst_ptr, H, W, src_stride, dst_stride,
                         BLOCK_W: tl.constexpr, ROWS: tl.constexpr, FUSED: tl.constexpr):
    """sobel_kernel (FUSED=False) or blur_sobel_kernel (FUSED=True) with
    clamped borders, one image row per loop iteration as in
    blur_rolling_kernel: each row is loaded once (3 or 5 taps) and filtered
    across both ways; the vertical halves use the rows carried in registers."""
    cols = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    y0 = tl.program_id(1) * ROWS
    last = (H - 1) * src_stride
    dst_off = y0 * dst_stride
    if FUSED:
        s0, d0 = row_taps5(src_ptr, clamp((y0 - 2) * src_stride, 0, last), cols, W)
        s1, d1 = row_taps5(src_ptr, clamp((y0 - 1) * src_stride, 0, last), cols, W)
        s2, d2 = row_taps5(src_ptr, tl.minimum(y0 * src_stride, last), cols, W)
        s3, d3 = row_taps5(src_ptr, tl.minimum((y0 + 1) * src_stride, last), cols, W)
        src_off = (y0 + 2) * src_stride
        for y in range(y0, tl.minimum(y0 + ROWS, H)):
            s4, d4 = row_taps5(src_ptr, tl.minimum(src_off, last), cols, W)
            gx = ((d0 + d4) + (d1 + d3) * 4 + d2 * 6) * (1.0 / 16)
            gy = ((s4 - s0) + (s3 - s1) * 2) * (1.0 / 16)
            tl.store(dst_ptr + dst_off + cols, tl.sqrt(gx * gx + gy * gy), mask=cols < W)
            s0, s1, s2, s3 = s1, s2, s3, s4
            d0, d1, d2, d3 = d1, d2, d3, d4
            src_off += src_stride
            dst_off += dst_stride
    else:
        s0, d0 = row_taps3(src_ptr, clamp((y0 - 1) * src_stride, 0, last), cols, W)
        s1, d1 = row_taps3(src_ptr, tl.minimum(y0 * src_stride, last), cols, W)
        src_off = (y0 + 1) * src_stride
        for y in range(y0, tl.minimum(y0 + ROWS, H)):
            s2, d2 = row_taps3(src_ptr, tl.minimum(src_off, last), cols, W)
            gx = d0 + d1 * 2 + d2
            gy = s2 - s0
            tl.store(dst_ptr + dst_off + cols, tl.sqrt(gx * gx + gy * gy), mask=cols < W)
            s0, s1 = s1, s2
            d0, d1 = d1, d2
            src_off += src_stride
            dst_off += dst_stride

@triton.jit
def sqrt_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """tl.sqrt alone: the host model runs it on the emulator (qpu_sqrt)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.sqrt(tl.load(x_ptr + offs, mask=offs < n, other=0.0)), mask=offs < n)


# --------------------------------------------------------------------------
# Host side: the test scene, the filters as launches, numpy models, PNGs
# --------------------------------------------------------------------------

SIGNATURE = {"src_ptr": "*fp32", "dst_ptr": "*fp32", "H": "i32", "W": "i32",
             "src_stride": "i32", "dst_stride": "i32"}

# 5 x 7 glyphs, one row per entry, bit 4 = leftmost column. scene.h in
# pi/examples/image_filters holds the same table (tests compile it and compare).
FONT = {
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
}
TEXT = "PITON"


def scene_params(H: int, W: int) -> dict:
    """The scene's integer constants (computed here: the ARM has no divide)."""
    return dict(w=W, h=H, sx=(1024 << 10) // W, sy=(768 << 10) // H,
                gx=(120 << 16) // W, gy=(60 << 16) // H)


def make_scene(H: int, W: int) -> np.ndarray:
    """A synthetic grayscale test image: integers 0..255 as float32.

    Integer arithmetic only, so pi/examples/image_filters/scene.h generates the
    same pixels on the Pi: a gradient, a checkerboard, a disc, a ring, a
    striped triangle, the word PITON in a 5x7 font and +-8 of hash noise,
    drawn on a 1024 x 768 canvas scaled to W x H."""
    p = scene_params(H, W)
    y, x = np.mgrid[0:H, 0:W].astype(np.int64)
    u = (x * p["sx"]) >> 10
    v = (y * p["sy"]) >> 10
    img = 40 + ((x * p["gx"]) >> 16) + ((y * p["gy"]) >> 16)
    rect = (u >= 80) & (u < 440) & (v >= 520) & (v < 720)
    img = np.where(rect, np.where(((u >> 5) ^ (v >> 5)) & 1, 100, 30), img)
    img = np.where((u - 250) ** 2 + (v - 340) ** 2 < 130 ** 2, 215, img)
    r2 = (u - 780) ** 2 + (v - 260) ** 2
    img = np.where((r2 >= 90 ** 2) & (r2 < 130 ** 2), 20, img)
    # triangle (560, 720), (980, 720), (770, 440): three edge functions
    e1 = (u - 560) * (440 - 720) - (v - 720) * (770 - 560)
    e2 = (u - 770) * (720 - 440) - (v - 440) * (980 - 770)
    tri = (v <= 720) & (e1 <= 0) & (e2 <= 0)
    img = np.where(tri, np.where(((u + v) >> 4) & 1, 170, 230), img)
    fu, fv = (u - 50) >> 4, (v - 30) >> 4             # 16 canvas units per font pixel
    box = (u >= 50) & (v >= 30) & (fu < 8 * len(TEXT)) & (fv < 7) & ((fu & 7) < 5)
    glyphs = np.array([FONT[c] for c in TEXT], dtype=np.int64)
    bits = glyphs[np.clip(fu >> 3, 0, len(TEXT) - 1), np.clip(fv, 0, 6)] >> np.clip(4 - (fu & 7), 0, 4)
    img = np.where(box & ((bits & 1) == 1), 250, img)
    h = (y * W + x).astype(np.uint32) * np.uint32(0x9E3779B1)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13)
    img = img + (h & np.uint32(15)).astype(np.int64) - 8
    return np.clip(img, 0, 255).astype(np.float32)


# -- the filters ------------------------------------------------------------

# (BLOCK_H, BLOCK_W, ROWS) per filter, for a VGA image: every program fits the
# QPU's ~500-instruction cache (sobel, blur_sobel: their loops do) and the
# registers; ROWS = 48 gives 200-400 programs at 640 x 480.
TILES = {
    "blur3": (4, 16, 48),
    "blur3_zero": (2, 16, 48),
    "blur5": (2, 16, 48),
    "blur5_separable": (4, 16, 48),
    "blur5_rolling": (1, 32, 48),
    "sobel": (4, 16, 48),
    "sobel_rolling": (1, 32, 48),
    "blur_sobel": (2, 16, 48),
    "blur_sobel_rolling": (1, 16, 48),
}

# filter -> the image it computes; filters with the same output give the same bits
OUTPUTS = {
    "blur3": "blur3", "blur3_zero": "blur3_zero", "blur5": "blur5", "blur5_separable": "blur5",
    "blur5_rolling": "blur5", "sobel": "sobel", "sobel_rolling": "sobel",
    "blur_sobel": "blur_sobel", "blur_sobel_rolling": "blur_sobel",
}

# tl.load taps per output pixel
LOADS = {"blur3": 9, "blur3_zero": 9, "blur5": 25, "blur5_separable": 10, "blur5_rolling": 5,
         "sobel": 8, "sobel_rolling": 3, "blur_sobel": 24, "blur_sobel_rolling": 5}


def launches(name: str, tiles=None) -> list:
    """[(kernel, constants)] for filter `name`, run in order; the separable blur
    is two launches through a temporary image."""
    bh, bw, rows = (tiles or TILES)[name]
    tile = dict(BLOCK_H=bh, BLOCK_W=bw, ROWS=rows)
    if name in ("blur3", "blur5", "blur3_zero"):
        return [(blur_kernel, dict(tile, RADIUS=1 if name.startswith("blur3") else 2,
                                   BORDER=ZERO if name == "blur3_zero" else CLAMP))]
    if name == "blur5_separable":
        return [(blur_pass_kernel, dict(tile, RADIUS=2, BORDER=CLAMP, AXIS=axis)) for axis in (1, 0)]
    if name == "blur5_rolling":
        return [(blur_rolling_kernel, dict(BLOCK_W=bw, ROWS=rows))]
    if name == "sobel":
        return [(sobel_kernel, dict(tile, BORDER=CLAMP))]
    if name in ("sobel_rolling", "blur_sobel_rolling"):
        return [(sobel_rolling_kernel, dict(BLOCK_W=bw, ROWS=rows, FUSED=name == "blur_sobel_rolling"))]
    if name == "blur_sobel":
        return [(blur_sobel_kernel, tile)]
    raise ValueError(f"unknown filter {name!r}; expected one of {sorted(TILES)}")


def grid_for(H: int, W: int, constants: dict) -> tuple:
    return (triton.cdiv(W, constants["BLOCK_W"]), triton.cdiv(H, constants["ROWS"]))


def compile_filter(name: str, H: int, W: int, tiles=None) -> list:
    """[(CompiledKernel, constants)] for images up to H x W (the grid bound
    lets range analysis narrow the program-id arithmetic)."""
    from piton import compiler
    return [(compiler.compile(kernel, SIGNATURE, constants, grid=grid_for(H, W, constants)), constants)
            for kernel, constants in launches(name, tiles)]


def run_filter(name: str, img: np.ndarray, tiles=None, compiled=None,
               src_stride=None, dst_stride=None) -> np.ndarray:
    """Filter `img` on the strict emulator. Strides (in floats) may exceed W;
    the output starts as NaN, so a pixel the kernel misses shows up."""
    from piton.runtime.launcher import run_emulator
    H, W = img.shape
    compiled = compiled or compile_filter(name, H, W, tiles)
    src_stride, dst_stride = src_stride or W, dst_stride or W
    src = np.full(H * src_stride, np.nan, np.float32)
    src.reshape(H, src_stride)[:, :W] = img
    for i, (ck, constants) in enumerate(compiled):
        stride_out = dst_stride if i == len(compiled) - 1 else W
        dst = np.full(H * stride_out, np.nan, np.float32)
        run_emulator(ck, grid_for(H, W, constants),
                     dict(src_ptr=src, dst_ptr=dst, H=H, W=W, src_stride=src_stride, dst_stride=stride_out))
        src, src_stride = dst, stride_out
    return dst.reshape(H, dst_stride)[:, :W].copy()


# -- numpy models -------------------------------------------------------------

def shifted(img: np.ndarray, dy: int, dx: int, border: int = CLAMP) -> np.ndarray:
    """img[y + dy, x + dx] at every (y, x) in float64, clamped or 0 outside."""
    H, W = img.shape
    r, c = np.arange(H) + dy, np.arange(W) + dx
    out = img[np.clip(r, 0, H - 1)][:, np.clip(c, 0, W - 1)].astype(np.float64)
    if border == ZERO:
        out[(r < 0) | (r >= H), :] = 0
        out[:, (c < 0) | (c >= W)] = 0
    return out


def binomial(radius: int) -> np.ndarray:
    """[1 2 1] or [1 4 6 4 1]: the binomial approximation of a Gaussian."""
    w = np.ones(1)
    for _ in range(2 * radius):
        w = np.convolve(w, [1, 1])
    return w


def blur_reference(img: np.ndarray, radius: int, border: int = CLAMP) -> np.ndarray:
    """The binomial blur in float64. On integer images every partial sum is an
    integer below 2**24, so this is also every float32 result, in any order."""
    w = binomial(radius)
    acc = sum(w[i] * w[j] * shifted(img, i - radius, j - radius, border)
              for i in range(2 * radius + 1) for j in range(2 * radius + 1))
    return (acc / w.sum() ** 2).astype(np.float32)


def rtz32(x: np.ndarray) -> np.ndarray:
    """float64 -> float32 rounding toward zero, as the QPU's fadd and fmul round
    (exact for values a float64 holds exactly, as all of them are here)."""
    f = x.astype(np.float32)
    return np.where(np.abs(f.astype(np.float64)) > np.abs(x), np.nextafter(f, np.float32(0)), f)


def sobel_squared(img: np.ndarray, border: int = CLAMP, fused: bool = False) -> np.ndarray:
    """gx * gx + gy * gy as the QPU computes it (float32, toward zero): Sobel of
    img, or (fused) Sobel of its 3x3 blur, blur_sobel_kernel's 5x5 stencil."""
    if fused:
        s, d, r = (1, 4, 6, 4, 1), (-1, -2, 0, 2, 1), 2
    else:
        s, d, r = (1, 2, 1), (-1, 0, 1), 1
    taps = [(i, j) for i in range(2 * r + 1) for j in range(2 * r + 1)]
    gx = sum(s[i] * d[j] * shifted(img, i - r, j - r, border) for i, j in taps if s[i] * d[j])
    gy = sum(d[i] * s[j] * shifted(img, i - r, j - r, border) for i, j in taps if d[i] * s[j])
    if fused:
        gx, gy = gx / 16, gy / 16              # exact: both are integers below 2**24 first
    return rtz32(rtz32(gx * gx).astype(np.float64) + rtz32(gy * gy).astype(np.float64))


_SQRT = {}


def qpu_sqrt(x: np.ndarray) -> np.ndarray:
    """PIton's tl.sqrt, bit for bit: sqrt_kernel on the strict emulator, once per
    distinct value (a VGA Sobel image has ~10,000-55,000 of them)."""
    from piton import compiler
    from piton.runtime.launcher import run_emulator
    values, inverse = np.unique(np.asarray(x, np.float32).ravel(), return_inverse=True)
    todo = np.array([v for v in values.view(np.uint32) if int(v) not in _SQRT], np.uint32).view(np.float32)
    if todo.size:
        block = 1024
        grid = (triton.cdiv(todo.size, block),)
        ck = compiler.compile(sqrt_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
                              dict(BLOCK=block), grid=grid)
        out = np.zeros_like(todo)
        run_emulator(ck, grid, dict(x_ptr=todo, out_ptr=out, n=int(todo.size)))
        _SQRT.update(zip(todo.view(np.uint32).tolist(), out.view(np.uint32).tolist()))
    result = np.array([_SQRT[int(v)] for v in values.view(np.uint32)], np.uint32).view(np.float32)
    return result[inverse].reshape(np.shape(x))


def model(output: str, img: np.ndarray, sqrt=qpu_sqrt) -> np.ndarray:
    """The exact QPU result of every filter with this output (OUTPUTS), from
    numpy plus PIton's sqrt. sqrt=np.sqrt gives the correctly rounded
    magnitude instead (PIton's is within a few ulp of it)."""
    if output in ("blur3", "blur5", "blur3_zero"):
        return blur_reference(img, 1 if output.startswith("blur3") else 2,
                              ZERO if output == "blur3_zero" else CLAMP)
    if output in ("sobel", "blur_sobel"):
        squared = sobel_squared(img, CLAMP, fused=output == "blur_sobel")
        return np.asarray(sqrt(squared), np.float32)
    raise ValueError(f"unknown output {output!r}")


def ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in units in the last place (float32 values of the same sign)."""
    return np.abs(np.asarray(a, np.float32).view(np.int32).astype(np.int64)
                  - np.asarray(b, np.float32).view(np.int32).astype(np.int64))


# -- images ---------------------------------------------------------------

# |grad| of a 0..255 image reaches 4 * 255 * sqrt(2); edges of interest are far below
EDGE_SCALE = 255.0 / 400.0


def to_u8(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """img * scale rounded half up into 0..255 (also the firmware's --dump-images)."""
    return np.clip(np.floor(np.asarray(img, np.float64) * scale + 0.5), 0, 255).astype(np.uint8)


def write_png(path: str, gray: np.ndarray) -> None:
    """An 8-bit grayscale PNG, with zlib and struct only (no PIL)."""
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + gray[row].tobytes() for row in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def read_png(path: str) -> np.ndarray:
    """The inverse of write_png (8-bit grayscale, filter 0 rows only)."""
    with open(path, "rb") as f:
        data = f.read()
    pos, idat, size = 8, b"", None
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            size = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    w, h = size
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w + 1)
    if rows[:, 0].any():
        raise ValueError("read_png handles unfiltered rows only")
    return rows[:, 1:].copy()


def display(output: str, img: np.ndarray) -> np.ndarray:
    """8-bit view of a filter output: blurs as they are, edges scaled by EDGE_SCALE."""
    return to_u8(img, EDGE_SCALE if "sobel" in output else 1.0)


if __name__ == "__main__":
    H, W = 120, 160
    image = make_scene(H, W)
    os.makedirs("build", exist_ok=True)
    write_png("build/image_filters-input.png", to_u8(image))
    for filt in ("blur5_rolling", "sobel_rolling", "blur_sobel_rolling"):
        result = run_filter(filt, image)
        assert np.array_equal(result, model(OUTPUTS[filt], image)), filt
        write_png(f"build/image_filters-{filt}.png", display(OUTPUTS[filt], result))
        print(f"wrote build/image_filters-{filt}.png (emulator, {H}x{W}, matches the model bit for bit)")
