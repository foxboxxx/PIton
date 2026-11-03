"""Mandelbrot escape-time renderer as a Triton kernel.

Each program colours a column strip BLOCK pixels wide and ROWS rows tall, in
tiles of TILE_H x BLOCK pixels (taller tiles escape more uniformly). PIton has
no `break`, so iterations run in chunks of UNROLL and a tile skips the rest of
its chunks once every pixel in it has escaped.

Escaped pixels keep iterating and stop being counted. Once |z| > 2 (with
|c| <= 2.5, as in every view here) |z| only grows, and on the QPUs it ends at
inf and stays there: inf - inf gives the second operand and 0 * inf gives +0,
so no NaN appears and x*x + y*y never drops back to 4 or below.

PITON_MANDEL_FAST=1 (read when the kernel is compiled) builds `mandelbrot_fast`
instead, which gives the same counts with less work; see its docstring.

    .venv/bin/python examples/mandelbrot.py            # render a preview in the emulator
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402


def fast_level() -> int:
    """PITON_MANDEL_FAST: 0 (default) the plain kernel, 1 `mandelbrot_fast`.
    Called inside the kernels, so it is read when they are compiled."""
    value = os.environ.get("PITON_MANDEL_FAST", "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"PITON_MANDEL_FAST={value!r}: expected 0 or 1")
    return int(value)


# Lanes whose c lies this far inside the main cardioid or the period-2 bulb
# never escape (see known_inside).
CARDIOID_SHRINK = 0.999
BULB_RADIUS = 0.999 / 4
BULB_R2_BITS = int(np.float32(BULB_RADIUS * BULB_RADIUS).view(np.int32))


@triton.jit
def known_inside(cx, cy):
    """-1 where c is inside the main cardioid or the period-2 bulb, with a margin; else 0.

    Around its cusp (1/4, 0) the main cardioid is rho <= (1 - cos theta) / 2;
    shrunk toward the cusp by s = CARDIOID_SHRINK that is t = 2 rho^2 / s +
    (x - 1/4) <= rho, i.e. t < 0 or t^2 < rho^2. The period-2 bulb is
    |c + 1| < 1/4; the margins keep the multiplier of the attracting fixed
    point (2-cycle) below ~0.999 away from the cusp. The squares are
    non-negative floats, so they are compared as integers."""
    xq = cx - 0.25
    q = xq * xq + cy * cy
    t = q * (2.0 / CARDIOID_SHRINK) + xq
    cardioid = (t.to(tl.int32, bitcast=True) | ((t * t).to(tl.int32, bitcast=True) - q.to(tl.int32, bitcast=True))) >> 31
    xb = cx + 1.0
    bulb = ((xb * xb + cy * cy).to(tl.int32, bitcast=True) - BULB_R2_BITS) >> 31
    return cardioid | bulb


@triton.jit
def fast_rows(out_ptr, palette_ptr, col, x0, y0, step, width, pitch, max_iter,
              UNROLL: tl.constexpr, ROWS: tl.constexpr, TH: tl.constexpr, TW: tl.constexpr):
    """mandelbrot_fast for the columns `col` (TW of them) of the program's ROWS rows."""
    cx = tl.broadcast_to((x0 + col.to(tl.float32) * step)[None, :], [TH, TW])
    for r in range(ROWS // TH):
        row = tl.program_id(1) * ROWS + r * TH + tl.arange(0, TH)
        cy = tl.broadcast_to((y0 + row.to(tl.float32) * step)[:, None], [TH, TW])
        x = tl.zeros([TH, TW], tl.float32)
        y = tl.zeros([TH, TW], tl.float32)
        if TH * TW <= 32:
            # INT_MIN: in the set, else 0. cy * 0.0 is +0 on the QPUs: it ties the
            # test to the row, so its parts are not hoisted and kept live
            misses = known_inside(cx + cy * 0.0, cy) << 31
        else:
            misses = tl.zeros([TH, TW], tl.int32)       # larger tiles would not fit the registers
        done = 0
        for chunk in range(max_iter // UNROLL):
            if tl.max(misses) >= 0:                     # a lane is still undecided
                for u in tl.static_range(UNROLL):
                    xx = x * x
                    yy = y * y
                    xy = x * y
                    x = xx - yy + cx
                    y = xy * 2.0 + cy
                    misses += (4.0 - (xx + yy)).to(tl.int32, bitcast=True) >> 31
                done += UNROLL
        # misses << 1 == 0: never escaped (0) or settled (INT_MIN)
        color = tl.load(palette_ptr + tl.where(misses << 1 == 0, 0, done + misses))
        tl.store(out_ptr + row[:, None] * pitch + col[None, :], color, mask=col[None, :] < width)


@triton.jit
def mandelbrot_fast(out_ptr, palette_ptr, x0, y0, step, width, pitch, max_iter,
                    BLOCK: tl.constexpr, UNROLL: tl.constexpr, ROWS: tl.constexpr,
                    TILE_H: tl.constexpr):
    """`mandelbrot_kernel` with the same counts and less work (PITON_MANDEL_FAST=1).

    A pixel's count does not depend on when its tile stops iterating, as long
    as the tile runs until the pixel escaped, so:
    - tiles are 2 x 16 pixels (two registers, a column block at a time) when
      BLOCK and ROWS allow, whatever TILE_H says: a compact tile waits less for
      its slowest pixel than a 1 x 32 one
    - lanes that known_inside() puts in the set never escape, so they do not
      keep a tile iterating, and a tile made only of them skips its iterations
    - misses is 0 exactly while a lane has not escaped, so the exit check reads
      misses (escaped is not carried); settled lanes start at INT_MIN and stay
      there, which keeps them out of the check
    - the check `tl.max(misses) >= 0` compiles to one flag-setting compare and
      a branch on any lane (the compiler's `fast_branches`, same switch), and
      the loop exits at the first failed check instead of testing every
      remaining chunk."""
    TW: tl.constexpr = 16 if BLOCK % 16 == 0 and ROWS % 2 == 0 else BLOCK
    TH: tl.constexpr = 2 if BLOCK % 16 == 0 and ROWS % 2 == 0 else TILE_H
    if TW == BLOCK:
        fast_rows(out_ptr, palette_ptr, tl.program_id(0) * BLOCK + tl.arange(0, BLOCK), x0, y0, step,
                  width, pitch, max_iter, UNROLL, ROWS, TH, TW)
    else:
        for cb in range(BLOCK // TW):
            fast_rows(out_ptr, palette_ptr, tl.program_id(0) * BLOCK + cb * TW + tl.arange(0, TW), x0, y0, step,
                      width, pitch, max_iter, UNROLL, ROWS, TH, TW)


@triton.jit
def mandelbrot_kernel(out_ptr, palette_ptr, x0, y0, step, width, pitch, max_iter,
                      BLOCK: tl.constexpr, UNROLL: tl.constexpr, ROWS: tl.constexpr,
                      TILE_H: tl.constexpr):
    if fast_level() > 0:
        mandelbrot_fast(out_ptr, palette_ptr, x0, y0, step, width, pitch, max_iter, BLOCK, UNROLL, ROWS, TILE_H)
        return
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cx = tl.broadcast_to((x0 + col.to(tl.float32) * step)[None, :], [TILE_H, BLOCK])
    # ROWS must divide the image height and be a multiple of TILE_H.
    for r in range(ROWS // TILE_H):
        row = tl.program_id(1) * ROWS + r * TILE_H + tl.arange(0, TILE_H)
        cy = tl.broadcast_to((y0 + row.to(tl.float32) * step)[:, None], [TILE_H, BLOCK])
        x = tl.zeros([TILE_H, BLOCK], tl.float32)
        y = tl.zeros([TILE_H, BLOCK], tl.float32)
        escaped = tl.zeros([TILE_H, BLOCK], tl.int32)   # 0 while |z|^2 <= 4, else -1
        misses = tl.zeros([TILE_H, BLOCK], tl.int32)    # minus the iterations spent escaped
        done = 0
        # max_iter must be a multiple of UNROLL.
        for chunk in range(max_iter // UNROLL):
            if tl.max(escaped) == 0:
                for u in tl.static_range(UNROLL):
                    xx = x * x
                    yy = y * y
                    # The sign of 4 - |z|^2 (one subtract with an immediate 4.0)
                    # is the escape test, exactly |z|^2 > 4.
                    escaped = (4.0 - (xx + yy)).to(tl.int32, bitcast=True) >> 31
                    misses += escaped
                    xy = x * y
                    x = xx - yy + cx
                    y = xy * 2.0 + cy
                done += UNROLL
        # Escaped pixels count at least 1; palette[0] colours the set itself,
        # whatever max_iter is.
        color = tl.load(palette_ptr + tl.where(misses == 0, 0, done + misses))
        tl.store(out_ptr + row[:, None] * pitch + col[None, :], color, mask=col[None, :] < width)


SIGNATURE = {"out_ptr": "*u32", "palette_ptr": "*u32", "x0": "fp32", "y0": "fp32",
             "step": "fp32", "width": "i32", "pitch": "i32", "max_iter": "i32"}

# Seahorse Valley: the buttons zoom toward this point.
TARGET = (-0.743643887037151, 0.131825904205330)


def make_palette(max_iter: int) -> np.ndarray:
    """0x00RRGGBB colours for counts 1..max_iter; entry 0 (the set itself) is black."""
    t = np.arange(max_iter + 1, dtype=np.float64) / 48.0
    r = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.15))
    b = 0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.35))
    rgb = (np.stack([r, g, b], axis=1) * 255).astype(np.uint32)
    palette = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    palette[0] = 0
    return palette


def reference_counts(x0, y0, step, width, height, max_iter):
    """Escape counts with the kernel's semantics, in float64 (numpy)."""
    cx = x0 + np.arange(width) * step
    cy = y0 + np.arange(height) * step
    c = cx[None, :] + 1j * cy[:, None]
    z = np.zeros_like(c)
    count = np.zeros(c.shape, dtype=np.int32)
    for _ in range(max_iter):
        inside = z.real ** 2 + z.imag ** 2 <= 4.0
        z = np.where(inside, z * z + c, z)
        count += inside
    return count


def escape_counts(x0, y0, step, width, height, max_iter, block=32, unroll=8, tile_h=1, rows=None,
                  device="emu"):
    """Raw escape counts from the kernel (an identity palette)."""
    identity = np.arange(max_iter + 1, dtype=np.uint32)
    identity[0] = max_iter      # the kernel looks up entry 0 for points in the set
    out = np.zeros(height * width, dtype=np.uint32)
    rows = rows or tile_h
    grid = (triton.cdiv(width, block), height // rows)
    mandelbrot_kernel[grid](out, identity, np.float32(x0), np.float32(y0), np.float32(step),
                            width, width, max_iter, BLOCK=block, UNROLL=unroll, ROWS=rows,
                            TILE_H=tile_h, device=device)
    return out.reshape(height, width)


def render(x0, y0, step, width, height, max_iter, block=32, unroll=8, tile_h=1, rows=None,
           device="emu"):
    palette = make_palette(max_iter)
    out = np.zeros(height * width, dtype=np.uint32)
    rows = rows or tile_h
    grid = (triton.cdiv(width, block), height // rows)
    mandelbrot_kernel[grid](out, palette, np.float32(x0), np.float32(y0), np.float32(step),
                            width, width, max_iter, BLOCK=block, UNROLL=unroll, ROWS=rows,
                            TILE_H=tile_h, device=device)
    return out.reshape(height, width), palette


def write_ppm(path, image):
    rgb = np.stack([(image >> 16) & 255, (image >> 8) & 255, image & 255], axis=-1).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (image.shape[1], image.shape[0]))
        f.write(rgb.tobytes())


if __name__ == "__main__":
    width, height, max_iter = 128, 96, 64
    view = 3.2
    step = view / width
    x0, y0 = -0.6 - view / 2, -step * height / 2
    image, _ = render(x0, y0, step, width, height, max_iter)
    os.makedirs("build", exist_ok=True)
    write_ppm("build/mandelbrot.ppm", image)
    print("wrote build/mandelbrot.ppm")
