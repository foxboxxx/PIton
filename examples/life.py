"""Conway's Game of Life on a torus, as Triton kernels.

The grid is row-major with a row stride; row 0 and row n_rows - 1 are
neighbours, and so are column 0 and column n_cols - 1. One launch computes
one generation from `src` into `dst` (ping-pong buffers). The rule, counted
over the 3x3 block including the cell itself: a cell is alive next if the
block holds 3 live cells, or 4 and the cell is alive.

* `life_kernel`: the textbook stencil on an int32 grid of 0/1 cells. Each
  program loads the nine shifted copies of a BLOCK_R x BLOCK_C tile (torus
  wrap with `tl.where` on the row and column indices) and applies the rule
  with `tl.where`. Indices are clamped into the grid so the loads need no
  mask; only the store is masked.
* `life_rows_kernel`: the same rule with a sliding window. Each program walks
  ROWS rows of a BLOCK-wide column strip and carries the horizontal 3-cell
  sums of the two previous rows in loop-carried registers, so every row is
  loaded three times (left, centre, right) instead of nine.
* `life_bits_kernel`: 32 cells per uint32 word (bit j of word k is column
  32k + j; n_cols must be a multiple of 32). The 3x3 counts are added
  bit-sliced, as three bit planes, with and/or/xor and shifts: one 16-lane
  instruction works on 512 cells.

    .venv/bin/python examples/life.py            # a small soup in the emulator, all three kernels
    .venv/bin/python examples/life.py --images   # docs/images/life-*.{gif,png} from numpy
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402


@triton.jit
def life_kernel(src_ptr, dst_ptr, n_rows, n_cols, stride,
                BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    # A 1-D launch grid, tiles in row-major order: the compiler can then run
    # several consecutive tiles per hardware program (fewer, longer programs).
    pid = tl.program_id(0)
    col_blocks = tl.cdiv(n_cols, BLOCK_C)
    rows = (pid // col_blocks) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = (pid % col_blocks) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    out_ptrs = dst_ptr + rows[:, None] * stride + cols[None, :]
    # Cells past the edge of a ragged grid read (and drop) in-bounds neighbours.
    rows = tl.minimum(rows, n_rows - 1)
    cols = tl.minimum(cols, n_cols - 1)
    mid = rows * stride
    up = tl.where(rows == 0, (n_rows - 1) * stride, mid - stride)[:, None]
    down = tl.where(rows == n_rows - 1, 0, mid + stride)[:, None]
    mid = mid[:, None]
    left = tl.where(cols == 0, n_cols - 1, cols - 1)[None, :]
    right = tl.where(cols == n_cols - 1, 0, cols + 1)[None, :]
    centre = cols[None, :]

    alive = tl.load(src_ptr + mid + centre)
    total = alive
    total += tl.load(src_ptr + up + left)
    total += tl.load(src_ptr + up + centre)
    total += tl.load(src_ptr + up + right)
    total += tl.load(src_ptr + mid + left)
    total += tl.load(src_ptr + mid + right)
    total += tl.load(src_ptr + down + left)
    total += tl.load(src_ptr + down + centre)
    total += tl.load(src_ptr + down + right)
    new = tl.where((total == 3) | ((total == 4) & (alive != 0)), 1, 0)
    tl.store(out_ptrs, new, mask=mask)


@triton.jit
def life_rows_kernel(src_ptr, dst_ptr, n_rows, n_cols, stride,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    centre = tl.minimum(cols, n_cols - 1)
    left = tl.where(centre == 0, n_cols - 1, centre - 1)
    right = tl.where(centre == n_cols - 1, 0, centre + 1)
    first = tl.program_id(1) * ROWS
    last = min(first + ROWS, n_rows)

    # horizontal 3-cell sums of the row above the strip and of its first row
    row = src_ptr + tl.where(first == 0, n_rows - 1, first - 1) * stride
    h_up = tl.load(row + left) + tl.load(row + centre) + tl.load(row + right)
    row = src_ptr + first * stride
    alive = tl.load(row + centre)
    h_mid = tl.load(row + left) + alive + tl.load(row + right)
    out = dst_ptr + first * stride + cols
    for r in range(first, last):
        row += stride                                   # the row below r
        row = tl.where(r == n_rows - 1, src_ptr, row)   # wraps to row 0
        below = tl.load(row + centre)
        h_down = tl.load(row + left) + below + tl.load(row + right)
        total = h_up + h_mid + h_down
        new = tl.where((total == 3) | ((total == 4) & (alive != 0)), 1, 0)
        tl.store(out, new, mask=mask)
        out += stride
        h_up, h_mid, alive = h_mid, h_down, below


@triton.jit
def row_bit_sums(left_word, word, right_word):
    """Bit planes (s0, s1) of the horizontal 3-cell sums of a word's 32 cells."""
    west = (word << 1) | (left_word >> 31)      # bit j: the cell left of column j
    east = (word >> 1) | (right_word << 31)     # bit j: the cell right of column j
    x = west ^ word
    return x ^ east, (west & word) | (x & east)


@triton.jit
def life_bits_kernel(src_ptr, dst_ptr, n_rows, n_words, stride,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = words < n_words
    centre = tl.minimum(words, n_words - 1)
    left = tl.where(centre == 0, n_words - 1, centre - 1)
    right = tl.where(centre == n_words - 1, 0, centre + 1)
    first = tl.program_id(1) * ROWS
    last = min(first + ROWS, n_rows)

    row = src_ptr + tl.where(first == 0, n_rows - 1, first - 1) * stride
    a0, a1 = row_bit_sums(tl.load(row + left), tl.load(row + centre), tl.load(row + right))
    row = src_ptr + first * stride
    alive = tl.load(row + centre)
    b0, b1 = row_bit_sums(tl.load(row + left), alive, tl.load(row + right))
    out = dst_ptr + first * stride + words
    for r in range(first, last):
        row += stride
        row = tl.where(r == n_rows - 1, src_ptr, row)
        below = tl.load(row + centre)
        c0, c1 = row_bit_sums(tl.load(row + left), below, tl.load(row + right))
        # 3x3 total = (a1 a0) + (b1 b0) + (c1 c0) mod 8, as bit planes t2 t1 t0
        # (a total of 8 or 9 wraps to 0 or 1: dead either way)
        t0 = a0 ^ b0 ^ c0
        k0 = (a0 & b0) | (c0 & (a0 ^ b0))       # carry into the 2s
        s = a1 ^ b1
        s2 = s ^ c1
        t1 = s2 ^ k0
        t2 = (a1 & b1) ^ (s & c1) ^ (s2 & k0)
        new = (t0 & t1 & ~t2) | (alive & t2 & ~(t0 | t1))     # total 3, or 4 and alive
        tl.store(out, new, mask=mask)
        out += stride
        a0, a1, b0, b1, alive = b0, b1, c0, c1, below


SIGNATURE = {"src_ptr": "*i32", "dst_ptr": "*i32", "n_rows": "i32", "n_cols": "i32", "stride": "i32"}
BITS_SIGNATURE = {"src_ptr": "*u32", "dst_ptr": "*u32", "n_rows": "i32", "n_words": "i32", "stride": "i32"}

# name: (kernel, signature, default constants, cells per grid element along a row).
# The defaults are the Pi configurations (examples/life_pi.py): the largest
# tiles that fit the registers, chosen by emulated time at 512 x 512.
KERNELS = {
    "tile": (life_kernel, SIGNATURE, dict(BLOCK_R=4, BLOCK_C=16), 1),
    "rows": (life_rows_kernel, SIGNATURE, dict(ROWS=32, BLOCK=64), 1),
    "bits": (life_bits_kernel, BITS_SIGNATURE, dict(ROWS=16, BLOCK=16), 32),
}


def launch_grid(name: str, n_rows: int, n_cols: int, constants: dict) -> tuple:
    """The launch grid of one generation of kernel `name` on an n_rows x n_cols torus."""
    n = n_cols // KERNELS[name][3]
    if name == "tile":
        return (-(-n // constants["BLOCK_C"]) * -(-n_rows // constants["BLOCK_R"]),)
    return -(-n // constants["BLOCK"]), -(-n_rows // constants["ROWS"])


# ---------------------------------------------------------------------------
# host side: references, packing, soups, traces
# ---------------------------------------------------------------------------

LCG_A, LCG_C = 1664525, 1013904223
HASH_SALT = 0x9E3779B9


def lcg_u32(n: int, seed: int) -> np.ndarray:
    """The first n outputs of piton_lcg() (pi/runtime/piton_rt.h) from `seed`.

    s_k = a^k s_0 + c (1 + a + ... + a^(k-1)) mod 2^32 (uint64 products wrap
    mod 2^64, which 2^32 divides)."""
    with np.errstate(over="ignore"):
        powers = np.cumprod(np.full(n, LCG_A, dtype=np.uint64))
        geometric = np.cumsum(np.concatenate((np.ones(1, np.uint64), powers[:-1])))
        s = powers * np.uint64(seed & 0xFFFFFFFF) + geometric * np.uint64(LCG_C)
    return (s & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def soup(n_rows: int, n_cols: int, seed: int, eighths: int = 3) -> np.ndarray:
    """Random int32 0/1 grid, as the firmware makes it: cell i (row-major) is
    alive when the top three bits of the i-th LCG output are below `eighths`."""
    return ((lcg_u32(n_rows * n_cols, seed) >> np.uint32(29)) < eighths).astype(np.int32).reshape(n_rows, n_cols)


def pack_bits(grid: np.ndarray) -> np.ndarray:
    """int 0/1 [rows, cols] -> uint32 [rows, cols / 32]; bit j of word k is column 32k + j."""
    g = np.asarray(grid)
    rows, cols = g.shape
    if cols % 32:
        raise ValueError("the packed layout needs a multiple of 32 columns")
    bits = (g.reshape(rows, cols // 32, 32) != 0).astype(np.uint64) << np.arange(32, dtype=np.uint64)
    return bits.sum(axis=-1, dtype=np.uint64).astype(np.uint32)


def unpack_bits(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint32)
    bits = (w[..., None] >> np.arange(32, dtype=np.uint32)) & np.uint32(1)
    return bits.reshape(w.shape[0], -1).astype(np.int32)


def life_step(grid: np.ndarray) -> np.ndarray:
    """One generation on the torus (numpy reference, any shape)."""
    g = (np.asarray(grid) != 0).astype(np.int32)
    total = sum(np.roll(g, (dr, dc), axis=(0, 1)) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return ((total == 3) | ((total == 4) & (g == 1))).astype(np.int32)


def life_step_bits(words: np.ndarray) -> np.ndarray:
    """One generation on a packed torus, bit-sliced like life_bits_kernel (numpy)."""
    w = np.asarray(words, dtype=np.uint32)
    one, top = np.uint32(1), np.uint32(31)
    west = (w << one) | (np.roll(w, 1, axis=1) >> top)
    east = (w >> one) | (np.roll(w, -1, axis=1) << top)
    x = west ^ w
    s0, s1 = x ^ east, (west & w) | (x & east)
    a0, a1, c0, c1 = np.roll(s0, 1, 0), np.roll(s1, 1, 0), np.roll(s0, -1, 0), np.roll(s1, -1, 0)
    t0 = a0 ^ s0 ^ c0
    k0 = (a0 & s0) | (c0 & (a0 ^ s0))
    s = a1 ^ s1
    s2 = s ^ c1
    t1 = s2 ^ k0
    t2 = (a1 & s1) ^ (s & c1) ^ (s2 & k0)
    return (t0 & t1 & ~t2) | (w & t2 & ~(t0 | t1))


def grid_hash(words: np.ndarray) -> int:
    """The firmware's trace hash (pi/examples/life/life_arm.c): the sum, mod
    2^32, of murmur3's fmix32 over the packed words w_i salted with i * 0x9e3779b9."""
    x = np.asarray(words, dtype=np.uint32).ravel() ^ (np.arange(np.size(words), dtype=np.uint32)
                                                      * np.uint32(HASH_SALT))
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x85EBCA6B)
    x ^= x >> np.uint32(13)
    x *= np.uint32(0xC2B2AE35)
    x ^= x >> np.uint32(16)
    return int(x.sum(dtype=np.uint64) & np.uint64(0xFFFFFFFF))


def population(words: np.ndarray) -> int:
    return int(np.unpackbits(np.asarray(words, dtype=np.uint32).view(np.uint8)).sum())


def trace(words: np.ndarray, gens: int, keep=()) -> tuple:
    """[(population, hash)] for generations 0..gens of a packed soup, and the
    packed grids of the generations in `keep` ({gen: words})."""
    out, kept = [], {}
    w = np.asarray(words, dtype=np.uint32)
    for g in range(gens + 1):
        if g:
            w = life_step_bits(w)
        out.append((population(w), grid_hash(w)))
        if g in keep:
            kept[g] = w
    return out, kept


def run(name: str, grid: np.ndarray, gens: int = 1, constants=None, device: str = "emu") -> np.ndarray:
    """`gens` generations of `grid` (int 0/1, or packed uint32 for "bits") with
    kernel `name`, ping-ponging two buffers through `kernel[grid](...)`."""
    kernel, _, defaults, per = KERNELS[name]
    constants = dict(defaults, **(constants or {}))
    dtype = np.uint32 if per == 32 else np.int32
    src = np.array(grid, dtype=dtype)              # a copy: the launches write it
    dst = np.zeros_like(src)
    n_rows, n_elems = src.shape
    launch = launch_grid(name, n_rows, n_elems * per, constants)
    for _ in range(gens):
        kernel[launch](src, dst, n_rows, n_elems, n_elems, device=device, **constants)
        src, dst = dst, src
    return src


# ---------------------------------------------------------------------------
# images: indexed PNG and animated GIF, standard library only (no PIL here)
# ---------------------------------------------------------------------------

PALETTE = ((13, 17, 23), (126, 231, 135), (72, 79, 88))        # dead, alive, frame border


def upscale(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(image), factor, axis=0), factor, axis=1)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(image: np.ndarray, palette=PALETTE) -> bytes:
    """An 8-bit indexed-colour PNG of a 2-D array of palette indices."""
    img = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
            + _png_chunk(b"PLTE", bytes(c for rgb in palette for c in rgb))
            + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b""))


def _lzw(indices: bytes, min_size: int) -> bytes:
    """GIF LZW (variable code size up to 12 bits, clear when the table is full)."""
    clear, end = 1 << min_size, (1 << min_size) + 1
    out, acc, nbits = bytearray(), 0, 0

    def emit(code, size):
        nonlocal acc, nbits
        acc |= code << nbits
        nbits += size
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8

    size, next_code, table = min_size + 1, end + 1, {}
    emit(clear, size)
    prefix = indices[0]
    for k in indices[1:]:
        key = (prefix << 8) | k
        code = table.get(key)
        if code is not None:
            prefix = code
            continue
        emit(prefix, size)
        if next_code < 4096:
            table[key] = next_code
            next_code += 1
            if next_code > (1 << size) and size < 12:
                size += 1
        else:
            emit(clear, size)
            size, next_code, table = min_size + 1, end + 1, {}
        prefix = k
    emit(prefix, size)
    emit(end, size)
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def gif_bytes(frames, palette=PALETTE, delay_cs: int = 8) -> bytes:
    """A looping animated GIF of 2-D palette-index frames (same shape)."""
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    height, width = frames[0].shape
    bits = max(1, (len(palette) - 1).bit_length())
    table = list(palette) + [(0, 0, 0)] * ((1 << bits) - len(palette))
    out = bytearray(b"GIF89a" + struct.pack("<HHBBB", width, height, 0x80 | (bits - 1) << 4 | (bits - 1), 0, 0))
    out += bytes(c for rgb in table for c in rgb)
    out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"          # loop forever
    min_size = max(2, bits)
    for frame in frames:
        if frame.shape != (height, width):
            raise ValueError("every GIF frame needs the same shape")
        out += b"\x21\xf9\x04\x04" + struct.pack("<H", delay_cs) + b"\x00\x00"
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, width, height, 0) + bytes([min_size])
        data = _lzw(frame.tobytes(), min_size)
        for i in range(0, len(data), 255):
            out += bytes([len(data[i:i + 255])]) + data[i:i + 255]
        out += b"\x00"
    return bytes(out + b"\x3b")


def strip(frames, gap: int = 6) -> np.ndarray:
    """Frames side by side, `gap` border-coloured columns between them."""
    height = frames[0].shape[0]
    parts = []
    for i, frame in enumerate(frames):
        if i:
            parts.append(np.full((height, gap), 2, np.uint8))
        parts.append(np.asarray(frame, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def write_docs_images(out_dir: str, n_rows: int = 512, n_cols: int = 512, seed: int = 2026, gens: int = 256,
                      crop: int = 128, every: int = 2, strip_gens=(0, 1, 16, 64, 128, 256), check: int = 4) -> list:
    """The Pi's soup evolved in numpy: an animation of the top-left crop, a strip
    of crops at `strip_gens` and the whole grid at the last generation. The first
    `check` generations of the whole grid are also run in the emulator (bits)."""
    os.makedirs(out_dir, exist_ok=True)
    grid = soup(n_rows, n_cols, seed)
    words = pack_bits(grid)
    emulated = unpack_bits(run("bits", words, check)) if check else None
    frames, kept = [], {}
    for g in range(gens + 1):
        if g:
            grid = life_step(grid)
        if g == check and emulated is not None and not np.array_equal(grid, emulated):
            raise AssertionError(f"emulated life_bits_kernel differs from numpy after {check} generations")
        if g % every == 0:
            frames.append(upscale(grid[:crop, :crop], 2))
        if g in strip_gens:
            kept[g] = upscale(grid[:crop, :crop], 2)
    paths = [os.path.join(out_dir, name) for name in ("life-soup.gif", "life-strip.png", "life-final.png")]
    with open(paths[0], "wb") as f:
        f.write(gif_bytes(frames, delay_cs=6))
    with open(paths[1], "wb") as f:
        f.write(png_bytes(strip([kept[g] for g in strip_gens])))
    with open(paths[2], "wb") as f:
        f.write(png_bytes(grid))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", action="store_true", help="write the docs images from numpy")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    cli = parser.parse_args()
    if cli.images:
        for path in write_docs_images(cli.out):
            print("wrote", os.path.relpath(path))
        sys.exit(0)
    n_rows, n_cols, gens = 32, 64, 8
    start = soup(n_rows, n_cols, seed=2026)
    expected = start
    for _ in range(gens):
        expected = life_step(expected)
    for name in KERNELS:
        if name == "bits":
            got = unpack_bits(run(name, pack_bits(start), gens))
        else:
            got = run(name, start, gens)
        print(f"{name}: {gens} generations of a {n_rows}x{n_cols} soup in the emulator: "
              f"{'match' if np.array_equal(got, expected) else 'MISMATCH'} numpy "
              f"(population {int(start.sum())} -> {int(got.sum())})")
