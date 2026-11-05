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

    .venv/bin/python examples/life.py      # a small soup in the emulator, all three kernels
"""
import os
import sys

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
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619


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
    """FNV-1a over the packed words, row-major (the firmware's hash)."""
    h = FNV_OFFSET
    for v in np.asarray(words, dtype=np.uint32).ravel().tolist():
        h = ((h ^ v) * FNV_PRIME) & 0xFFFFFFFF
    return h


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


if __name__ == "__main__":
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
