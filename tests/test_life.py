"""Game of Life (examples/life.py): the three kernels in the strict emulator
against numpy, the firmware's C code natively, and the Pi driver's log checks
on a host simulation of the firmware (examples/life_pi.py --dry-run)."""
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from life import (KERNELS, gif_bytes, grid_hash, launch_grid, life_step, life_step_bits, pack_bits,  # noqa: E402
                  png_bytes, population, soup, unpack_bits)
from piton import compiler  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from piton.tools.compile import emit_c  # noqa: E402

_compiled = {}


def emulate(name, grid, gens=1, stride=None, **constants):
    """`gens` generations with kernel `name` in the strict emulator, ping-pong;
    `grid` is int 0/1 cells (packed words for "bits"). With `stride` the grid is
    the left part of wider rows, whose other entries must stay untouched."""
    kernel, signature, defaults, per = KERNELS[name]
    constants = dict(defaults, **constants)
    dtype = np.uint32 if per == 32 else np.int32
    rows, n = grid.shape
    stride = stride or n
    launch = launch_grid(name, rows, n * per, constants)
    key = (name, tuple(sorted(constants.items())), launch)
    if key not in _compiled:
        _compiled[key] = compiler.compile(kernel, signature, constants, grid=launch)
    ck = _compiled[key]
    sentinel = 0x5A5A5A5A
    bufs = [np.full((rows, stride), sentinel, dtype) for _ in range(2)]
    bufs[0][:, :n] = grid
    size_arg = "n_words" if per == 32 else "n_cols"
    for _ in range(gens):
        run_emulator(ck, launch, {"src_ptr": bufs[0], "dst_ptr": bufs[1], "n_rows": rows, size_arg: n,
                                  "stride": stride})
        assert np.all(bufs[1][:, n:] == dtype(sentinel)), "wrote outside the grid"
        bufs = bufs[::-1]
    return bufs[0][:, :n]


def numpy_gens(grid, gens):
    for _ in range(gens):
        grid = life_step(grid)
    return grid


def place(shape, cells, at=(0, 0)):
    grid = np.zeros(shape, np.int32)
    for r, c in cells:
        grid[(at[0] + r) % shape[0], (at[1] + c) % shape[1]] = 1
    return grid


GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
BLINKER = [(1, 0), (1, 1), (1, 2)]
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# numpy references
# ---------------------------------------------------------------------------

def test_reference_patterns():
    # a glider moves one cell down and right every four generations, across the wrap
    shape = (16, 32)
    start = place(shape, GLIDER, at=(14, 30))
    np.testing.assert_array_equal(numpy_gens(start, 4), place(shape, GLIDER, at=(15, 31)))
    np.testing.assert_array_equal(numpy_gens(start, 4 * 32), start)     # once around the 32-column torus
    blinker = place(shape, BLINKER, at=(5, 31))                          # straddles the column wrap
    assert not np.array_equal(life_step(blinker), blinker)
    np.testing.assert_array_equal(numpy_gens(blinker, 2), blinker)
    block = place(shape, BLOCK, at=(15, 31))                             # the four corners of the torus
    np.testing.assert_array_equal(life_step(block), block)


@pytest.mark.parametrize("shape", [(1, 32), (2, 64), (7, 96), (33, 32), (40, 160)])
def test_bit_sliced_reference_matches_cells(shape):
    grid = soup(*shape, seed=sum(shape))
    words = pack_bits(grid)
    np.testing.assert_array_equal(unpack_bits(words), grid)
    for _ in range(6):
        grid, words = life_step(grid), life_step_bits(words)
        np.testing.assert_array_equal(unpack_bits(words), grid)


def test_soup_and_hash():
    grid = soup(4, 64, seed=1)
    assert 0.2 < grid.mean() < 0.55 and set(np.unique(grid)) <= {0, 1}
    words = pack_bits(grid)
    assert population(words) == int(grid.sum())
    flipped = words.copy()
    flipped[3, 1] ^= np.uint32(1 << 7)
    assert grid_hash(flipped) != grid_hash(words)
    moved = np.roll(words, 1, axis=1)          # the same words elsewhere: a different hash
    assert grid_hash(moved) != grid_hash(words)


# ---------------------------------------------------------------------------
# kernels in the strict emulator
# ---------------------------------------------------------------------------

CELL_SHAPES = [(8, 16), (13, 37), (1, 20), (5, 3), (2, 2), (40, 70)]


@pytest.mark.parametrize("name", ["tile", "rows"])
@pytest.mark.parametrize("shape", CELL_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_cell_kernels_match_numpy(name, shape):
    grid = soup(*shape, seed=7 * shape[0] + shape[1])
    np.testing.assert_array_equal(emulate(name, grid, 3), numpy_gens(grid, 3))


@pytest.mark.parametrize("name,constants", [("tile", dict(BLOCK_R=2, BLOCK_C=32)), ("tile", dict(BLOCK_R=1, BLOCK_C=64)),
                                            ("rows", dict(ROWS=4, BLOCK=16)), ("rows", dict(ROWS=16, BLOCK=32))])
def test_other_tiles(name, constants):
    grid = soup(21, 50, seed=3)
    np.testing.assert_array_equal(emulate(name, grid, 2, **constants), numpy_gens(grid, 2))


@pytest.mark.parametrize("name", ["tile", "rows"])
def test_cell_kernels_with_row_stride(name):
    # the grid is the left 29 columns of 40-wide rows: the kernels must wrap at
    # column 29 and leave columns 29..39 alone (emulate checks the sentinel)
    grid = soup(10, 29, seed=11)
    np.testing.assert_array_equal(emulate(name, grid, 2, stride=40), numpy_gens(grid, 2))


@pytest.mark.parametrize("name", ["tile", "rows"])
def test_cell_kernels_glider_and_blinker_across_the_wrap(name):
    shape = (12, 20)
    start = place(shape, GLIDER, at=(10, 18)) | place(shape, BLINKER, at=(5, 19))
    np.testing.assert_array_equal(emulate(name, start, 4), numpy_gens(start, 4))


@pytest.mark.parametrize("shape,constants", [((1, 32), {}), ((7, 64), {}), ((33, 96), dict(ROWS=4)),
                                             ((18, 544), {}), ((9, 128), dict(ROWS=8, BLOCK=32))],
                         ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v))
def test_bits_kernel_matches_numpy(shape, constants):
    grid = soup(*shape, seed=shape[0] + 5 * shape[1])
    got = emulate("bits", pack_bits(grid), 3, **constants)
    np.testing.assert_array_equal(unpack_bits(got), numpy_gens(grid, 3))


def test_bits_kernel_with_row_stride_and_wrapping_patterns():
    shape = (16, 64)
    start = place(shape, GLIDER, at=(14, 62)) | place(shape, BLINKER, at=(3, 31)) | place(shape, BLOCK, at=(9, 63))
    got = emulate("bits", pack_bits(start), 5, stride=3)
    np.testing.assert_array_equal(unpack_bits(got), numpy_gens(start, 5))


def test_kernels_agree_over_many_generations():
    grid = soup(24, 64, seed=2026)
    want = numpy_gens(grid, 12)
    np.testing.assert_array_equal(emulate("tile", grid, 12), want)
    np.testing.assert_array_equal(emulate("rows", grid, 12), want)
    np.testing.assert_array_equal(unpack_bits(emulate("bits", pack_bits(grid), 12)), want)


def test_kernels_compile_at_pi_sizes():
    """The firmware configurations (examples/life_pi.py defaults) compile, fit
    the ~500-instruction instruction cache and emit a C launcher."""
    for name, size in (("tile", 512), ("rows", 512), ("bits", 512), ("bits", 2048)):
        kernel, signature, constants, _ = KERNELS[name]
        grid = launch_grid(name, size, size, constants)
        ck = compiler.compile(kernel, signature, constants, grid=grid)
        assert len(ck.words) < 480, (name, len(ck.words))
        header, source = emit_c(ck, f"life_{name}")
        assert f"life_{name}_launch" in header and "src_ptr" in header


# ---------------------------------------------------------------------------
# firmware C code, natively
# ---------------------------------------------------------------------------

CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include "life_arm.h"
int main(int argc, char **argv) {
    unsigned rows = atoi(argv[1]), cols = atoi(argv[2]), gens = atoi(argv[3]), seed = atoi(argv[4]);
    int32_t *c[2] = {calloc(rows * cols, 4), calloc(rows * cols, 4)}, *r[2] = {calloc(rows * cols, 4), calloc(rows * cols, 4)};
    int32_t *scratch = calloc(cols, 4);
    uint32_t *w[2] = {calloc(rows * cols / 32 + 1, 4), calloc(rows * cols / 32 + 1, 4)}, *ws = calloc(6 * cols / 32 + 6, 4);
    int words = cols % 32 == 0;
    life_soup_cells(c[0], rows, cols, seed, 3);
    life_soup_cells(r[0], rows, cols, seed, 3);
    if (words) life_soup_words(w[0], rows, cols, seed, 3);
    for (unsigned g = 0; g <= gens; g++) {
        unsigned k = g & 1;
        if (g) {
            life_arm_naive(c[k ^ 1], c[k], rows, cols);
            life_arm_rows(r[k ^ 1], r[k], rows, cols, scratch);
            if (words) life_arm_bits(w[k ^ 1], w[k], rows, cols, ws);
        }
        for (unsigned i = 0; i < rows * cols; i++)
            if (c[k][i] != r[k][i]) { printf("ROWS_DIFFER %u %u\n", g, i); return 1; }
        if (words) {
            life_trace_t a = life_trace_cells(c[k], rows, cols), b = life_trace_words(w[k], rows, cols);
            if (life_diff_cells(c[k], w[k], rows, cols)) { printf("BITS_DIFFER %u\n", g); return 1; }
            printf("%u %u %u %u %u\n", g, a.pop, a.hash, a.bad, b.pop == a.pop && b.hash == a.hash);
        }
        for (unsigned i = 0; i < cols; i++) printf("%d", c[k][i]);   // the first row, any width
        printf("\n");
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def c_driver(tmp_path_factory):
    if CC is None:
        pytest.skip("no host C compiler")
    out = tmp_path_factory.mktemp("life_c")
    (out / "driver.c").write_text(DRIVER)
    exe = out / "driver"
    subprocess.run([CC, "-O2", "-std=gnu99", "-Wall", "-Werror", f"-I{ROOT / 'pi/examples/life'}",
                    str(out / "driver.c"), str(ROOT / "pi/examples/life/life_arm.c"), "-o", str(exe)],
                   check=True, capture_output=True)
    return exe


@pytest.mark.parametrize("shape", [(7, 64), (1, 32), (33, 96), (5, 3), (2, 1), (12, 45)],
                         ids=lambda s: f"{s[0]}x{s[1]}")
def test_firmware_c_matches_numpy(c_driver, shape):
    rows, cols = shape
    gens, seed = 8, 99 + rows
    out = subprocess.run([str(c_driver), str(rows), str(cols), str(gens), str(seed)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    grid = soup(rows, cols, seed)
    lines = iter(out)
    for g in range(gens + 1):
        if g:
            grid = life_step(grid)
        if cols % 32 == 0:
            words = pack_bits(grid)
            assert next(lines).split() == [str(g), str(population(words)), str(grid_hash(words)), "0", "1"]
        assert next(lines) == "".join(map(str, grid[0]))


# ---------------------------------------------------------------------------
# examples/life_pi.py: the firmware's log, from a host simulation
# ---------------------------------------------------------------------------

def small_args(*extra):
    import life_pi
    return life_pi.parser().parse_args(["--size", "64", "--gens", "16", "--timed-gens", "6", "--repeats", "2",
                                        "--snap-every", "4", "--crop", "32", "--big", "96", "--big-gens", "5",
                                        "--big-timed-gens", "3", "--tile", "2x16", "--rows-tile", "16x32",
                                        "--bits-tile", "8x16", *extra])


@pytest.fixture(scope="module")
def host_run(tmp_path_factory):
    if CC is None:
        pytest.skip("no host C compiler")
    import life_pi
    args = small_args()
    build_dir = tmp_path_factory.mktemp("life_host")
    life_pi.generate(args, build_dir)
    log = life_pi.host_simulation(build_dir)
    return args, life_pi.expectations(args), log


def test_host_simulation_passes_every_check(host_run, tmp_path):
    import life_pi
    args, expected, log = host_run
    result = life_pi.parse_log(log, args, expected, host=True)
    assert result["snapshot_generations"] == list(range(0, 17, 4))
    assert {(t["implementation"], t["size"]) for t in result["timings"]} == \
        {(i, 64) for i in life_pi.SMALL_IMPLS} | {(i, 96) for i in life_pi.BIG_IMPLS}
    assert all(row["trace_mismatch"] == 0 for row in result["validation"])
    paths = life_pi.write_images(result, args, tmp_path, "life-test")
    assert decode_png(paths[2].read_bytes()).shape == (64, 64)
    frames = decode_gif(paths[0].read_bytes())
    assert len(frames) == 5 and frames[0].shape == (64, 64)
    np.testing.assert_array_equal(frames[0][::2, ::2], soup(64, 64, args.seed)[:32, :32])


def _tamper(log, needle, old, new):
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if needle in line and old in line:
            lines[i] = line.replace(old, new, 1)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"no line with {needle!r} and {old!r}")


def test_log_checks_reject_bad_runs(host_run):
    import life_pi
    args, expected, log = host_run
    first_trace_line = log.split("LIFE_TRACE impl=qpu_rows size=64x64 gens=16\n")[1].splitlines()[0]
    pop = first_trace_line.split(":")[0]
    bad_logs = [
        log.replace("LIFE_CHECK_OK", "LIFE_CHECK_FAILED"),
        log.replace("PITON_DONE", ""),
        log.replace(f"LIFE_TRACE impl=qpu_rows size=64x64 gens=16\n{pop}:",
                    f"LIFE_TRACE impl=qpu_rows size=64x64 gens=16\n{int(pop) + 1}:"),
        _tamper(log, "LIFE_VALIDATE impl=qpu_bits size=96x96", "cell_mismatch=0", "cell_mismatch=2"),
        _tamper(log, "LIFE_TIME impl=qpu_tile", "us=", "us=0 x="),
        _tamper(log, "LIFE_TIMED_CHECK impl=qpu_bits size=64x64", "pop=", "pop=1"),
        _tamper(log, "LIFE_CONFIG", "gens=16", "gens=17"),
    ]
    snap = log.index("LIFE_SNAP gen=8\n") + len("LIFE_SNAP gen=8\n")
    bad_logs.append(log[:snap] + ("1" if log[snap] != "1" else "2") + log[snap + 1:])
    final = log.index("\n", log.index("LIFE_FINAL ")) + 1
    bad_logs.append(log[:final] + log[final + 4:])
    for bad in bad_logs:
        with pytest.raises(ValueError):
            life_pi.parse_log(bad, args, expected, host=True)


def test_decode_words_round_trip():
    import life_pi
    rng = np.random.default_rng(5)
    for n in (1, 2, 3, 7, 512):
        words = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
        data = words.astype("<u4").tobytes()
        chars = []
        for i in range(0, len(data), 3):
            chunk = data[i:i + 3]
            v = int.from_bytes(chunk, "little")
            chars += [life_pi.ALPHABET[(v >> (6 * j)) & 63] for j in range(len(chunk) + 1)]
        text = "".join(chars)
        lines = [text[i:i + 64] for i in range(0, len(text), 64)]
        np.testing.assert_array_equal(life_pi.decode_words(lines, n), words)
    assert not any(c.isupper() for c in life_pi.ALPHABET) and len(set(life_pi.ALPHABET)) == 64


# ---------------------------------------------------------------------------
# images: minimal decoders for the PNG / GIF writers
# ---------------------------------------------------------------------------

def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    assert all(raw[r * (width + 1)] == 0 for r in range(height))
    return np.frombuffer(b"".join(raw[r * (width + 1) + 1:(r + 1) * (width + 1)] for r in range(height)),
                         np.uint8).reshape(height, width)


def _lzw_decode(data, min_size):
    clear, end = 1 << min_size, (1 << min_size) + 1
    acc = nbits = pos = 0
    out, table, prev, size = bytearray(), None, None, min_size + 1
    while True:
        while nbits < size:
            acc |= data[pos] << nbits
            pos, nbits = pos + 1, nbits + 8
        code, acc, nbits = acc & ((1 << size) - 1), acc >> size, nbits - size
        if code == clear:
            table, prev, size = [bytes([i]) for i in range(clear)] + [b"", b""], None, min_size + 1
            continue
        if code == end:
            return bytes(out)
        if prev is None:
            entry = table[code]
        else:
            entry = table[code] if code < len(table) else prev + prev[:1]
            table.append(prev + entry[:1])
            if len(table) == (1 << size) and size < 12:
                size += 1
        out += entry
        prev = entry


def decode_gif(data):
    assert data[:6] == b"GIF89a"
    packed = data[10]
    pos, frames = 13 + (3 * (2 << (packed & 7)) if packed & 0x80 else 0), []
    while data[pos] != 0x3B:
        if data[pos] == 0x21:
            pos += 2
            while data[pos]:
                pos += data[pos] + 1
            pos += 1
            continue
        assert data[pos] == 0x2C
        width, height = struct.unpack("<HH", data[pos + 5:pos + 9])
        min_size, pos, body = data[pos + 10], pos + 11, bytearray()
        while data[pos]:
            body += data[pos + 1:pos + 1 + data[pos]]
            pos += data[pos] + 1
        pos += 1
        frames.append(np.frombuffer(_lzw_decode(bytes(body), min_size), np.uint8).reshape(height, width))
    return frames


def test_png_and_gif_round_trip():
    rng = np.random.default_rng(0)
    frames = [(rng.random((37, 53)) * 3).astype(np.uint8) for _ in range(3)]
    frames += [np.zeros((37, 53), np.uint8), np.ones((37, 53), np.uint8)]
    for group in (frames, [(rng.random((300, 300)) < 0.3).astype(np.uint8)]):   # the second resets the table
        got = decode_gif(gif_bytes(group))
        assert len(got) == len(group) and all(np.array_equal(a, b) for a, b in zip(got, group))
    np.testing.assert_array_equal(decode_png(png_bytes(frames[0])), frames[0])
