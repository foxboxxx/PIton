"""Conway's Game of Life on the attached bare-metal Pi: three PIton kernels
against three ARM1176 implementations, in one boot (pi/examples/life).

    .venv/bin/python examples/life_pi.py               # build, boot, check, save results and images
    .venv/bin/python examples/life_pi.py --build-only  # compile the kernels and the firmware only
    .venv/bin/python examples/life_pi.py --dry-run     # build, then run the firmware natively with the
                                                       # ARM code standing in for the QPUs; check its log

On a 512 x 512 torus (a random soup, generated on the board):

* qpu_tile, qpu_rows, qpu_bits: examples/life.py life_kernel (the textbook
  9-load stencil, int32 cells), life_rows_kernel (sliding window, 3 loads per
  cell) and life_bits_kernel (bit-sliced, 32 cells per word).
* arm_naive, arm_rows, arm_bits: the same three algorithms in C on the ARM
  (pi/examples/life/life_arm.c, gcc -O3, MMU + L1 caches on): the honest
  baselines for each, and arm_bits is the fastest CPU code here.

Every implementation runs --gens generations; the population and a hash of
every generation are compared on the board with arm_naive's and here with
numpy, and the final grids cell by cell. Then --repeats timed runs of
--timed-gens generations each (ping-pong buffers, nothing else in the loop);
QPU time is the whole launch (uniforms, cache clears, queueing, waiting) per
generation, with the ARM caches off as in every PIton benchmark. The grid
after the timed runs is checked too. With --big N the bit-sliced kernel and
the bit-sliced ARM code also run on an N x N torus.

Results: benchmarks/results/life-pi-<date>.{json,log}; images from the
board's frames: docs/images/life-pi-{anim.gif,strip.png,final.png}.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from life import (KERNELS, gif_bytes, grid_hash, life_step_bits, pack_bits, png_bytes,  # noqa: E402
                  population, soup, strip, unpack_bits, upscale)
from piton import compiler  # noqa: E402
from piton.runtime.launcher import launch_coarsen  # noqa: E402
from piton.runtime.pi import PiError, PiRunner, libpi_root  # noqa: E402
from piton.tools.compile import emit_c  # noqa: E402

SMALL_IMPLS = ("arm_naive", "qpu_tile", "qpu_rows", "qpu_bits", "arm_rows", "arm_bits")
BIG_IMPLS = ("arm_bits", "qpu_bits")
QPU_VS_ARM = (("qpu_tile", "arm_naive"), ("qpu_rows", "arm_rows"), ("qpu_bits", "arm_bits"))
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz!#$%&()*+,-./:;<=>?@[]^_{|}~"   # main.c print_words
STRIP_GENS = (0, 4, 16, 64, 128, 256)
SOURCES = [ROOT / "pi/examples/life" / n for n in ("main.c", "Makefile", "life_arm.c", "life_arm.h")]
SOURCES += [ROOT / "pi/runtime" / n for n in ("piton_rt.c", "piton_rt.h")]
SOURCES += [ROOT / "pi/examples/matmul" / n for n in ("arm_sgemm.c", "arm_sgemm.h", "arm_kernel_vfp.S")]
HOST_SOURCES = [ROOT / "pi/examples/life/host" / n for n in ("rpi.h", "host_shim.c")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(text: str) -> tuple[int, int]:
    a, b = (int(v) for v in text.lower().split("x"))
    return a, b


def kernel_constants(args) -> dict:
    (tile_r, tile_c), (rows_r, rows_b), (bits_r, bits_b) = args.tile, args.rows_tile, args.bits_tile
    return {"tile": dict(BLOCK_R=tile_r, BLOCK_C=tile_c), "rows": dict(ROWS=rows_r, BLOCK=rows_b),
            "bits": dict(ROWS=bits_r, BLOCK=bits_b)}


def launch_grid(name: str, rows: int, cols: int, constants: dict) -> tuple[int, int]:
    """(gx, gy) of one generation (examples/life.py launch_grid, padded to 2-D)."""
    from life import launch_grid as grid_of
    grid = grid_of(name, rows, cols, constants)
    return (grid[0], grid[1] if len(grid) > 1 else 1)


def check_args(args) -> None:
    if args.size % 32 or args.size < 32:
        raise SystemExit("--size must be a positive multiple of 32 (the packed layout)")
    if args.crop % 32 or args.crop > args.size:
        raise SystemExit("--crop must be a multiple of 32 and at most --size")
    if min(args.gens, args.timed_gens, args.repeats, args.snap_every) < 1 or args.gens % args.snap_every:
        raise SystemExit("--gens, --timed-gens, --repeats, --snap-every must be positive; --snap-every must divide --gens")
    if args.big and (args.big % 32 or args.big == args.size or min(args.big_gens, args.big_timed_gens) < 1):
        raise SystemExit("--big must be a multiple of 32 other than --size, and --big-gens, "
                         "--big-timed-gens positive")


def generate(args, build_dir: Path) -> dict:
    """Compile the kernels and write the firmware sources into build_dir:
    life_{tile,rows,bits}.{c,h,qasm} (emit_c), life_config.h and copies of
    pi/examples/life, the runtime and the ARM cache code."""
    build_dir.mkdir(parents=True, exist_ok=True)
    constants = kernel_constants(args)
    grids = {name: launch_grid(name, args.size, args.size, constants[name]) for name in KERNELS}
    big_grid = launch_grid("bits", args.big, args.big, constants["bits"]) if args.big else (0, 0)
    kernels = {}
    for name, (kernel, signature, _, _) in KERNELS.items():
        grid = grids[name]
        if name == "bits" and args.big:          # one build serves both tori
            grid = (max(grid[0], big_grid[0]), max(grid[1], big_grid[1]))
        started = time.monotonic()
        ck = compiler.compile(kernel, signature, constants[name], grid=grid if grid[1] > 1 else grid[:1])
        header, source = emit_c(ck, f"life_{name}")
        (build_dir / f"life_{name}.h").write_text(header)
        (build_dir / f"life_{name}.c").write_text(source)
        (build_dir / f"life_{name}.qasm").write_text(ck.asm["qasm"])
        launch = grids[name] if grids[name][1] > 1 else grids[name][:1]
        kernels[name] = dict(kernel=kernel.fn.__name__, constants=constants[name], instructions=len(ck.words),
                             launch_grid=list(grids[name]), ids_per_program=launch_coarsen(ck, launch),
                             compile_seconds=round(time.monotonic() - started, 2),
                             code_sha256=hashlib.sha256(b"".join(int(w).to_bytes(8, "little")
                                                                 for w in ck.words)).hexdigest())
        print(f"Kernel {name}: {kernel.fn.__name__} {constants[name]}, {len(ck.words)} QPU instructions, "
              f"grid {grids[name]}", flush=True)
    config = ["#pragma once",
              f"#define LIFE_H {args.size}", f"#define LIFE_W {args.size}",
              f"#define LIFE_SEED {args.seed}u", f"#define LIFE_EIGHTHS {args.eighths}",
              f"#define LIFE_GENS {args.gens}", f"#define LIFE_TIMED_GENS {args.timed_gens}",
              f"#define LIFE_REPEATS {args.repeats}", f"#define LIFE_SNAP_EVERY {args.snap_every}",
              f"#define LIFE_CROP {args.crop}"]
    for name in KERNELS:
        config += [f"#define LIFE_{name.upper()}_GX {grids[name][0]}", f"#define LIFE_{name.upper()}_GY {grids[name][1]}"]
    config += [f"#define LIFE_BIG_H {args.big}", f"#define LIFE_BIG_W {args.big}",
               f"#define LIFE_BIG_SEED {args.big_seed}u", f"#define LIFE_BIG_GENS {args.big_gens}",
               f"#define LIFE_BIG_TIMED_GENS {args.big_timed_gens}",
               f"#define LIFE_BIG_GX {big_grid[0]}", f"#define LIFE_BIG_GY {big_grid[1]}"]
    (build_dir / "life_config.h").write_text("\n".join(config) + "\n")
    for source in SOURCES:
        shutil.copyfile(source, build_dir / source.name)
    return dict(size=args.size, seed=args.seed, eighths=args.eighths, gens=args.gens,
                timed_gens=args.timed_gens, repeats=args.repeats, snap_every=args.snap_every,
                crop=args.crop, big=args.big, big_seed=args.big_seed, big_gens=args.big_gens,
                big_timed_gens=args.big_timed_gens, big_grid=list(big_grid), kernels=kernels,
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                               SOURCES + [ROOT / "examples/life.py", ROOT / "examples/life_pi.py"]})


def build(args, build_dir: Path) -> tuple[Path, dict]:
    """generate(), then build main.bin with the course libpi (no boot)."""
    manifest = generate(args, build_dir)
    course_root = libpi_root()
    if course_root is None:
        raise SystemExit("Set CS240LX_2025_PATH to the course repository containing libpi "
                         "(needed to build the firmware)")
    env = {**os.environ, "CS240LX_2025_PATH": course_root}
    built = subprocess.run(["make", "-s", "RUN=0"], cwd=build_dir, env=env, capture_output=True, text=True)
    (build_dir / "build.log").write_text(built.stdout + built.stderr)
    binary = build_dir / "main.bin"
    if built.returncode or not binary.exists():
        raise SystemExit(f"Firmware build failed (see {build_dir / 'build.log'}):\n{built.stdout}\n{built.stderr}")
    manifest.update(firmware_bytes=binary.stat().st_size, firmware_sha256=sha256(binary))
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {binary.stat().st_size:,}-byte Pi firmware: {binary}", flush=True)
    return binary, manifest


def expectations(args) -> dict:
    """numpy traces for every generation the firmware reports, and the grids it dumps."""
    snap_gens = range(0, args.gens + 1, args.snap_every)
    timed = args.repeats * args.timed_gens
    small, kept = trace_to(pack_bits(soup(args.size, args.size, args.seed, args.eighths)),
                           max(args.gens, timed), set(snap_gens) | {args.gens})
    out = dict(small=small, small_timed=small[timed], final=kept[args.gens],
               crops={g: unpack_bits(kept[g])[:args.crop, :args.crop] for g in snap_gens})
    if args.big:
        big_timed = args.repeats * args.big_timed_gens
        big, _ = trace_to(pack_bits(soup(args.big, args.big, args.big_seed, args.eighths)),
                          max(args.big_gens, big_timed), ())
        out.update(big=big, big_timed=big[big_timed])
    return out


def trace_to(words: np.ndarray, gens: int, keep) -> tuple[list, dict]:
    out, kept = [], {}
    for g in range(gens + 1):
        if g:
            words = life_step_bits(words)
        out.append((population(words), grid_hash(words)))
        if g in keep:
            kept[g] = words
    return out, kept


def decode_words(lines: list[str], n_words: int) -> np.ndarray:
    """Inverse of main.c print_words: little-endian bytes, 3 per 4 characters."""
    index = {c: i for i, c in enumerate(ALPHABET)}
    text = "".join(line.strip() for line in lines)
    data = bytearray()
    for i in range(0, len(text), 4):
        group = text[i:i + 4]
        if len(group) == 1 or any(c not in index for c in group):
            raise ValueError("malformed word dump")
        v = sum(index[c] << (6 * j) for j, c in enumerate(group))
        data += v.to_bytes(3, "little")[:len(group) - 1]
    if len(data) != 4 * n_words:
        raise ValueError(f"word dump has {len(data)} bytes, expected {4 * n_words}")
    return np.frombuffer(bytes(data), dtype="<u4").astype(np.uint32)


def records(log: str, tag: str) -> list[dict]:
    out = []
    for m in re.finditer(r"^.*?" + tag + r" (.*)$", log, re.MULTILINE):
        out.append(dict(re.findall(r"(\w+)=(\S+)", m.group(1))))
    return out


def blocks(log: str, start: str, end: str) -> list[tuple[str, list[str]]]:
    """(header line, body lines) of every start ... end block."""
    out, header, body = [], None, []
    for line in log.splitlines():
        if header is None:
            if start + " " in line:
                header, body = line[line.index(start):], []
        elif line.strip() == end:
            out.append((header, body))
            header = None
        else:
            body.append(line)
    if header is not None:
        raise ValueError(f"unterminated {start} block")
    return out


def size_of(rec: dict) -> int:
    rows, cols = (int(v) for v in rec["size"].split("x"))
    if rows != cols:
        raise ValueError(f"unexpected size {rec['size']}")
    return rows


def parse_log(log: str, args, expected: dict, host: bool = False) -> dict:
    """Check everything the firmware printed against numpy; raise ValueError on
    any missing, malformed or wrong record."""
    if "LIFE_FATAL" in log or "LIFE_CHECK_FAILED" in log:
        raise ValueError("the firmware reported a failure:\n" + "\n".join(
            line for line in log.splitlines() if "LIFE_" in line and ("mismatch=" in line or "FATAL" in line)))
    if not re.search(r"^LIFE_CHECK_OK\s*$", log, re.MULTILINE) or not re.search(r"^PITON_DONE\s*$", log, re.MULTILINE):
        raise ValueError("the firmware did not finish (no LIFE_CHECK_OK / PITON_DONE)")
    cfg = records(log, "LIFE_CONFIG")
    want = dict(rows=args.size, cols=args.size, seed=args.seed, eighths=args.eighths, gens=args.gens,
                timed_gens=args.timed_gens, repeats=args.repeats, snap_every=args.snap_every, crop=args.crop,
                big_rows=args.big, big_cols=args.big, big_seed=args.big_seed, big_gens=args.big_gens,
                big_timed_gens=args.big_timed_gens)
    if len(cfg) != 1 or {k: int(cfg[0].get(k, -1)) for k in want} != want:
        raise ValueError(f"firmware configuration {cfg} does not match this build {want}")
    clocks = {r["name"]: int(r["mhz"]) for r in records(log, "LIFE_CLOCK")}
    if sorted(clocks) != ["arm", "core", "sdram", "v3d"] or min(clocks.values()) <= 0:
        raise ValueError(f"missing or invalid clocks {clocks}")
    setup = records(log, "LIFE_SETUP")
    if len(setup) != 1:
        raise ValueError("missing LIFE_SETUP")
    sizes = {args.size: (SMALL_IMPLS, expected["small"], args.gens)}
    if args.big:
        sizes[args.big] = (BIG_IMPLS, expected["big"], args.big_gens)
    wanted = sorted((impl, n) for n, (impls, _, _) in sizes.items() for impl in impls)

    traces = {}
    for header, body in blocks(log, "LIFE_TRACE", "LIFE_TRACE_END"):
        rec = dict(re.findall(r"(\w+)=(\S+)", header))
        key = (rec["impl"], size_of(rec))
        values = [(int(p), int(h, 16)) for p, h in re.findall(r"(\d+):(?:0x)?([0-9a-f]+)", " ".join(body))]
        if key not in [tuple(w) for w in wanted] or key in traces:
            raise ValueError(f"unexpected or repeated trace {key}")
        _, reference, gens = sizes[key[1]]
        if int(rec["gens"]) != gens or len(values) != gens + 1:
            raise ValueError(f"trace {key}: expected {gens + 1} generations, got {len(values)}")
        wrong = [g for g, (got, ref) in enumerate(zip(values, reference)) if got != ref]
        if wrong:
            raise ValueError(f"trace {key} differs from numpy at generations {wrong[:10]} "
                             f"(first: board {values[wrong[0]]}, numpy {reference[wrong[0]]})")
        traces[key] = values
    if sorted(traces) != wanted:
        raise ValueError(f"traces {sorted(traces)}, expected {wanted}")

    validation = []
    for rec in records(log, "LIFE_VALIDATE"):
        row = dict(implementation=rec["impl"], size=size_of(rec), generations=int(rec["gens"]),
                   trace_mismatch=int(rec["trace_mismatch"]), cell_mismatch=int(rec["cell_mismatch"]),
                   bad_cells=int(rec["bad"]), check_us=int(rec["us"]), reference="reference" in rec)
        if row["trace_mismatch"] or row["cell_mismatch"] or row["bad_cells"] or \
                row["generations"] != sizes[row["size"]][2]:
            raise ValueError(f"failed validation {row}")
        validation.append(row)
    if sorted((r["implementation"], r["size"]) for r in validation) != wanted:
        raise ValueError("expected one validation per implementation and size")

    timings, timed_checks = [], []
    for impl, n in wanted:
        gens = args.timed_gens if n == args.size else args.big_timed_gens
        rows = [r for r in records(log, "LIFE_TIME") if r["impl"] == impl and size_of(r) == n]
        if len(rows) != args.repeats or any(int(r["gens"]) != gens or int(r["us"]) <= 0 for r in rows):
            raise ValueError(f"expected {args.repeats} positive timings of {gens} generations for {impl} {n}x{n}")
        us = [int(r["us"]) for r in rows]
        median = statistics.median(us)
        timings.append(dict(implementation=impl, size=n, generations=gens, times_us=us,
                            launch_us=[int(r["launch_us"]) for r in rows], median_us=median, min_us=min(us),
                            us_per_generation=median / gens,
                            cell_updates_per_second=n * n * gens * 1e6 / median))
        checks = [r for r in records(log, "LIFE_TIMED_CHECK") if r["impl"] == impl and size_of(r) == n]
        ref = expected["small_timed"] if n == args.size else expected["big_timed"]
        if len(checks) != 1:
            raise ValueError(f"expected one timed check for {impl} {n}x{n}")
        c = checks[0]
        got = (int(c["pop"]), int(c["hash"], 16))
        if int(c["gens"]) != args.repeats * gens or got != ref or int(c["cell_mismatch"]) or int(c["bad"]):
            raise ValueError(f"timed run of {impl} {n}x{n}: {c}, numpy (pop, hash) {ref}")
        timed_checks.append(dict(implementation=impl, size=n, generations=args.repeats * gens,
                                 population=got[0], hash=f"{got[1]:08x}"))
    if len(records(log, "LIFE_TIME")) != len(wanted) * args.repeats:
        raise ValueError("unexpected LIFE_TIME records")

    snaps = blocks(log, "LIFE_SNAPS", "LIFE_SNAPS_END")
    if len(snaps) != 1:
        raise ValueError("expected one LIFE_SNAPS block")
    frames, current, body = {}, None, []
    crop_words = args.crop * args.crop // 32
    for line in snaps[0][1] + ["LIFE_SNAP gen=-1"]:
        m = re.search(r"LIFE_SNAP gen=(-?\d+)", line)
        if not m:
            body.append(line)
            continue
        if current is not None:
            frames[current] = unpack_bits(decode_words(body, crop_words).reshape(args.crop, args.crop // 32))
        current, body = int(m.group(1)), []
    if sorted(frames) != sorted(expected["crops"]):
        raise ValueError(f"snapshot generations {sorted(frames)}")
    wrong = [g for g in frames if not np.array_equal(frames[g], expected["crops"][g])]
    if wrong:
        raise ValueError(f"snapshots differ from numpy at generations {wrong[:10]}")
    final = blocks(log, "LIFE_FINAL", "LIFE_FINAL_END")
    if len(final) != 1:
        raise ValueError("expected one LIFE_FINAL block")
    final_words = decode_words(final[0][1], args.size * args.size // 32).reshape(args.size, args.size // 32)
    if not np.array_equal(final_words, expected["final"]):
        raise ValueError(f"the final grid differs from numpy in "
                         f"{int(np.unpackbits((final_words ^ expected['final']).view(np.uint8)).sum())} cells")

    by = {(t["implementation"], t["size"]): t for t in timings}
    speedups = {}
    for qpu, arm in QPU_VS_ARM:
        speedups[f"{qpu}_vs_{arm}"] = by[(arm, args.size)]["median_us"] / by[(qpu, args.size)]["median_us"]
        speedups[f"{qpu}_vs_best_arm"] = min(by[(a, args.size)]["median_us"] for a in
                                              ("arm_naive", "arm_rows", "arm_bits")) / by[(qpu, args.size)]["median_us"]
    if args.big:
        speedups["big_qpu_bits_vs_arm_bits"] = by[("arm_bits", args.big)]["median_us"] / by[("qpu_bits", args.big)]["median_us"]
    return dict(host_simulation=host, clocks_mhz=clocks, setup_us=int(setup[0]["us"]), validation=validation,
                timings=timings, timed_checks=timed_checks, speedups=speedups,
                snapshot_generations=sorted(frames), frames=frames, final=unpack_bits(final_words),
                traces={f"{impl}@{n}": [[p, f"{h:08x}"] for p, h in v] for (impl, n), v in traces.items()},
                timing_scope="Median of the repeats; each repeat runs timed_gens generations back to back "
                             "(ping-pong). QPU: every launch in full (uniform writes, GPU cache clears, "
                             "queueing, waiting), ARM caches off. ARM: gcc -O3, MMU and L1 caches on. "
                             "Soup generation, checks and printing are outside the timer.")


def write_images(result: dict, args, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = result["frames"]
    paths = [out_dir / f"{prefix}-anim.gif", out_dir / f"{prefix}-strip.png", out_dir / f"{prefix}-final.png"]
    paths[0].write_bytes(gif_bytes([upscale(frames[g], 2) for g in sorted(frames)], delay_cs=10))
    chosen = [g for g in STRIP_GENS if g in frames] or sorted(frames)[:6]
    paths[1].write_bytes(png_bytes(strip([upscale(frames[g], 2) for g in chosen])))
    paths[2].write_bytes(png_bytes(result["final"]))
    return paths


def summary(result: dict, args) -> str:
    lines = [f"{'implementation':10} {'size':>9} {'us/gen':>10} {'Mcell/s':>9}"]
    for t in result["timings"]:
        lines.append(f"{t['implementation']:10} {t['size']:>4}x{t['size']:<4} {t['us_per_generation']:10.1f} "
                     f"{t['cell_updates_per_second'] / 1e6:9.1f}")
    lines += [f"{k}: {v:.2f}x" for k, v in result["speedups"].items()]
    return "\n".join(lines)


def host_simulation(build_dir: Path) -> str:
    """Compile main.c natively (host/host_shim.c in place of libpi and the QPU
    launchers) and return its output."""
    host_dir = build_dir / "host"
    host_dir.mkdir(exist_ok=True)
    for source in HOST_SOURCES:
        shutil.copyfile(source, host_dir / source.name)
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        raise SystemExit("--dry-run needs a host C compiler (cc)")
    exe = host_dir / "life_host"
    built = subprocess.run([cc, "-O2", "-std=gnu99", "-DLIFE_HOST", f"-I{host_dir}", f"-I{build_dir}",
                            str(build_dir / "main.c"), str(build_dir / "life_arm.c"), str(host_dir / "host_shim.c"),
                            "-o", str(exe)], capture_output=True, text=True)
    if built.returncode:
        raise SystemExit(f"host build failed:\n{built.stdout}\n{built.stderr}")
    ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    if ran.returncode:
        raise SystemExit(f"host simulation failed ({ran.returncode}):\n{ran.stdout[-2000:]}\n{ran.stderr}")
    return ran.stdout


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=512, help="side of the torus (multiple of 32)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eighths", type=int, default=3, help="initial density in eighths")
    ap.add_argument("--gens", type=int, default=256, help="checked generations")
    ap.add_argument("--timed-gens", type=int, default=100, help="generations per timed repeat")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--snap-every", type=int, default=4, help="generations between animation frames")
    ap.add_argument("--crop", type=int, default=128, help="side of the animated crop (multiple of 32)")
    ap.add_argument("--big", type=int, default=2048, help="side of the bit-sliced torus (0: none)")
    ap.add_argument("--big-seed", type=int, default=2027)
    ap.add_argument("--big-gens", type=int, default=64)
    ap.add_argument("--big-timed-gens", type=int, default=64)
    ap.add_argument("--tile", type=pair, default="4x16", help="life_kernel BLOCK_RxBLOCK_C")
    ap.add_argument("--rows-tile", type=pair, default="32x64", help="life_rows_kernel ROWSxBLOCK")
    ap.add_argument("--bits-tile", type=pair, default="16x16", help="life_bits_kernel ROWSxBLOCK (words)")
    ap.add_argument("--build-only", action="store_true", help="compile the kernels and firmware, do not boot")
    ap.add_argument("--dry-run", action="store_true",
                    help="build, run the firmware natively (ARM code for the QPU) and check its log; no Pi")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build/life")
    ap.add_argument("--tty")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--verbose", action="store_true")
    # dated by the run, so a new measurement never overwrites the one the docs cite
    ap.add_argument("--output", type=Path,
                    default=ROOT / f"benchmarks/results/life-pi-{datetime.now():%Y-%m-%d}.json")
    ap.add_argument("--images-dir", type=Path, default=ROOT / "docs/images")
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    check_args(args)
    build_dir = args.build_dir.resolve()
    binary, manifest = build(args, build_dir)
    if args.build_only:
        return
    started = time.monotonic()
    expected = expectations(args)
    print(f"numpy reference: {time.monotonic() - started:.1f} s", flush=True)
    if args.dry_run:
        out_dir = build_dir / "dry-run"
        out_dir.mkdir(exist_ok=True)
        log = host_simulation(build_dir)
        (out_dir / "life-host.log").write_text(log)
        result = parse_log(log, args, expected, host=True)
        paths = write_images(result, args, out_dir, "life-host")
        print(summary(result, args))
        print(f"Dry run passed: the firmware's flow, checks and log format (QPU launches replaced by the "
              f"ARM code; host timings are meaningless). Log and images in {out_dir}:",
              ", ".join(p.name for p in paths), flush=True)
        return
    try:
        runner = PiRunner(tty=args.tty, timeout=args.timeout, verbose=args.verbose)
    except PiError as ex:
        raise SystemExit(f"No Pi found: {ex}. Use --dry-run or --build-only without one.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    print(f"Booting {binary} (about 2 minutes; most of it is the UART dump of the frames)...", flush=True)
    started = time.monotonic()
    try:
        log = runner._boot_recovering(str(binary))
    except PiError as ex:
        if ex.log:
            log_path.write_text(ex.log)
        raise SystemExit(f"Pi run failed (partial log in {log_path}): {ex}")
    seconds = time.monotonic() - started
    log_path.write_text(log)            # keep the raw log even if the checks reject it
    result = parse_log(log, args, expected)
    paths = write_images(result, args, args.images_dir, "life-pi")
    frames = result.pop("frames")
    result.pop("final")
    result.update(manifest=manifest, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  boot_and_run_seconds=round(seconds, 1), snapshot_count=len(frames),
                  raw_log=str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
                  images=[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(summary(result, args))
    print(f"Pi verified: every generation of every implementation matched numpy; saved {args.output}, "
          f"{log_path} and {', '.join(str(p) for p in paths)}", flush=True)


if __name__ == "__main__":
    main()
