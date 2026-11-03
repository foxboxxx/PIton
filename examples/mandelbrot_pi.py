"""Build and run the interactive Mandelbrot renderer on the attached bare-metal Pi.

    .venv/bin/python examples/mandelbrot_pi.py                 # build, boot, run (Ctrl-C to stop)
    .venv/bin/python examples/mandelbrot_pi.py --bench-only    # checks + benchmarks, then exit
    .venv/bin/python examples/mandelbrot_pi.py --profile       # per-frame time + V3D counters per zoom
    .venv/bin/python examples/mandelbrot_pi.py --build-only

Needs an HDMI monitor on the Pi. Control it from this terminal: i/o (or +/-)
zoom, w/a/s/d or the arrow keys pan, r resets, space starts the automatic tour,
q quits. Every compiled kernel configuration is checked against emulator counts
and benchmarked on the board; the fastest one runs the interactive loop.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import termios
import tty

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from mandelbrot import SIGNATURE, TARGET, escape_counts, make_palette, mandelbrot_kernel
from piton import compiler
from piton.runtime.pi import PiRunner, find_bootloader, find_tty, libpi_root
from piton.tools.compile import emit_c

FULL_VIEW = 3.5          # must match main.c
MIN_ITER, ITER_PER_OCTAVE = 64, 32
PATCH_W, PATCH_H, PATCH_ITERS, PATCH_ZOOM = 64, 8, 256, 100.0


def parse_configs(text: str) -> list[tuple[int, int, int]]:
    """BLOCKxUNROLL or BLOCKxUNROLLxTILE_H items."""
    configs = []
    for item in text.split(","):
        values = [int(v) for v in item.lower().split("x")]
        configs.append((values[0], values[1], values[2] if len(values) > 2 else 1))
    return configs


def config_name(block: int, unroll: int, tile_h: int) -> str:
    return f"{block}x{unroll}" + (f"x{tile_h}" if tile_h > 1 else "")


def c_float(value: float) -> str:
    return float(np.float32(value)).hex() + "f"


def c_u32_array(name: str, values: np.ndarray) -> str:
    strings = [f"0x{int(v):08x}" for v in np.asarray(values).ravel()]
    lines = [", ".join(strings[i:i + 8]) + "," for i in range(0, len(strings), 8)]
    return f"static const uint32_t {name}[{len(strings)}] = {{\n" + "\n".join(lines) + "\n};\n"


def kernels_header(configs: list[tuple[int, int, int]], width: int, height: int, rows: int) -> tuple[str, list[str]]:
    lines = ["#pragma once", "#include <stdint.h>"]
    names = [f"mandel_b{block}u{unroll}t{tile_h}" for block, unroll, tile_h in configs]
    lines += [f'#include "{name}.h"' for name in names]
    lines += ["", "typedef uint32_t (*mandel_run_t)(uint32_t out, uint32_t palette, float x0, float y0,",
              "    float step, int32_t width, int32_t pitch, int32_t max_iter, unsigned gx, unsigned gy);",
              "typedef struct { const char *name; unsigned block, unroll, tile_h, grid_x, grid_y, patch_grid_x, patch_grid_y;",
              "                 mandel_run_t run; } mandel_config_t;",
              ""]
    for name in names:
        lines += [f"static uint32_t {name}_run(uint32_t out, uint32_t palette, float x0, float y0, float step,",
                  "        int32_t width, int32_t pitch, int32_t max_iter, unsigned gx, unsigned gy) {",
                  f"    {name}_args_t a = {{ .out_ptr = out, .palette_ptr = palette, .x0 = x0, .y0 = y0,",
                  "        .step = step, .width = width, .pitch = pitch, .max_iter = max_iter };",
                  f"    return {name}_launch(&a, gx, gy, 1);", "}", ""]
    lines.append("static const mandel_config_t mandel_configs[] = {")
    for name, (block, unroll, tile_h) in zip(names, configs):
        lines.append(f'    {{ "{config_name(block, unroll, tile_h)}", {block}, {unroll}, {tile_h}, '
                     f"{-(-width // block)}, {height // rows}, "
                     f"{PATCH_W // block}, {PATCH_H // rows}, {name}_run }},")
    lines.append("};")
    return "\n".join(lines) + "\n", names


def build_firmware(args, build_dir: Path) -> Path:
    configs = parse_configs(args.configs)
    align = max(unroll for _, unroll, _ in configs)
    for block, unroll, tile_h in configs:
        if args.rows % tile_h:
            raise ValueError(f"tile height {tile_h} must divide --rows {args.rows}")
        if block & (block - 1) or unroll & (unroll - 1) or PATCH_W % block:
            raise ValueError(f"block {block} and unroll {unroll} must be powers of two, block <= {PATCH_W}")
    if MIN_ITER % align or ITER_PER_OCTAVE % align or PATCH_ITERS % align or args.max_iter % align:
        raise ValueError(f"iteration counts must be multiples of the largest unroll ({align})")
    if args.height % args.rows or PATCH_H % args.rows:
        raise ValueError(f"--rows must divide the height and {PATCH_H}")
    build_dir.mkdir(parents=True, exist_ok=True)

    for (block, unroll, tile_h), name in zip(configs, kernels_header(configs, args.width, args.height, args.rows)[1]):
        grid = (-(-args.width // block), args.height // args.rows)
        compiled = compiler.compile(mandelbrot_kernel, SIGNATURE,
                                    dict(BLOCK=block, UNROLL=unroll, ROWS=args.rows, TILE_H=tile_h), grid=grid)
        header, source = emit_c(compiled, name)
        (build_dir / f"{name}.h").write_text(header)
        (build_dir / f"{name}.c").write_text(source)
        (build_dir / f"{name}.qasm").write_text(compiled.asm["qasm"])
        print(f"Kernel {config_name(block, unroll, tile_h)}: {len(compiled.words)} QPU instructions, "
              f"{grid[0] * grid[1]} programs per frame", flush=True)
    header, names = kernels_header(configs, args.width, args.height, args.rows)
    (build_dir / "mandel_kernels.h").write_text(header)
    (build_dir / "kernels.mk").write_text("MANDEL_KERNEL_SRC := " + " ".join(f"{n}.c" for n in names) + "\n")

    # Exact reference: the emulator models the QPUs bit for bit.
    step = np.float32(FULL_VIEW / PATCH_ZOOM / args.width)
    x0 = np.float32(TARGET[0] - (PATCH_W // 2) * float(step))
    y0 = np.float32(TARGET[1] - (PATCH_H // 2) * float(step))
    print(f"Emulating a {PATCH_W}x{PATCH_H} reference patch ({PATCH_ITERS} iterations)...", flush=True)
    expected = escape_counts(x0, y0, step, PATCH_W, PATCH_H, PATCH_ITERS, *configs[0], rows=args.rows)
    expected[expected == PATCH_ITERS] = 0     # the Pi's identity palette keeps entry 0

    palette = make_palette(args.max_iter).astype(np.uint32)
    if args.swap_rb:
        palette = ((palette & 0xff) << 16) | (palette & 0xff00) | ((palette >> 16) & 0xff)
    palette |= np.uint32(0xff000000)
    (build_dir / "mandel_data.h").write_text(
        "#pragma once\n#include <stdint.h>\n" + c_u32_array("mandel_palette", palette)
        + c_u32_array("mandel_patch_expected", expected))
    config = [
        "#pragma once",
        f"#define MANDEL_WIDTH {args.width}", f"#define MANDEL_HEIGHT {args.height}",
        f"#define MANDEL_NUM_CONFIGS {len(configs)}",
        f"#define MANDEL_MIN_ITER {MIN_ITER}", f"#define MANDEL_ITER_PER_OCTAVE {ITER_PER_OCTAVE}",
        f"#define MANDEL_MAX_ITER {args.max_iter}", f"#define MANDEL_ITER_ALIGN {align}",
        f"#define MANDEL_TARGET_X {TARGET[0]!r}", f"#define MANDEL_TARGET_Y {TARGET[1]!r}",
        f"#define MANDEL_AUTO {int(not args.no_auto)}", f"#define MANDEL_BENCH_ONLY {int(args.bench_only)}",
        f"#define MANDEL_PROFILE_ONLY {int(args.profile)}",
        f"#define MANDEL_PATCH_W {PATCH_W}", f"#define MANDEL_PATCH_H {PATCH_H}",
        f"#define MANDEL_PATCH_ITERS {PATCH_ITERS}",
        f"#define MANDEL_PATCH_X0 {c_float(x0)}", f"#define MANDEL_PATCH_Y0 {c_float(y0)}",
        f"#define MANDEL_PATCH_STEP {c_float(step)}",
    ]
    (build_dir / "mandel_config.h").write_text("\n".join(config) + "\n")

    sources = [ROOT / "pi/examples/mandelbrot" / name for name in ("main.c", "Makefile")]
    sources += [ROOT / "pi/runtime" / name for name in ("piton_rt.c", "piton_rt.h")]
    sources += [ROOT / "pi/examples/matmul" / name
                for name in ("arm_sgemm.c", "arm_sgemm.h", "arm_kernel_vfp.S")]
    for source in sources:
        shutil.copyfile(source, build_dir / source.name)
    course_root = libpi_root()
    if course_root is None:
        raise RuntimeError("Set CS240LX_2025_PATH to the course repository containing libpi")
    env = {**os.environ, "CS240LX_2025_PATH": course_root}
    built = subprocess.run(["make", "-s", "RUN=0"], cwd=build_dir, env=env, capture_output=True, text=True)
    (build_dir / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise RuntimeError(f"Firmware build failed:\n{built.stdout}\n{built.stderr}")
    binary = build_dir / "main.bin"
    print(f"Built {binary.stat().st_size:,}-byte Pi firmware: {binary}", flush=True)
    return binary


def records(log: str, tag: str) -> list[dict]:
    out = []
    for line in log.splitlines():
        if line.startswith(tag + " "):
            out.append({k: v for k, v in re.findall(r"(\w+)=(\S+)", line)})
    return out


def summarize_profile(log: str) -> str:
    """Per-frame breakdown and QPU utilisation from the MANDEL_PROFILE lines."""
    clocks = {r["name"]: int(r["mhz"]) for r in records(log, "MANDEL_CLOCK")}
    work = {(r["kernel"], r["zoom"]): r for r in records(log, "MANDEL_WORK")}
    rows = ["kernel arm      zoom iters progs   fps  frame_ms gpu_ms setup_ms flip_ms  "
            "qpu_busy tmu_stall icache_miss lane_eff"]
    for r in records(log, "MANDEL_PROFILE"):
        n = {k: float(v) for k, v in r.items() if re.fullmatch(r"[\d.]+", v)}
        qpu_cycles = 12 * clocks.get("v3d", 250) * n["gpu_us"]
        fetches = n["icache_hit"] + n["icache_miss"]
        w = work.get((r["kernel"], r["zoom"]))
        rows.append(
            f"{r['kernel']:6} {r['arm']:7} {r['zoom']:>5} {r['iters']:>5} {r['programs']:>5} "
            f"{1e6 / n['frame_us']:5.1f} {n['frame_us'] / 1e3:9.2f} {n['gpu_us'] / 1e3:6.2f} "
            f"{n['setup_us'] / 1e3:8.2f} {n['flip_us'] / 1e3:7.2f} "
            f"{100 * n['qpu_valid'] / qpu_cycles:8.1f}% {100 * n['tmu_stall'] / qpu_cycles:8.1f}% "
            f"{100 * n['icache_miss'] / max(fetches, 1):10.2f}% "
            f"{float(w['lane_efficiency']) * 100 if w else float('nan'):7.1f}%")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--configs", default="32x16,16x16x2",
                    help="BLOCKxUNROLL[xTILE_H] kernels to compare")
    ap.add_argument("--max-iter", type=int, default=512)
    ap.add_argument("--rows", type=int, default=8, help="image rows drawn by each GPU program")
    ap.add_argument("--swap-rb", action="store_true", help="swap red and blue if colours look wrong")
    ap.add_argument("--no-auto", action="store_true", help="no automatic tour when idle")
    ap.add_argument("--bench-only", action="store_true", help="exit after the checks and benchmarks")
    ap.add_argument("--profile", action="store_true", help="profile every kernel at five zoom levels, then exit")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "mandelbrot")
    args = ap.parse_args()
    binary = build_firmware(args, args.build_dir)
    if args.build_only:
        return
    if args.profile:
        log = PiRunner(timeout=600, verbose=True)._boot(str(binary))
        (args.build_dir / "profile.log").write_text(log)
        print("\n" + summarize_profile(log))
        print(f"\nFull log: {args.build_dir / 'profile.log'}")
        return
    tty_path, bootloader = find_tty(), find_bootloader()
    if tty_path is None or bootloader is None:
        raise SystemExit("No Pi found: need a USB serial device and my-install/pi-install on PATH")
    print("Booting; keys go to the Pi once the benchmarks finish (q quits).", flush=True)
    # The bootloader forwards stdin to the Pi; cbreak mode sends each key
    # immediately instead of a line at a time (Ctrl-C still works).
    saved = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    try:
        if saved:
            tty.setcbreak(sys.stdin)
        subprocess.run([bootloader, tty_path, str(binary)])
    except KeyboardInterrupt:
        pass
    finally:
        if saved:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)


if __name__ == "__main__":
    main()
