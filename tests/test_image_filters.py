"""Image filters (examples/image_filters.py) on the strict emulator, and the
host side of their Pi benchmark (examples/image_filters_pi.py)."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import image_filters as F  # noqa: E402
import image_filters_pi as P  # noqa: E402

FIRMWARE = ROOT / "pi/examples/image_filters"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

# Small tiles with short strips, so a 21 x 37 image already has several
# row blocks and column blocks, ragged at the right and the bottom.
SMALL_TILES = {name: (bh, 16, 4 * bh) for name, (bh, _, _) in F.TILES.items()}
SMALL_TILES.update(blur5_rolling=(1, 16, 8), sobel_rolling=(1, 16, 8), blur_sobel_rolling=(1, 16, 8))


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


@pytest.fixture(scope="module")
def scene():
    return F.make_scene(21, 37)


@pytest.fixture(scope="module")
def pi_compiled():
    """Every filter compiled as the Pi benchmark builds it (640 x 480 bound)."""
    return {name: F.compile_filter(name, 480, 640) for name in F.TILES}


# -- the scene --------------------------------------------------------------

def test_scene_is_integer_valued_and_covers_the_range():
    img = F.make_scene(480, 640)
    assert img.dtype == np.float32 and img.shape == (480, 640)
    assert np.array_equal(img, np.round(img)) and img.min() >= 0 and img.max() <= 255
    assert len(np.unique(img)) > 200            # gradient + noise: plenty of levels
    assert np.array_equal(F.make_scene(480, 640), img)


@pytest.mark.skipif(CC is None, reason="needs a host C compiler")
@pytest.mark.parametrize("H,W", [(480, 640), (61, 83), (7, 5), (1, 1)])
def test_scene_h_matches_make_scene(tmp_path, H, W):
    """The board generates the scene with scene.h: it must give the same pixels."""
    src = tmp_path / "scene_main.c"
    src.write_text('#include <stdio.h>\n#include <stdlib.h>\n#include "scene.h"\n'
                   "int main(int argc, char **argv) {\n"
                   "    (void)argc;\n"
                   "    scene_params_t p = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),\n"
                   "                         atoi(argv[5]), atoi(argv[6]) };\n"
                   "    for (int y = 0; y < p.h; y++) for (int x = 0; x < p.w; x++) putchar(scene_pixel(&p, x, y));\n"
                   "    return 0;\n}\n")
    exe = tmp_path / "scene_main"
    subprocess.run([CC, "-O2", "-Wall", "-Werror", "-I", str(FIRMWARE), str(src), "-o", str(exe)], check=True)
    p = F.scene_params(H, W)
    out = subprocess.run([str(exe)] + [str(p[k]) for k in ("w", "h", "sx", "sy", "gx", "gy")],
                         capture_output=True, check=True).stdout
    np.testing.assert_array_equal(np.frombuffer(out, np.uint8).reshape(H, W), F.make_scene(H, W))


# -- every filter against the model ------------------------------------------

@pytest.mark.parametrize("name", list(F.TILES))
def test_filter_matches_model_bit_for_bit(scene, name):
    """Ragged in both directions, several programs each way: the QPU result is
    the model's exactly (numpy for the sums, PIton's own sqrt), and a Sobel
    magnitude is within 4 ulp of the correctly rounded one."""
    got = F.run_filter(name, scene, tiles=SMALL_TILES)
    output = F.OUTPUTS[name]
    np.testing.assert_array_equal(bits(got), bits(F.model(output, scene)))
    if "sobel" in output:
        assert F.ulp_diff(got, F.model(output, scene, sqrt=np.sqrt)).max() <= 4
    else:
        radius = 1 if output.startswith("blur3") else 2
        border = F.ZERO if output == "blur3_zero" else F.CLAMP
        # the blurs are exact: float32 of the float64 sum, whatever the order
        np.testing.assert_array_equal(got, F.blur_reference(scene, radius, border))


def test_pi_tiles_at_the_check_size(pi_compiled):
    """The shipped tiles (the code the Pi runs) on the ragged check image."""
    img = F.make_scene(*P.CHECK_SIZE)
    for name in ("blur5", "blur5_rolling", "blur_sobel_rolling", "sobel"):
        got = F.run_filter(name, img, compiled=pi_compiled[name])
        np.testing.assert_array_equal(bits(got), bits(F.model(F.OUTPUTS[name], img)), err_msg=name)


def test_fused_blur_sobel_is_blur_then_sobel_away_from_the_border(scene):
    blurred = F.run_filter("blur3", scene, tiles=SMALL_TILES)
    two_pass = F.run_filter("sobel", blurred, tiles=SMALL_TILES)
    fused = F.run_filter("blur_sobel_rolling", scene, tiles=SMALL_TILES)
    np.testing.assert_array_equal(bits(fused[2:-2, 2:-2]), bits(two_pass[2:-2, 2:-2]))
    assert not np.array_equal(fused, two_pass)        # the borders do differ (clamping)


@pytest.mark.parametrize("H,W", [(1, 1), (1, 5), (4, 1), (3, 17), (2, 2)])
def test_tiny_images_clamp_every_tap(H, W):
    """Images smaller than the stencil: every out-of-image tap is the edge pixel."""
    img = F.make_scene(H, W)
    for name in ("blur5", "blur5_rolling", "sobel_rolling", "blur_sobel"):
        got = F.run_filter(name, img, tiles=SMALL_TILES)
        np.testing.assert_array_equal(bits(got), bits(F.model(F.OUTPUTS[name], img)), err_msg=name)
    if H == 1 and W == 1:
        assert F.run_filter("sobel", img, tiles=SMALL_TILES)[0, 0] == 0     # flat: sqrt(0) = +0


def test_zero_border_kernels():
    """BORDER=ZERO: masked loads with other=0 (blur, separable pass, Sobel)."""
    img = F.make_scene(13, 21)
    got = F.run_filter("blur3_zero", img, tiles=SMALL_TILES)
    np.testing.assert_array_equal(got, F.blur_reference(img, 1, F.ZERO))
    assert got[0, 0] < img[0, 0] * 0.75              # the zeros outside darken the corner
    from piton import compiler
    from piton.runtime.launcher import run_emulator
    for kernel, constants, want in [
        (F.blur_pass_kernel, dict(BLOCK_H=2, BLOCK_W=16, ROWS=4, RADIUS=2, BORDER=F.ZERO, AXIS=0),
         (sum(w * F.shifted(img, j - 2, 0, F.ZERO) for j, w in enumerate(F.binomial(2))) / 16).astype(np.float32)),
        (F.sobel_kernel, dict(BLOCK_H=2, BLOCK_W=16, ROWS=4, BORDER=F.ZERO),
         F.qpu_sqrt(F.sobel_squared(img, F.ZERO))),
    ]:
        grid = F.grid_for(13, 21, constants)
        ck = compiler.compile(kernel, F.SIGNATURE, constants, grid=grid)
        out = np.full(img.size, np.nan, np.float32)
        run_emulator(ck, grid, dict(src_ptr=img.ravel().copy(), dst_ptr=out, H=13, W=21,
                                    src_stride=21, dst_stride=21))
        np.testing.assert_array_equal(bits(out.reshape(13, 21)), bits(want), err_msg=kernel.__name__)


def test_row_strides_and_untouched_padding(scene):
    """Strides wider than the image: same pixels, and the padding is never written."""
    H, W = scene.shape
    for name in ("blur3", "blur5_rolling", "sobel_rolling"):
        got = F.run_filter(name, scene, tiles=SMALL_TILES, src_stride=W + 11, dst_stride=W + 5)
        np.testing.assert_array_equal(bits(got), bits(F.model(F.OUTPUTS[name], scene)), err_msg=name)
    from piton.runtime.launcher import run_emulator
    (ck, constants), = F.compile_filter("blur5_rolling", H, W, SMALL_TILES)
    dst = np.full(H * (W + 5), np.nan, np.float32)
    run_emulator(ck, F.grid_for(H, W, constants), dict(src_ptr=scene.ravel().copy(), dst_ptr=dst, H=H, W=W,
                                                         src_stride=W, dst_stride=W + 5))
    assert np.isnan(dst.reshape(H, W + 5)[:, W:]).all() and not np.isnan(dst.reshape(H, W + 5)[:, :W]).any()


def test_non_integer_images_within_rounding_tolerance():
    """On arbitrary floats the summation order matters (and the QPU rounds
    toward zero): each blur is within a few ulp of float64, below it, and the
    three 5x5 versions agree to the same tolerance."""
    rng = np.random.default_rng(7)
    img = rng.uniform(0, 1, (19, 35)).astype(np.float32)
    exact = F.blur_reference(img, 2).astype(np.float64)
    w = F.binomial(2)
    ref = sum(w[i] * w[j] * F.shifted(img, i - 2, j - 2) for i in range(5) for j in range(5)) / 256
    outs = {name: F.run_filter(name, img, tiles=SMALL_TILES).astype(np.float64)
            for name in ("blur5", "blur5_separable", "blur5_rolling")}
    for name, got in outs.items():
        rel = np.abs(got - ref) / ref
        assert rel.max() < 2e-6, name               # ~a dozen roundings of 2**-24 each
        assert (got <= ref * (1 + 1e-12)).all(), name  # toward zero: never above the exact value
    assert np.abs(exact - ref).max() < 1e-6


# -- the Pi build -------------------------------------------------------------

def test_filters_compile_at_the_pi_size(pi_compiled):
    """Every filter fits the registers at 640 x 480 with the shipped tiles,
    and the programs stay near the ~500-instruction QPU cache."""
    from piton.tools.compile import emit_c
    for name, launches in pi_compiled.items():
        assert len(launches) == (2 if name == "blur5_separable" else 1)
        for i, (ck, constants) in enumerate(launches):
            assert len(ck.program.instrs) < 600, name
            header, source = emit_c(ck, P.symbol(name, i))
            assert f"{P.symbol(name, i)}_launch" in header and "piton_launch" in source
            assert [n for k, n in ck.uniforms if k == "arg"] == list(F.SIGNATURE)
    # the rolling versions are the short ones
    size = {n: sum(len(ck.program.instrs) for ck, _ in v) for n, v in pi_compiled.items()}
    assert size["sobel_rolling"] < size["sobel"] and size["blur_sobel_rolling"] < size["blur_sobel"]


@pytest.mark.skipif(CC is None, reason="needs a host C compiler")
def test_row_hash_h_matches_row_hashes(tmp_path):
    src = tmp_path / "hash_main.c"
    src.write_text('#include <stdio.h>\n#include "row_hash.h"\n'
                   "int main(void) {\n    static uint32_t img[7 * 13];\n"
                   "    if (fread(img, 4, 7 * 13, stdin) != 7 * 13) return 1;\n"
                   '    for (int y = 0; y < 7; y++) printf("%u\\n", row_hash(img + 13 * y, 13));\n'
                   "    return 0;\n}\n")
    exe = tmp_path / "hash_main"
    subprocess.run([CC, "-O2", "-Wall", "-Werror", "-I", str(FIRMWARE), str(src), "-o", str(exe)], check=True)
    img = np.random.default_rng(3).normal(size=(7, 13)).astype(np.float32)
    out = subprocess.run([str(exe)], input=img.tobytes(), capture_output=True, check=True).stdout.split()
    np.testing.assert_array_equal(np.array([int(v) for v in out], np.uint32), P.row_hashes(img))


def test_png_round_trip(tmp_path):
    gray = (np.arange(35 * 23) % 256).astype(np.uint8).reshape(23, 35)
    F.write_png(str(tmp_path / "a.png"), gray)
    np.testing.assert_array_equal(F.read_png(str(tmp_path / "a.png")), gray)
    assert (tmp_path / "a.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# The firmware's own logic on the host: main.c with the real generated
# if_kernels.h, but each generated launcher replaced by a C stand-in for the
# QPU kernel (QPU rounding modelled with fesetround), and rpi.h / piton_rt.h
# stubs. Its log must pass parse_log against a model with a correctly rounded
# sqrt; a broken row, a missing record or a failed comparison must not.

SIM_RPI_H = r"""
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define printk printf
#define panic(...) (printf(__VA_ARGS__), exit(1))
static inline uint32_t timer_get_usec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)(t.tv_sec * 1000000u + t.tv_nsec / 1000u) | 1u;
}
static inline void caches_enable(void) {}
static inline void caches_disable(void) {}
"""

SIM_RT_H = r"""
#pragma once
#include <stdint.h>
typedef struct { uint32_t handle, bus; volatile void *cpu; uint32_t size; } piton_buf_t;
void *sim_ptr(uint32_t bus);
piton_buf_t piton_alloc(uint32_t size);
static inline void piton_init(void) {}
static inline uint32_t piton_qpu_count(void) { return 12; }
#define PITON_CLOCK_ARM 3
#define PITON_CLOCK_CORE 4
#define PITON_CLOCK_V3D 5
#define PITON_CLOCK_SDRAM 8
static inline uint32_t piton_clock_hz(uint32_t id, int max) { (void)max; return id == 3 ? 700000000u : 250000000u; }
"""

SIM_SGEMM_H = "#pragma once\nstatic inline void arm_caches_on(void) {}\nstatic inline void arm_caches_off(void) {}\n"

SIM_C = r"""
#include <fenv.h>
#include <stdlib.h>
#include <math.h>
#include "piton_rt.h"
static void *blocks[64];
static unsigned nblocks;
void *sim_ptr(uint32_t bus) { return blocks[bus >> 24]; }
piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b = { nblocks + 1, nblocks << 24, calloc(size, 1), size };
    blocks[nblocks++] = (void *)b.cpu;
    return b;
}
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static float at(const float *s, int y, int x, int h, int w, int stride, int zero) {
    if (zero && (y < 0 || y >= h || x < 0 || x >= w)) return 0;
    return s[clampi(y, 0, h - 1) * stride + clampi(x, 0, w - 1)];
}
static const float b3[3] = {1, 2, 1}, b5[5] = {1, 4, 6, 4, 1}, d3[3] = {-1, 0, 1}, d5[5] = {-1, -2, 0, 2, 1};
// mode: 0 blur, 1 row pass, 2 column pass, 3 sobel
void sim_filter(uint32_t src, uint32_t dst, int h, int w, int ss, int ds, int mode, int r, int zero) {
    const float *s = sim_ptr(src); float *d = sim_ptr(dst);
    const float *k = r == 1 ? b3 : b5, *dk = r == 1 ? d3 : d5;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float acc = 0, gx = 0, gy = 0;
            for (int i = -r; i <= r; i++)
                for (int j = -r; j <= r; j++) {
                    if ((mode == 1 && i) || (mode == 2 && j)) continue;
                    float v = at(s, y + i, x + j, h, w, ss, zero);
                    acc += (mode == 0 ? k[i + r] * k[j + r] : mode == 1 ? k[j + r] : k[i + r]) * v;
                    gx += k[i + r] * dk[j + r] * v;
                    gy += dk[i + r] * k[j + r] * v;
                }
            float out = acc * (mode == 0 ? (r == 1 ? 1.0f / 16 : 1.0f / 256) : (r == 1 ? 0.25f : 1.0f / 16));
            if (mode == 3) {
                volatile float vx = gx * (r == 2 ? 1.0f / 16 : 1.0f), vy = gy * (r == 2 ? 1.0f / 16 : 1.0f);
                fesetround(FE_TOWARDZERO);              // the QPU's fmul / fadd
                volatile float sq = vx * vx + vy * vy;
                fesetround(FE_TONEAREST);
                out = sqrtf(sq);
            }
            d[y * ds + x] = out;
        }
}
"""


def sim_launchers(names, compiled):
    """C stand-ins for every generated <symbol>_launch: check the grid the
    wrapper computed, then filter in C."""
    lines = ['#include "rpi.h"', '#include "if_kernels.h"',
             "void sim_filter(uint32_t, uint32_t, int, int, int, int, int, int, int);"]
    for name in names:
        for i, (ck, c) in enumerate(compiled[name]):
            s = P.symbol(name, i)
            if name == "blur5_separable":
                mode, r, zero = (1 if c["AXIS"] == 1 else 2), 2, 0
            elif "sobel" in name:
                mode, r, zero = 3, (2 if name.startswith("blur_sobel") else 1), 0
            else:
                mode, r, zero = 0, (1 if name.startswith("blur3") else 2), int(name == "blur3_zero")
            lines += [f"uint32_t {s}_launch(const {s}_args_t *a, unsigned gx, unsigned gy, unsigned gz) {{",
                      f"    if (gx != (a->W + {c['BLOCK_W'] - 1}u) / {c['BLOCK_W']}u || "
                      f"gy != (a->H + {c['ROWS'] - 1}u) / {c['ROWS']}u || gz != 1) panic(\"bad grid\\n\");",
                      f"    sim_filter(a->src_ptr, a->dst_ptr, a->H, a->W, a->src_stride, a->dst_stride, "
                      f"{mode}, {r}, {zero});",
                      "    return 1;", "}"]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def host_firmware(tmp_path_factory, pi_compiled):
    if CC is None:
        pytest.skip("needs a host C compiler")
    from piton.tools.compile import emit_c
    d = tmp_path_factory.mktemp("image_filters_sim")
    names = list(F.TILES)
    for name in names:
        for i, (ck, _) in enumerate(pi_compiled[name]):
            (d / f"{P.symbol(name, i)}.h").write_text(emit_c(ck, P.symbol(name, i))[0])
    (d / "if_kernels.h").write_text(P.kernels_header(names, pi_compiled))
    (H, W), (ch, cw), repeats = (37, 45), (11, 13), 2
    (d / "if_config.h").write_text("\n".join([
        "#pragma once", f"#define IF_W {W}", f"#define IF_H {H}", f"#define IF_REPEATS {repeats}",
        f"#define IF_MAX_ULP {P.MAX_ULP}", "#define IF_DUMP 1",
        f"#define IF_SCENE_FULL {P.c_scene(F.scene_params(H, W))}",
        f"#define IF_SCENE_CHECK {P.c_scene(F.scene_params(ch, cw))}"]) + "\n")
    for name in ("main.c", "scene.h", "row_hash.h", "arm_filters.c", "arm_filters.h"):
        shutil.copyfile(FIRMWARE / name, d / name)
    (d / "rpi.h").write_text(SIM_RPI_H)
    (d / "piton_rt.h").write_text(SIM_RT_H)
    (d / "arm_sgemm.h").write_text(SIM_SGEMM_H)
    (d / "sim.c").write_text(SIM_C)
    (d / "launchers.c").write_text(sim_launchers(names, pi_compiled))
    (d / "start.c").write_text("void notmain(void);\nint main(void) { notmain(); return 0; }\n")
    # no fused multiply-adds: the ARM1176 has none, and the RTZ model rounds each step
    subprocess.run([CC, "-O1", "-std=gnu99", "-ffp-contract=off", "-Wall", "-Werror", "-Wno-unused-function",
                    "-Wno-unused-variable",
                    "-I", str(d), *(str(d / f) for f in ("main.c", "arm_filters.c", "sim.c", "launchers.c",
                                                          "start.c")), "-lm", "-o", str(d / "sim")], check=True)
    log = subprocess.run([str(d / "sim")], capture_output=True, text=True, check=True).stdout
    model = {}
    for size, (h, w) in (("check", (ch, cw)), ("full", (H, W))):
        img = F.make_scene(h, w)
        model[size] = {"scene": img, **{o: F.model(o, img, sqrt=np.sqrt) for o in P.OUTPUTS}}
    return log, names, model, repeats


def test_firmware_logic_on_the_host(host_firmware):
    log, names, model, repeats = host_firmware
    result = P.parse_log(log, names, model, repeats, dump=True)
    assert result["rows_checked"] == len(names) * (11 + 37)
    assert {t["impl"] for t in result["timings"]} == {"qpu", "qpu_icache", "arm"}
    assert len(result["timings"]) == 2 * len(names) + len(P.ARM_IMPLS)
    checks = {c["arm"]: c for c in result["arm_checks"]}
    assert checks["arm_blur5_separable"]["exact"] == checks["arm_blur5_separable"]["checked"] == str(37 * 45)
    assert checks["arm_blur_sobel_two_pass"]["checked"] == str(33 * 41)      # 2-pixel border left out
    np.testing.assert_array_equal(result["images"]["blur5"], F.display("blur5", model["full"]["blur5"]))
    assert sorted(result["images"]) == sorted(P.PI_DUMP)
    edges = F.display("blur_sobel", model["full"]["blur_sobel"]).astype(int)
    assert np.abs(result["images"]["blur_sobel"].astype(int) - edges).max() <= 1
    table, speedups = P.summarize(result, names)
    assert set(speedups) == {"blur3", "blur5", "sobel", "blur_sobel"} and "blur_sobel_rolling" in table


def corrupt_row(log, filter_name, size):
    lines = log.splitlines()
    i = lines.index(next(l for l in lines if l.startswith(f"IF_ROWS filter={filter_name} size={size} ")))
    first = lines[i + 1].split()
    first[0] = f"{(int(first[0], 16) ^ 1):08x}"
    lines[i + 1] = " ".join(first)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("mutation,message", [
    (lambda log: log.replace("PITON_DONE", ""), "PITON_DONE"),
    (lambda log: corrupt_row(log, "blur5_rolling", "full"), "blur5_rolling full: 1 of 37 rows"),
    (lambda log: corrupt_row(log, "sobel", "check"), "sobel check: 1 of 11 rows"),
    (lambda log: re.sub(r"(IF_SAME filter=blur5_rolling output=blur5 mismatches=)0", r"\g<1>3", log), "3 pixels"),
    (lambda log: re.sub(r"^IF_TIME impl=arm name=arm_sobel [^\n]*\n", "", log, count=1, flags=re.M),
     "timings for arm arm_sobel"),
    (lambda log: re.sub(r"(IF_CHECK arm=arm_blur5 [^\n]*failures=)0", r"\g<1>2", log), "arm_blur5 vs the QPU"),
    (lambda log: log.replace("IF_CHECK_OK", "IF_CHECK_FAILED"), "IF_CHECK_FAILED"),
    (lambda log: re.sub(r"(IF_SCENE size=full h=37 w=45 hash=)\d+", r"\g<1>1", log), "full scene"),
    (lambda log: log.replace("IF_ROWS_END", "", 1), "cut off"),
])
def test_parse_log_rejects_bad_runs(host_firmware, mutation, message):
    log, names, model, repeats = host_firmware
    with pytest.raises(ValueError, match=re.escape(message)):
        P.parse_log(mutation(log), names, model, repeats, dump=True)


@pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None or P.libpi_root() is None,
                    reason="needs arm-none-eabi-gcc and CS240LX_2025_PATH")
def test_firmware_builds(tmp_path, pi_compiled):
    """The bare-metal program compiles and links (no boot)."""
    args = P.argparse.Namespace(build_dir=tmp_path / "build", width=640, height=480, check_size=P.CHECK_SIZE,
                                repeats=5, dump_images=False)
    binary, manifest, _ = P.build_firmware(args, list(F.TILES), compiled=pi_compiled)
    assert binary.stat().st_size > 10_000 and manifest["filters"] == list(F.TILES)
    assert len(manifest["kernels"]) == len(F.TILES) + 1        # the separable blur is two kernels
