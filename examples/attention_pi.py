"""Scaled dot-product attention on the attached bare-metal Pi: QPUs against the ARM.

    .venv/bin/python examples/attention_pi.py               # build, boot, check, time (~3 min)
    .venv/bin/python examples/attention_pi.py --build-only  # compile kernels and firmware only
    .venv/bin/python examples/attention_pi.py --dry-run     # also parse a synthetic log

Builds pi/examples/attention: every tile configuration of examples/attention.py
in PI_TILES for HEAD_DIM 32 and 64, causal and not, at N_CTX 128, 256 and 512
(one kernel per configuration, N_CTX a runtime argument), plus a ragged N_CTX
of 499 (masked tails), K stored pre-transposed, and the 64-token pattern with
V = I whose output is the attention matrix.  Inputs are generated on the board
(the piton_rt.h LCG, mirrored here), so the boot image carries only code.

On the board, the ARM computes every problem with the straightforward C
baseline (pi/examples/attention/arm_attention.c, -O3, data cache on) and times
it; then every case runs on the 12 QPUs, is checked against the ARM over all
outputs, timed (the whole launch call: uniforms, cache clear, queue, wait), and
checked again.  A sample of each case's outputs (every outputs of the pattern)
is printed and compared here with float64 numpy.  Results go to
benchmarks/results/attention-pi-<date>.{json,log} and
docs/images/attention-pi-weights.png.
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

from attention import (PI_TILES, attention_flops, attention_reference, compile_attention,  # noqa: E402
                       pattern_qk, weights_image, write_png)
from piton.runtime.launcher import slot_semaphores  # noqa: E402
from piton.runtime.pi import PiError, PiRunner, lcg_floats, libpi_root  # noqa: E402
from piton.tools.compile import emit_c  # noqa: E402

SIZES = (128, 256, 512)
MAX_N = 512
RAGGED_N = 499
PATTERN_N = 64
SEED_Q, SEED_K, SEED_V = 1000, 2000, 3000      # + HEAD_DIM, as in main.c
Q_GAIN = 4.0                                    # scores with a standard deviation near 1.3
ATOL, RTOL = 5e-5, 1e-4                         # |out - ref| <= ATOL + RTOL |ref| (emulator: <= 4e-6)
LCG, PATTERN = 0, 1
FIRMWARE = ROOT / "pi/examples/attention"


# -- what runs --------------------------------------------------------------------

def plan(sizes=SIZES):
    """(kernels, problems, cases): kernel dicts, problem dicts, (kernel, problem) pairs."""
    kernels, problems, cases = [], [], []

    def kernel(d, bm, bn, causal, even=True, layout="nd"):
        key = (d, bm, bn, causal, even, layout)
        for i, k in enumerate(kernels):
            if k["key"] == key:
                return i
        kernels.append(dict(key=key, head_dim=d, block_m=bm, block_n=bn, causal=causal, even_n=even,
                            k_layout=layout, name=f"d{d}_{bm}x{bn}{'_causal' if causal else ''}"
                            f"{'' if even else '_ragged'}{'_kt' if layout == 'dn' else ''}"))
        return len(kernels) - 1

    def problem(n, d, causal, inputs=LCG):
        key = (n, d, causal, inputs)
        for i, p in enumerate(problems):
            if p["key"] == key:
                return i
        problems.append(dict(key=key, n_ctx=n, head_dim=d, causal=causal, inputs=inputs))
        return len(problems) - 1

    for (d, causal), tiles in PI_TILES.items():
        for bm, bn in tiles:
            k = kernel(d, bm, bn, causal)
            for n in sizes:
                cases.append((k, problem(n, d, causal)))
    for causal in (False, True):            # ragged N_CTX: masked key tail and query rows
        cases.append((kernel(32, 2, 32, causal, even=False), problem(RAGGED_N, 32, causal)))
    # K stored [HEAD_DIM, N_CTX], against the same tiles on [N_CTX, HEAD_DIM] (at HEAD_DIM 64
    # the transposed layout fits only 1-row tiles, which PI_TILES does not run)
    for bm, bn in ((2, 32), (2, 16)):
        cases.append((kernel(32, bm, bn, False, layout="dn"), problem(MAX_N, 32, False)))
    for causal in (False, True):            # the attention matrix itself (V = I)
        cases.append((kernel(64, 2, 16, causal), problem(PATTERN_N, 64, causal, PATTERN)))
    return kernels, problems, cases


def sample_stride(problem) -> int:
    if problem["inputs"] == PATTERN:
        return 1
    return max(1, problem["n_ctx"] * problem["head_dim"] // 256) | 1      # odd: every column


def inputs_for(problem):
    """The board's inputs for a problem, bit for bit."""
    n, d = problem["n_ctx"], problem["head_dim"]
    if problem["inputs"] == PATTERN:
        q, k = pattern_qk(n, d)
        return q, k, np.eye(n, dtype=np.float32)
    q = np.float32(Q_GAIN) * lcg_floats(MAX_N * d, SEED_Q + d).reshape(MAX_N, d)
    k = lcg_floats(MAX_N * d, SEED_K + d).reshape(MAX_N, d)
    v = lcg_floats(MAX_N * d, SEED_V + d).reshape(MAX_N, d)
    return q[:n], k[:n], v[:n]


def references(problems):
    cache, out = {}, []
    for p in problems:
        if p["inputs"] == LCG and p["head_dim"] not in cache:
            cache[p["head_dim"]] = inputs_for(dict(p, n_ctx=MAX_N))
        if p["inputs"] == LCG:
            q, k, v = (x[:p["n_ctx"]] for x in cache[p["head_dim"]])
        else:
            q, k, v = inputs_for(p)
        out.append(attention_reference(q, k, v, p["causal"]).astype(np.float64))
    return out


# -- firmware ---------------------------------------------------------------------

def c_floats(name, values):
    values = np.asarray(values, np.float32).ravel()
    items = [float(x).hex() + "f" for x in values]
    lines = [", ".join(items[i:i + 8]) + "," for i in range(0, len(items), 8)]
    return f"static const float {name}[{len(values)}] = {{\n" + "\n".join(lines) + "\n};\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_kernel(spec: dict, symbol: str) -> dict:
    """One kernel of the plan at the Pi size, as C (picklable, for a process pool)."""
    started = time.monotonic()
    ck = compile_attention(spec["head_dim"], spec["block_m"], spec["block_n"], spec["causal"],
                           spec["even_n"], spec["k_layout"], n_ctx=MAX_N, max_n_ctx=MAX_N)
    header, source = emit_c(ck, symbol)
    return dict(symbol=symbol, header=header, source=source, qasm=ck.asm["qasm"],
                instructions=len(ck.words), signature=list(ck.signature),
                semaphores=slot_semaphores(ck.words), seconds=time.monotonic() - started,
                code_sha256=hashlib.sha256(b"".join(int(w).to_bytes(8, "little") for w in ck.words)).hexdigest())


def compile_kernels(kernels, jobs: int = 1) -> list:
    """compile_kernel for every kernel, `jobs` at a time (each takes 10-45 s with the
    default pairing schedules, PITON_PAIR)."""
    symbols = [f"attn_k{i}" for i in range(len(kernels))]
    if jobs <= 1:
        return [compile_kernel(k, s) for k, s in zip(kernels, symbols)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(compile_kernel, kernels, symbols))


def build_firmware(build_dir: Path, repeats: int, arm_repeats: int, sizes=SIZES, make=True, jobs=1):
    kernels, problems, cases = plan(sizes)
    build_dir.mkdir(parents=True, exist_ok=True)
    kernel_src, launchers, table, semaphores = [], [], [], set()
    started = time.monotonic()
    for i, (k, out) in enumerate(zip(kernels, compile_kernels(kernels, jobs))):
        sym = out["symbol"]
        if out["semaphores"]:
            semaphores.add(out["semaphores"])
        if len(semaphores) > 1:
            # piton_launch_slots_once sets the store-slot semaphores up once per boot
            raise RuntimeError(f"kernels use different store-slot semaphores {semaphores}")
        (build_dir / f"{sym}.h").write_text(out["header"])
        (build_dir / f"{sym}.c").write_text(out["source"])
        (build_dir / f"{sym}.qasm").write_text(out["qasm"])
        kernel_src.append(f"{sym}.c")
        fields = [".Q = c->q", ".K = c->k", ".V = c->v", ".Out = c->out", ".sm_scale = c->sm_scale",
                  ".N_CTX = (int32_t)c->n_ctx"]
        fields += [f".{name} = 0" for name in out["signature"] if name.startswith("stride_")]
        launchers.append(f'#include "{sym}.h"\nstatic uint32_t launch_{i}(const attn_call_t *c) {{\n'
                         f'    {sym}_args_t a = {{ {", ".join(fields)} }};\n'
                         f'    return {sym}_launch(&a, c->grid_x, 1, 1);\n}}\n')
        table.append(f'    {{ "{k["name"]}", {k["head_dim"]}, {k["block_m"]}, {k["block_n"]}, '
                     f'{int(k["causal"])}, {int(k["even_n"])}, {int(k["k_layout"] == "dn")}, '
                     f'{out["instructions"]}, launch_{i} }},')
        k.update(instructions=out["instructions"], symbol=sym, compile_seconds=round(out["seconds"], 2),
                 code_sha256=out["code_sha256"])
        print(f"kernel {i:2d} {k['name']:24s} {out['instructions']:5d} QPU instructions "
              f"({out['seconds']:.1f} s)", flush=True)
    print(f"compiled {len(kernels)} kernels in {time.monotonic() - started:.0f} s ({jobs} at a time)", flush=True)
    for p in problems:
        p["sm_scale"] = float(np.float32(1.0 / np.sqrt(p["head_dim"])))
    for c, (ki, pi) in enumerate(cases):
        k, p = kernels[ki], problems[pi]
        if k["head_dim"] != p["head_dim"] or k["causal"] != p["causal"]:
            raise AssertionError(f"case {c}: kernel and problem disagree")
        if not k["even_n"] and p["n_ctx"] % k["block_n"] == 0:
            raise AssertionError(f"case {c}: the ragged kernel should run a ragged N_CTX")
        if k["even_n"] and p["n_ctx"] % k["block_n"]:
            raise AssertionError(f"case {c}: EVEN_N kernel on N_CTX {p['n_ctx']}")
        if k["k_layout"] == "dn" and p["n_ctx"] != MAX_N:
            raise AssertionError(f"case {c}: the pre-transposed K kernels are built for N_CTX {MAX_N}")
    header = ["#pragma once", "#include <stdint.h>",
              f"#define ATTN_KERNEL_COUNT {len(kernels)}", f"#define ATTN_PROBLEM_COUNT {len(problems)}",
              f"#define ATTN_CASE_COUNT {len(cases)}", f"#define ATTN_REPEATS {repeats}",
              f"#define ATTN_ARM_REPEATS {arm_repeats}", f"#define ATTN_MAX_N {MAX_N}",
              f"#define ATTN_REF_FLOATS {sum(p['n_ctx'] * p['head_dim'] for p in problems)}",
              f"#define ATTN_SEED_Q {SEED_Q}u", f"#define ATTN_SEED_K {SEED_K}u", f"#define ATTN_SEED_V {SEED_V}u",
              f"#define ATTN_Q_GAIN {Q_GAIN!r}f", f"#define ATTN_ATOL {ATOL!r}f", f"#define ATTN_RTOL {RTOL!r}f",
              f"#define ATTN_INPUTS_LCG {LCG}", f"#define ATTN_INPUTS_PATTERN {PATTERN}", "",
              "typedef struct { uint32_t q, k, v, out; float sm_scale; uint32_t n_ctx; unsigned grid_x; } attn_call_t;",
              "typedef struct { const char *name; unsigned head_dim, block_m, block_n, causal, even_n, k_transposed,",
              "                 instructions; uint32_t (*launch)(const attn_call_t *); } attn_kernel_t;",
              "typedef struct { unsigned n_ctx, head_dim, causal, inputs; float sm_scale; } attn_problem_t;",
              "typedef struct { unsigned kernel, problem, grid_x, sample_stride; } attn_case_t;",
              "extern const attn_kernel_t attn_kernels[ATTN_KERNEL_COUNT];",
              "extern const attn_problem_t attn_problems[ATTN_PROBLEM_COUNT];",
              "extern const attn_case_t attn_cases[ATTN_CASE_COUNT];", ""]
    (build_dir / "attn_cases.h").write_text("\n".join(header))
    body = ['#include "attn_cases.h"', *launchers,
            "const attn_kernel_t attn_kernels[ATTN_KERNEL_COUNT] = {", *table, "};",
            "const attn_problem_t attn_problems[ATTN_PROBLEM_COUNT] = {",
            *(f"    {{ {p['n_ctx']}, {p['head_dim']}, {int(p['causal'])}, {p['inputs']}, "
              f"{float(p['sm_scale']).hex()}f }}," for p in problems), "};",
            "const attn_case_t attn_cases[ATTN_CASE_COUNT] = {",
            *(f"    {{ {ki}, {pi}, {-(-problems[pi]['n_ctx'] // kernels[ki]['block_m'])}, "
              f"{sample_stride(problems[pi])} }}," for ki, pi in cases), "};", ""]
    (build_dir / "attn_cases.c").write_text("\n".join(body))
    (build_dir / "kernels.mk").write_text("ATTN_KERNEL_SRC := " + " ".join(kernel_src) + "\n")
    q, _ = pattern_qk(PATTERN_N, 64)
    (build_dir / "attn_data.h").write_text("#pragma once\n" + c_floats("attn_pattern_qk", q))
    sources = [FIRMWARE / n for n in ("main.c", "Makefile", "arm_attention.c", "arm_attention.h")]
    sources += [ROOT / "pi/runtime" / n for n in ("piton_rt.c", "piton_rt.h")]
    sources += [ROOT / "pi/examples/matmul" / n for n in ("arm_sgemm.c", "arm_sgemm.h", "arm_kernel_vfp.S")]
    for src in sources:
        shutil.copyfile(src, build_dir / src.name)
    manifest = dict(kernels=[{k2: v for k2, v in k.items() if k2 != "key"} for k in kernels],
                    problems=[{k2: v for k2, v in p.items() if k2 != "key"} for p in problems],
                    cases=[dict(kernel=ki, problem=pi, sample_stride=sample_stride(problems[pi])) for ki, pi in cases],
                    repeats=repeats, arm_repeats=arm_repeats, atol=ATOL, rtol=RTOL, q_gain=Q_GAIN,
                    seeds=dict(q=SEED_Q, k=SEED_K, v=SEED_V),
                    source_sha256={str(s.relative_to(ROOT)): sha256(s) for s in sources + [
                        ROOT / "examples/attention.py", ROOT / "examples/attention_pi.py"]})
    if make:
        course = libpi_root()
        if course is None:
            raise RuntimeError("Set CS240LX_2025_PATH to the course repository containing libpi")
        env = {**os.environ, "CS240LX_2025_PATH": course}
        built = subprocess.run(["make", "-s", "RUN=0"], cwd=build_dir, env=env, capture_output=True, text=True)
        (build_dir / "build.log").write_text(built.stdout + built.stderr)
        if built.returncode or not (build_dir / "main.bin").exists():
            raise RuntimeError(f"Firmware build failed:\n{built.stdout}\n{built.stderr}")
        manifest.update(firmware_bytes=(build_dir / "main.bin").stat().st_size,
                        firmware_sha256=sha256(build_dir / "main.bin"))
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return kernels, problems, cases, manifest


# -- the serial log ---------------------------------------------------------------

def _bits_to_float(text: str) -> float:
    return float(np.array([int(text, 16)], np.uint32).view(np.float32)[0])


def parse_log(log, kernels, problems, cases, repeats, arm_repeats, refs):
    """Every record of a complete run, checked; raises ValueError on anything missing or wrong."""
    if not re.search(r"^PITON_DONE\s*$", log, re.MULTILINE):
        raise ValueError("Pi completion marker is missing")
    if "ATTN_CHECK_FAILED" in log or not re.search(r"^ATTN_CHECK_OK\s*$", log, re.MULTILINE):
        raise ValueError("the board's check against the ARM reference failed")
    clocks = dict((n, int(v)) for n, v in re.findall(r"^ATTN_CLOCK name=(\w+) mhz=(\d+)\s*$", log, re.MULTILINE))
    if sorted(clocks) != ["arm", "core", "sdram", "v3d"] or min(clocks.values()) <= 0:
        raise ValueError("missing or invalid clocks")
    arm = {}
    for p, us in re.findall(r"^ATTN_ARM problem=(\d+) us=(\d+)\s*$", log, re.MULTILINE):
        arm.setdefault(int(p), []).append(int(us))
    for p, differ in re.findall(r"^ATTN_ARM_REPEAT problem=(\d+) differing=(\d+)\s*$", log, re.MULTILINE):
        if int(differ):
            raise ValueError(f"ARM problem {p}: timed runs differ from the reference run")
    if len(re.findall(r"^ATTN_ARM_REPEAT ", log, re.MULTILINE)) != len(problems):
        raise ValueError("missing ARM repeat checks")
    if sorted(arm) != list(range(len(problems))) or any(len(t) != arm_repeats or min(t) <= 0 for t in arm.values()):
        raise ValueError("missing or invalid ARM timings")
    times, checks, samples = {}, {}, {}
    for c, us in re.findall(r"^ATTN_TIME case=(\d+) us=(\d+)\s*$", log, re.MULTILINE):
        times.setdefault(int(c), []).append(int(us))
    for c, pas, checked, failures, nonfinite, bits in re.findall(
            r"^ATTN_CHECK case=(\d+) pass=(\w+) checked=(\d+) failures=(\d+) nonfinite=(\d+) "
            r"max_abs_err_bits=([0-9a-f]+)\s*$", log, re.MULTILINE):
        checks[(int(c), pas)] = dict(checked=int(checked), failures=int(failures), nonfinite=int(nonfinite),
                                     max_abs_err=_bits_to_float(bits))
    for c, stride, count, block in re.findall(
            r"^ATTN_SAMPLE case=(\d+) stride=(\d+) count=(\d+)\s*\n(.*?)^ATTN_SAMPLE_END case=\1\s*$",
            log, re.MULTILINE | re.DOTALL):
        words = [int(w, 16) for w in block.split()]
        if len(words) != int(count):
            raise ValueError(f"case {c}: {len(words)} sampled outputs, expected {count}")
        samples[int(c)] = (int(stride), np.array(words, np.uint32).view(np.float32))
    rows = []
    for c, (ki, pi) in enumerate(cases):
        k, p = kernels[ki], problems[pi]
        n_out = p["n_ctx"] * p["head_dim"]
        t = times.get(c, [])
        if len(t) != repeats or min(t) <= 0:
            raise ValueError(f"case {c}: expected {repeats} positive timings, got {t}")
        for pas in ("first", "timed"):
            ch = checks.get((c, pas))
            if ch is None or ch["checked"] != n_out or ch["failures"] or ch["nonfinite"] or \
                    not np.isfinite(ch["max_abs_err"]):
                raise ValueError(f"case {c}: bad or missing {pas} check {ch}")
        if c not in samples:
            raise ValueError(f"case {c}: sampled outputs missing")
        stride, got = samples[c]
        if stride != sample_stride(p):
            raise ValueError(f"case {c}: sample stride {stride}, expected {sample_stride(p)}")
        ref = refs[pi].ravel()[::stride]
        if got.shape != ref.shape:
            raise ValueError(f"case {c}: {got.size} samples, expected {ref.size}")
        err = np.abs(got.astype(np.float64) - ref)
        if not np.isfinite(got).all() or not (err <= ATOL + RTOL * np.abs(ref)).all():
            raise ValueError(f"case {c} ({k['name']}, N={p['n_ctx']}): sampled outputs differ from float64 "
                             f"numpy by up to {err.max():.3g}")
        med = statistics.median(t)
        flops = attention_flops(p["n_ctx"], p["head_dim"], p["causal"])
        arm_med = statistics.median(arm[pi])
        rows.append(dict(case=c, kernel=k["name"], head_dim=p["head_dim"], n_ctx=p["n_ctx"], causal=p["causal"],
                         block_m=k["block_m"], block_n=k["block_n"], k_layout=k["k_layout"], even_n=k["even_n"],
                         inputs="pattern" if p["inputs"] == PATTERN else "lcg",
                         times_us=t, median_us=med, min_us=min(t),
                         gflops=flops / med / 1e3, gflops_best=flops / min(t) / 1e3,
                         arm_times_us=arm[pi], arm_median_us=arm_med, arm_gflops=flops / arm_med / 1e3,
                         speedup_vs_arm=arm_med / med,
                         board_max_abs_err_vs_arm=checks[(c, "timed")]["max_abs_err"],
                         host_samples=int(got.size), host_max_abs_err_vs_float64=float(err.max())))
    return dict(clocks_mhz=clocks, cases=rows, samples=samples)


def best_rows(rows):
    """The fastest tile per (HEAD_DIM, N_CTX, causal) among the plain LCG cases."""
    best = {}
    for r in rows:
        if r["inputs"] != "lcg" or r["k_layout"] != "nd" or not r["even_n"]:
            continue
        key = (r["head_dim"], r["n_ctx"], r["causal"])
        if key not in best or r["median_us"] < best[key]["median_us"]:
            best[key] = r
    return [best[k] for k in sorted(best)]


def synthetic_log(kernels, problems, cases, repeats, arm_repeats, refs):
    """A complete log as the firmware prints it, from the float64 references
    (for --dry-run and the tests)."""
    lines = ["PIton attention (synthetic)"] + [f"ATTN_CLOCK name={n} mhz={m}" for n, m in
                                            (("arm", 700), ("core", 250), ("v3d", 250), ("sdram", 400))]
    for i, p in enumerate(problems):
        us = int(attention_flops(p["n_ctx"], p["head_dim"], p["causal"]) / 300) + 1
        lines += [f"ATTN_ARM problem={i} us={us + r}" for r in range(arm_repeats)]
        lines.append(f"ATTN_ARM_REPEAT problem={i} differing=0")
    for c, (ki, pi) in enumerate(cases):
        p = problems[pi]
        n_out = p["n_ctx"] * p["head_dim"]
        us = int(attention_flops(p["n_ctx"], p["head_dim"], p["causal"]) / 4000) + 50
        lines.append(f"ATTN_FIRST case={c} us={us + 100}")
        lines.append(f"ATTN_CHECK case={c} pass=first checked={n_out} failures=0 nonfinite=0 max_abs_err_bits=36800000")
        lines += [f"ATTN_TIME case={c} us={us + r}" for r in range(repeats)]
        lines.append(f"ATTN_CHECK case={c} pass=timed checked={n_out} failures=0 nonfinite=0 max_abs_err_bits=36800000")
        words = refs[pi].ravel()[::sample_stride(p)].astype(np.float32).view(np.uint32)
        lines.append(f"ATTN_SAMPLE case={c} stride={sample_stride(p)} count={words.size}")
        lines += [" ".join(f"{w:x}" for w in words[i:i + 8]) + " " for i in range(0, words.size, 8)]
        lines.append(f"ATTN_SAMPLE_END case={c}")
    return "\n".join(lines + ["ATTN_CHECK_OK", "PITON_DONE"]) + "\n"


def pattern_panels(parsed, kernels, problems, cases):
    panels = {}
    for c, (ki, pi) in enumerate(cases):
        p = problems[pi]
        if p["inputs"] == PATTERN:
            panels[p["causal"]] = parsed["samples"][c][1].reshape(p["n_ctx"], p["n_ctx"])
    return [panels[False], panels[True]]


def report(result):
    print(f"{'kernel':24s} {'N':>4s} {'QPU us':>9s} {'GFLOPS':>7s} {'ARM us':>10s} {'speedup':>8s} {'err':>9s}")
    for r in result["cases"]:
        print(f"{r['kernel']:24s} {r['n_ctx']:4d} {r['median_us']:9.0f} {r['gflops']:7.2f} "
              f"{r['arm_median_us']:10.0f} {r['speedup_vs_arm']:7.1f}x {r['host_max_abs_err_vs_float64']:9.2e}")
    print("best tile per problem:")
    for r in result["best"]:
        print(f"  D={r['head_dim']} N={r['n_ctx']:3d} {'causal' if r['causal'] else 'full  '}: "
              f"{r['block_m']}x{r['block_n']} {r['median_us']:.0f} us, {r['gflops']:.2f} GFLOPS, "
              f"{r['speedup_vs_arm']:.1f}x the ARM")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-only", action="store_true", help="compile kernels and firmware, do not boot")
    parser.add_argument("--dry-run", action="store_true",
                        help="--build-only, then parse and report a synthetic log (nothing written to benchmarks/)")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/attention")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--arm-repeats", type=int, default=3)
    parser.add_argument("--tty")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="kernels compiled at a time (default: up to 4)")
    today = f"{datetime.now():%Y-%m-%d}"
    parser.add_argument("--output", type=Path, default=ROOT / f"benchmarks/results/attention-pi-{today}.json")
    parser.add_argument("--image", type=Path, default=ROOT / "docs/images/attention-pi-weights.png")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.arm_repeats < 1:
        parser.error("repeats must be positive")
    runner = None
    if not (args.build_only or args.dry_run):
        try:
            runner = PiRunner(tty=args.tty, timeout=args.timeout, verbose=args.verbose)
        except PiError as ex:
            sys.exit(f"error: {ex}.\nConnect the Pi (PITON_TTY, my-install, CS240LX_2025_PATH) or use "
                     "--build-only / --dry-run.")
    kernels, problems, cases, manifest = build_firmware(args.build_dir.resolve(), args.repeats, args.arm_repeats,
                                                      jobs=args.jobs)
    print(f"Built {manifest['firmware_bytes']:,}-byte firmware: {len(kernels)} kernels, "
          f"{len(problems)} problems, {len(cases)} cases", flush=True)
    refs = references(problems)
    if args.build_only and not args.dry_run:
        return
    if args.dry_run:
        log = synthetic_log(kernels, problems, cases, args.repeats, args.arm_repeats, refs)
        parsed = parse_log(log, kernels, problems, cases, args.repeats, args.arm_repeats, refs)
        result = dict(cases=parsed["cases"], best=best_rows(parsed["cases"]))
        print("synthetic log (made-up timings; outputs are the float64 references):")
        report(result)
        write_png(args.build_dir / "attention-pi-weights-synthetic.png",
                  weights_image(pattern_panels(parsed, kernels, problems, cases)))
        print(f"dry run: synthetic log parsed; would boot {args.build_dir / 'main.bin'}")
        return
    started = time.monotonic()
    try:
        log = runner._boot_recovering(str(args.build_dir.resolve() / "main.bin"))
    except PiError as ex:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".log").write_text(ex.log)
        sys.exit(f"error: the Pi run failed ({ex}); partial log in {args.output.with_suffix('.log')}")
    seconds = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    log_path.write_text(log)
    parsed = parse_log(log, kernels, problems, cases, args.repeats, args.arm_repeats, refs)
    write_png(args.image, weights_image(pattern_panels(parsed, kernels, problems, cases)))
    result = dict(cases=parsed["cases"], best=best_rows(parsed["cases"]), clocks_mhz=parsed["clocks_mhz"],
                  manifest=manifest, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  upload_and_run_seconds=seconds, raw_log=str(log_path.relative_to(ROOT)),
                  image=str(args.image.relative_to(ROOT)) if args.image.is_relative_to(ROOT) else str(args.image),
                  timing_scope="median of the whole launch call (uniform writes, GPU cache clear, queue, wait); "
                               "inputs resident in GPU memory",
                  flop_count="4 N^2 D (two matrix products), halved when causal, as in Triton tutorial 06",
                  arm_baseline="pi/examples/attention/arm_attention.c: straightforward C (scores, max, exp, sum, "
                               "weighted sum of V rows), -O3, data cache on, VFP RunFast, single thread",
                  numerical_check=f"board: every output vs the ARM, |d| <= {ATOL} + {RTOL}|ref|; host: sampled "
                                  "outputs (all of the pattern) vs float64 numpy, same bound")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report(result)
    print(f"Saved {args.output.relative_to(ROOT)}, {log_path.relative_to(ROOT)} and {result['image']} "
          f"({seconds:.0f} s on the board)")


if __name__ == "__main__":
    main()
