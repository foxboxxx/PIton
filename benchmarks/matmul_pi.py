"""Benchmark the Triton tutorial matmul on a Raspberry Pi (all 12 QPUs).

Inputs are generated on the board (bit-exact LCG shared with the host), the
result is checked on a sample of C entries against a float64 reference.

    python3 benchmarks/matmul_pi.py 256 512 --bm 8 --bn 64 --bk 4
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TILES = ((8, 64, 8), (4, 64, 8), (4, 64, 16), (4, 128, 4),
                 (4, 128, 8), (16, 32, 4), (8, 32, 4), (8, 32, 8))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton  # noqa: E402
import piton.compiler as pc  # noqa: E402
from piton.runtime.pi import bench_matmul_on_pi  # noqa: E402
from kernels import matmul_kernel  # noqa: E402


def compile_matmul(n, bm, bn, bk, group=4, stages=2, specialize_layout=False):
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "M": "i32", "N": "i32",
           "stride_am": "i32", "stride_bk": "i32", "stride_cm": "i32"}
    cons = dict(K=n, stride_ak=1, stride_bn=1, stride_cn=1, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk, GROUP_SIZE_M=group)
    if specialize_layout:
        cons.update(M=n, N=n, stride_am=n, stride_bk=n, stride_cm=n)
    grid = (piton.cdiv(n, bm) * piton.cdiv(n, bn),)
    ck = pc.compile(matmul_kernel, sig, cons, num_stages=stages, grid=grid)
    return ck, grid


def _run_bench(ck, grid, n, cfg, repeats, max_error):
    r = bench_matmul_on_pi(ck, n, n, n, grid,
                           {"M": n, "N": n, "stride_am": n, "stride_bk": n, "stride_cm": n},
                           repeats=repeats)
    r.update(n=n, cfg=cfg, instructions=len(ck.program.instrs))
    validate_result(r, repeats, max_error)
    return r


def validate_result(result, repeats, max_error):
    """Gate sampled final outputs and require a complete, finite timing set."""
    times = result.get("times_us", [])
    valid_times = (len(times) == repeats and
                   all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times))
    err = result.get("max_err", math.inf)
    correct = isinstance(err, (int, float)) and math.isfinite(err) and 0 <= err <= max_error
    result.update(correct=correct, valid_timings=valid_times, max_error_allowed=max_error,
                  validation="sampled final repetition, absolute error vs float64",
                  sample_stride=97, seeds=[12345, 67890])
    if valid_times:
        result["median_us"] = statistics.median(times)
        result["median_gflops"] = 2.0 * result["n"] ** 3 / (result["median_us"] * 1e3)
    return correct and valid_times


def bench(n, bm, bn, bk, group=4, stages=2, repeats=3, max_error=None, specialize_layout=False):
    ck, grid = compile_matmul(n, bm, bn, bk, group, stages, specialize_layout)
    result = _run_bench(ck, grid, n, (bm, bn, bk, group, stages), repeats,
                        n * 1e-5 if max_error is None else max_error)
    result["specialize_layout"] = specialize_layout
    return result


def source_identity():
    """Hash compiler, kernel, runtime and harness contents, including dirty edits."""
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    sources = list((ROOT / "piton").rglob("*.py"))
    sources += list((ROOT / "pi" / "runtime").glob("*.[ch]"))
    sources += [ROOT / "examples" / "kernels.py", Path(__file__).resolve(),
                ROOT / "benchmarks" / "sweep_matmul.py"]
    digest = hashlib.sha256()
    for path in sorted(set(sources)):
        if path.exists():
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return {"revision": revision, "source_sha256": digest.hexdigest()}


def case_key(n, cfg, repeats, max_error, identity, specialize_layout=False):
    settings = dict(n=n, cfg=cfg, repeats=repeats, max_error=max_error, compiler=identity,
                    specialize_layout=specialize_layout)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_completed(path, retry_errors=False):
    """Ignore an interrupted final JSONL write; reject corruption in earlier rows."""
    if not path or not Path(path).exists():
        return set()
    lines = Path(path).read_text().splitlines()
    completed = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"Ignoring incomplete final result in {path}", file=sys.stderr)
                continue
            raise ValueError(f"invalid JSON in {path}:{i + 1}") from None
        status = row.get("status")
        if status != "interrupted" and (not retry_errors or status == "ok"):
            if row.get("case_key"):
                completed.add(row["case_key"])
    return completed


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def append_result(path, result):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove only a partial final line left by a killed write before appending.
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as existing:
            data = existing.read()
            if not data.endswith(b"\n"):
                tail = data.rsplit(b"\n", 1)[-1]
                try:
                    json.loads(tail)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    existing.truncate(len(data) - len(tail))
                else:
                    existing.write(b"\n")
    with path.open("a") as out:
        out.write(json.dumps(_json_safe(result), sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def measure_case(n, cfg, repeats, max_error, identity, specialize_layout=False):
    """Return a durable record even when compilation or execution fails."""
    row = dict(schema=1, timestamp=datetime.now(timezone.utc).isoformat(),
               case_key=case_key(n, cfg, repeats, max_error, identity, specialize_layout), compiler=identity,
               n=n, cfg=cfg, repeats=repeats, max_error_allowed=max_error,
               specialize_layout=specialize_layout)
    start = time.monotonic()
    phase = "compile"
    try:
        ck, grid = compile_matmul(n, *cfg, specialize_layout=specialize_layout)
        row.update(instructions=len(ck.program.instrs),
                   program_sha256=hashlib.sha256(b"".join(int(w).to_bytes(8, "little")
                                                        for w in ck.words)).hexdigest())
        phase = "run"
        row.update(_run_bench(ck, grid, n, cfg, repeats, max_error))
        row["status"] = "ok" if row["correct"] and row["valid_timings"] else "incorrect"
    except KeyboardInterrupt:
        row.update(status="interrupted", phase=phase, error_type="KeyboardInterrupt",
                   error="interrupted by user")
    except Exception as exc:
        row.update(status=f"{phase}_error", phase=phase,
                   error_type=type(exc).__name__, error=str(exc))
    row["wall_seconds"] = time.monotonic() - start
    return row


def _positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def _tile(value):
    try:
        tile = tuple(_positive_int(n) for n in value.lower().replace(",", "x").split("x"))
        if len(tile) != 3:
            raise ValueError
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("expected BMxBNxBK, e.g. 8x64x8") from None
    return tile


def main(argv=None, sweep=False):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sizes", nargs="*", type=_positive_int, default=[512, 1024] if sweep else [256])
    ap.add_argument("--bm", type=_positive_int, default=8)
    ap.add_argument("--bn", type=_positive_int, default=64)
    ap.add_argument("--bk", type=_positive_int, default=4)
    ap.add_argument("--group", type=_positive_int, default=4)
    ap.add_argument("--stages", type=_positive_int, default=2)
    ap.add_argument("--repeats", type=_positive_int, default=3)
    ap.add_argument("--specialize-layout", action="store_true",
                    help="specialize square dimensions and contiguous strides at compile time")
    ap.add_argument("--max-error", type=float,
                    help="maximum sampled absolute error vs float64 (default: 1e-5 * K)")
    ap.add_argument("--jsonl", default="benchmarks/results/matmul-sweep.jsonl" if sweep else None,
                    help="append each success, failure and interruption to this file")
    ap.add_argument("--resume", action="store_true", default=sweep,
                    help="skip completed cases with matching source and settings")
    ap.add_argument("--retry-errors", action="store_true",
                    help="when resuming, rerun failed/incorrect cases as well")
    if sweep:
        ap.add_argument("--tiles", nargs="+", type=_tile, default=DEFAULT_TILES,
                        help="tile sizes BMxBNxBK (default: previous winner + unfinished sweep)")
        ap.add_argument("--groups", nargs="+", type=_positive_int,
                        help="override --group with multiple values")
        ap.add_argument("--stage-counts", nargs="+", type=_positive_int,
                        help="override --stages with multiple values")
    a = ap.parse_args(argv)
    if a.max_error is not None and (not math.isfinite(a.max_error) or a.max_error < 0):
        ap.error("--max-error must be finite and nonnegative")
    if a.resume and not a.jsonl:
        ap.error("--resume needs --jsonl")
    identity = source_identity()
    completed = load_completed(a.jsonl, a.retry_errors) if a.resume else set()
    tiles = a.tiles if sweep else [(a.bm, a.bn, a.bk)]
    groups = (a.groups or [a.group]) if sweep else [a.group]
    stages = (a.stage_counts or [a.stages]) if sweep else [a.stages]
    failures = 0
    for tile, group, stage, n in itertools.product(tiles, groups, stages, a.sizes):
        cfg = (*tile, group, stage)
        max_error = n * 1e-5 if a.max_error is None else a.max_error
        key = case_key(n, cfg, a.repeats, max_error, identity, a.specialize_layout)
        if key in completed:
            print(f"N={n:5d} cfg={cfg}  skipped (already recorded)", flush=True)
            continue
        print(f"N={n:5d} cfg={cfg}  compiling and measuring...", flush=True)
        row = measure_case(n, cfg, a.repeats, max_error, identity, a.specialize_layout)
        append_result(a.jsonl, row)
        if row["status"] in ("ok", "incorrect"):
            print(f"N={n:5d} cfg(BM,BN,BK,G,stages)={cfg}  best {row['best_us']:8d} us  "
                  f"{row['gflops']:.3f} GFLOPS  max sampled err {row['max_err']:.2e}  "
                  f"({row['programs']} programs)  {row['status']}  times_us={row['times_us']}", flush=True)
        else:
            print(f"N={n:5d} cfg={cfg}  {row['status']}: "
                  f"{row['error_type']}: {row['error']}", flush=True)
        if row["status"] == "interrupted":
            return 130
        failures += row["status"] != "ok"
        completed.add(key)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
