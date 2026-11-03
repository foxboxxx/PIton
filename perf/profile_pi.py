"""Run every suite case on the attached Pi with the V3D performance counters.

One boot per case. Inputs are generated on the board from the LCG specs in
perf/suite.py, so nothing large crosses the UART; sampled outputs are checked
against the same numpy reference as perf/test_perf_suite.py. For each case it
reports the best launch time and, from the counters of that launch:

  busy   QPU cycles executing instructions / (12 QPUs x V3D clock x time)
  tmu    cycles stalled waiting for TMU (memory) results
  idle   QPU cycles with no program (launch gaps, tail imbalance)
  icache instruction cache miss rate; tmu/l2 miss counts per launch

High busy means the kernel is instruction bound (compiler work pays off);
high tmu or idle means memory or launch bound.

    .venv/bin/python perf/profile_pi.py                # every case
    .venv/bin/python perf/profile_pi.py exp matmul_8x64x8
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf.suite import CASES, CASES_BY_NAME, compile_case, default_check  # noqa: E402
from piton.qpu.isa import f2u  # noqa: E402
from piton.runtime.launcher import program_ids  # noqa: E402
from piton.runtime.pi import Buffer, PiRunner  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EVENTS = {"idle": 13, "valid": 16, "tmu_stall": 17, "sb_stall": 18, "icache_hit": 20, "icache_miss": 21,
          "unif_miss": 23, "tmu_quads": 24, "tmu_miss": 25, "vdw_stall": 26, "l2_hit": 28, "l2_miss": 29}
V3D_MHZ = 250
MAX_SAMPLES = 1024


def c_float(x) -> str:
    return float(np.float32(x)).hex() + "f"


def buffer_for(name, spec) -> Buffer:
    nbytes = 4 * spec.count
    if spec.kind == "float":
        expr = f"v.f = piton_lcg_float(&s) * {c_float(spec.scale)} + {c_float(spec.offset)};"
    elif spec.kind == "int":
        expr = f"v.u = (uint32_t)(((int32_t)piton_lcg(&s) >> {spec.shift}) | {spec.or_mask});"
    elif spec.kind == "ramp":                       # identity palette, entry 0 = spec.fill
        expr = f"v.u = i ? i : {spec.fill}u;"
    else:
        expr = f"v.u = 0x{spec.fill:08x}u;"
    return Buffer(name, nbytes, lcg_seed=spec.seed, c_init=expr)


def sample_stride(count: int) -> int:
    stride = max(1, -(-count // MAX_SAMPLES))
    return stride if stride == 1 else stride | 1          # odd: avoids aliasing with tile strides


def run_case(case, repeats: int, runner: PiRunner) -> dict:
    ck, run = compile_case(case, "pi")
    values = run.inputs.values
    buffers = [buffer_for(n, run.inputs.specs[n]) for n in run.inputs.specs]
    buffers += [Buffer("__zero", 64), Buffer("__scratch", 64)]
    programs = []
    for pid, g in program_ids(run.grid):
        u = []
        for kind, name in ck.uniforms:
            if kind == "arg":
                v = values[name]
                if isinstance(v, np.ndarray):
                    u.append(("buf", name, 0))
                elif isinstance(v, (float, np.floating)):
                    u.append(f2u(float(v)))
                else:
                    u.append(int(v) & 0xFFFFFFFF)
            elif kind == "pid":
                u.append(pid[int(name[3:])])
            elif kind == "nprog":
                u.append(g[int(name[5:])])
            elif kind in ("zero", "scratch"):
                u.append(("buf", f"__{kind}", 0))
            elif kind == "self":
                u.append(("self",))
        programs.append(u)
    outputs = list(run.expected)
    strides = {n: sample_stride(run.inputs.specs[n].count) for n in outputs}
    res = runner.run(ck.words, buffers, programs, outputs, repeats=repeats, label=case.name,
                     sample=strides, perf_events=list(EVENTS.values()))
    errors = {}
    for name in outputs:
        idx = np.arange(0, run.inputs.specs[name].count, strides[name])
        got = res.outputs[name][:idx.size].view(run.inputs.specs[name].dtype)
        err = default_check(run, name, got, np.asarray(run.expected[name]).ravel()[idx])
        if err:
            errors[name] = err
    best = int(np.argmin(res.times_us))
    t = res.times_us[best]
    counters = dict(zip(EVENTS, res.perf[best])) if res.perf else {}
    capacity = 12 * V3D_MHZ * t
    fetches = counters.get("icache_hit", 0) + counters.get("icache_miss", 0)
    return dict(name=case.name, note=case.note, instructions=len(ck.program.instrs), programs=len(programs),
                times_us=res.times_us, best_us=t, counters=counters,
                busy=counters.get("valid", 0) / capacity, tmu=counters.get("tmu_stall", 0) / capacity,
                idle=counters.get("idle", 0) / capacity,
                icache_miss=counters.get("icache_miss", 0) / fetches if fetches else 0.0,
                checked={n: int(-(-run.inputs.specs[n].count // strides[n])) for n in outputs},
                errors=errors)


def table(rows) -> str:
    head = (f"{'case':16} {'instr':>6} {'progs':>6} {'best_us':>9}  {'busy':>6} {'tmu':>6} {'idle':>6} "
            f"{'icache':>7} {'tmu_miss':>9} {'l2_miss':>8}  check")
    out = [head, "-" * len(head)]
    for r in rows:
        if "failed" in r:
            out.append(f"{r['name']:16} FAILED: {r['failed'].splitlines()[0][:80]}")
            continue
        c = r["counters"]
        out.append(f"{r['name']:16} {r['instructions']:6} {r['programs']:6} {r['best_us']:9}  "
                   f"{100 * r['busy']:5.1f}% {100 * r['tmu']:5.1f}% {100 * r['idle']:5.1f}% "
                   f"{100 * r['icache_miss']:6.2f}% {c.get('tmu_miss', 0):9} {c.get('l2_miss', 0):8}  "
                   f"{'ok' if not r['errors'] else 'WRONG ' + '; '.join(r['errors'].values())[:60]}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--label", default="", help="suffix for the results file, e.g. 'before'")
    args = ap.parse_args(argv)
    cases = [CASES_BY_NAME[n] for n in args.cases] if args.cases else CASES
    runner = PiRunner(timeout=300)
    rows = []
    for case in cases:
        start = time.time()
        print(f"{case.name} ...", end=" ", file=sys.stderr, flush=True)
        try:
            rows.append(run_case(case, args.repeats, runner))
            print(f"{rows[-1]['best_us']} us ({time.time() - start:.0f} s)", file=sys.stderr, flush=True)
        except Exception as ex:                      # keep going: one bad case should not stop the suite
            rows.append(dict(name=case.name, failed=f"{type(ex).__name__}: {ex}"))
            print("FAILED", file=sys.stderr, flush=True)
    print(table(rows))
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"pi-{date.today().isoformat()}{'-' + args.label if args.label else ''}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
