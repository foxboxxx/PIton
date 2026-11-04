"""Pi sweep behind compiler.COARSEN_LAUNCH_MIN_GRID / _PROGRAMS: one
PITON_COARSEN_RUNTIME build of a suite case launched on smaller grids with
every factor k (1, 2, 4 up to the build's), one boot each, two rounds in
alternating order, best of five launches per boot; inputs are generated on
the board as in perf/profile_pi.py and the computed prefix is checked.
Appends JSON lines and skips points already in the file.

    PITON_COARSEN_RUNTIME=1 .venv/bin/python perf/coarsen_sweep.py perf/results/coarsen-launch-<date>.jsonl
"""
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf import suite                      # noqa: E402
from perf.profile_pi import EVENTS, V3D_MHZ, buffer_for, sample_stride   # noqa: E402
from piton.qpu.isa import f2u               # noqa: E402
from piton.runtime.launcher import launch_coarsen, program_ids   # noqa: E402
from piton.runtime.pi import Buffer, PiRunner   # noqa: E402

if os.environ.get("PITON_COARSEN_RUNTIME") != "1":
    sys.exit("set PITON_COARSEN_RUNTIME=1: the sweep chooses k per launch")
out_path = sys.argv[1]
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        r = json.loads(line)
        done.add((r["case"], r["gx"], r["k"], r["rep"]))

CASES = {c.name: c for c in suite.CASES}
PLAN = {"add": [32, 64, 128, 256, 512, 1024], "dense_relu": [2, 8, 32, 64, 128, 256, 450, 900]}


def forced(ck, gx, k):
    """ck with a launch rule that picks exactly k for gx."""
    p = {1: gx + 1, 2: gx // ck.coarsen + 1 if ck.coarsen > 2 else 1, 4: 1}[k]
    v = dataclasses.replace(ck, coarsen_launch=(0, max(1, p)))
    assert launch_coarsen(v, (gx,)) == k, (gx, k, p)
    return v


def launch(ck, run, gx, k, runner):
    """perf/profile_pi.run_case on grid (gx,) with factor k: inputs generated on the board."""
    values = run.inputs.values
    buffers = [buffer_for(n, run.inputs.specs[n]) for n in run.inputs.specs]
    buffers += [Buffer("__zero", 64), Buffer("__scratch", 64)]
    programs = []
    for pid, g in program_ids((gx,), k):
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
            elif kind == "coarsen":
                u.append(k)
            elif kind in ("zero", "scratch"):
                u.append(("buf", f"__{kind}", 0))
            elif kind == "self":
                u.append(("self",))
        programs.append(u)
    outputs = list(run.expected)
    strides = {n: sample_stride(run.inputs.specs[n].count) for n in outputs}
    res = runner.run(ck.words, buffers, programs, outputs, repeats=5, label=ck.name,
                     sample=strides, perf_events=list(EVENTS.values()))
    errs = {}
    for n in outputs:
        count = run.inputs.specs[n].count
        idx = np.arange(0, count, strides[n])
        got = res.outputs[n][:idx.size].view(run.inputs.specs[n].dtype).astype(np.float64)
        want = np.asarray(run.expected[n], dtype=np.float64).ravel()[idx]
        live = idx < gx * (count // run.grid[0])
        errs[n] = float(np.max(np.abs(got[live] - want[live]))) if live.any() else 0.0
    best = int(np.argmin(res.times_us))
    t = res.times_us[best]
    counters = dict(zip(EVENTS, res.perf[best])) if res.perf else {}
    return dict(times_us=res.times_us, best_us=t, max_err=errs,
                busy=counters.get("valid", 0) / (12 * V3D_MHZ * t), idle=counters.get("idle", 0) / (12 * V3D_MHZ * t))


runner = PiRunner()
for name, grids in PLAN.items():
    case = CASES[name]
    ck, run = suite.compile_case(case, "pi")
    assert ck.coarsen_launch is not None and ("coarsen", "coarsen0") in ck.uniforms, ck
    ks = [k for k in (1, 2, 4) if k <= ck.coarsen]
    for rep in (1, 2):
        for gx in grids:
            order = ks if rep == 1 else ks[::-1]
            for k in order:
                if (name, gx, k, rep) in done:
                    continue
                r = launch(forced(ck, gx, k), run, gx, k, runner)
                row = dict(case=name, gx=gx, k=k, rep=rep, programs=-(-gx // k), coarsen_max=ck.coarsen,
                           **r)
                print(json.dumps(row), flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
