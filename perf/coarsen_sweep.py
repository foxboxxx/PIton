"""Pi sweep behind compiler.COARSEN_LAUNCH_*: one PITON_COARSEN_RUNTIME build
of a kernel launched on smaller grids with every factor k (1, 2, 4 up to the
build's), one boot each, two rounds in alternating order, best of five launches
per boot. Inputs are generated on the board as in perf/profile_pi.py; the
outputs of the launched ids are checked against numpy and every other output
must be left as the NaN fill. Appends JSON lines and skips points already in
the file.

    PITON_COARSEN_RUNTIME=1 .venv/bin/python perf/coarsen_sweep.py perf/results/coarsen-launch-<date>.jsonl
"""
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.nn_kernels import dense_layer_kernel, dense_tiles   # noqa: E402
from perf import suite                      # noqa: E402
from perf.profile_pi import EVENTS, V3D_MHZ, buffer_for, sample_stride   # noqa: E402
import piton.compiler                       # noqa: E402
from piton.qpu.isa import f2u               # noqa: E402
from piton.runtime.launcher import launch_coarsen, program_ids   # noqa: E402
from piton.runtime.pi import Buffer, PiRunner   # noqa: E402


def digits_dense(layer):
    """examples/digits_pi.py's layer 2 (128 -> 64, ReLU) or 3 (64 -> 16) on the full batch."""
    sig = {"x_ptr": suite.PTR, "w_ptr": suite.PTR, "bias_ptr": suite.PTR, "y_ptr": suite.PTR, "M": "i32"}
    n_in, n_out, m = {2: (128, 64), 3: (64, 16)}[layer] + (1800,)
    bm, bn, bk = dense_tiles(n_out)
    relu = layer < 3

    def make(size):
        g = suite.Inputs()
        x = g.floats("x_ptr", m * n_in, 0.0, 1.0).reshape(m, n_in).astype(np.float64)
        w = g.floats("w_ptr", n_in * n_out, -0.3, 0.3).reshape(n_in, n_out).astype(np.float64)
        b = g.floats("bias_ptr", n_out, -0.1, 0.1).astype(np.float64)
        g.output("y_ptr", m * n_out)
        g.scalar("M", m)
        y = x @ w + b
        return suite.Run(g, ((m // bm) * (n_out // bn),), {"y_ptr": (np.maximum(y, 0) if relu else y).ravel()},
                         1e-4, 1e-4)
    cons = dict(N=n_out, K=n_in, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                ADD_BIAS=True, RELU=relu)
    return suite.Case(f"digits_dense{layer}", dense_layer_kernel, sig, cons, make, f"digits layer {layer}", ("dot",))


CASES = dict(suite.CASES_BY_NAME, digits_dense2=digits_dense(2), digits_dense3=digits_dense(3))
# (case, grid the build is compiled for (None: the suite's), launched grids)
PLAN = [("add", None, [32, 64, 128, 256, 512, 1024]),
        ("dense_relu", None, [2, 8, 32, 64, 128, 256, 450, 900]),
        ("digits_dense2", None, [16, 32, 64, 128, 225, 450]),
        ("digits_dense3", None, [16, 32, 64, 128, 225]),
        ("matmul_8x64x8", 512, [32, 64, 128]),
        ("row_sum", 256, [16, 32, 64])]


def covered(case, run, gx, name):
    """Mask of the output elements ids 0 ... gx-1 write."""
    count = run.inputs.specs[name].count
    c = case.constants
    if case.kernel is suite.matmul_kernel and c["GROUP_SIZE_M"] > 1:      # tutorial 03's grouped order
        n = run.inputs.values["N"]
        m = count // n
        bm, bn, group = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["GROUP_SIZE_M"]
        num_m, num_n = -(-m // bm), -(-n // bn)
        mask = np.zeros((m, n), bool)
        for pid in range(gx):
            first = pid // (group * num_n) * group
            size = min(num_m - first, group)
            pm, pn = first + (pid % (group * num_n)) % size, (pid % (group * num_n)) // size
            mask[pm * bm:(pm + 1) * bm, pn * bn:(pn + 1) * bn] = True
        return mask.ravel()
    return np.arange(count) < gx * (count // run.grid[0])


def forced(ck, gx, k):
    """ck with a launch rule that picks exactly k for gx."""
    p = {1: gx + 1, 2: gx // ck.coarsen + 1 if ck.coarsen > 2 else 1, 4: 1}[k]
    v = dataclasses.replace(ck, coarsen_launch=(0, max(1, p)))
    assert launch_coarsen(v, (gx,)) == k, (gx, k, p)
    return v


def launch(case, ck, run, gx, k, runner):
    """perf/profile_pi.run_case on grid (gx,) with factor k."""
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
    errs, untouched = {}, True
    for n in outputs:
        idx = np.arange(0, run.inputs.specs[n].count, strides[n])
        got = res.outputs[n][:idx.size].view(run.inputs.specs[n].dtype).astype(np.float64)
        want = np.asarray(run.expected[n], dtype=np.float64).ravel()[idx]
        live = covered(case, run, gx, n)[idx]
        errs[n] = float(np.max(np.abs(got[live] - want[live]))) if live.any() else 0.0
        untouched = untouched and bool(np.all(np.isnan(got[~live])))
    best = int(np.argmin(res.times_us))
    t = res.times_us[best]
    counters = dict(zip(EVENTS, res.perf[best])) if res.perf else {}
    return dict(times_us=res.times_us, best_us=t, max_err=errs, rest_untouched=untouched,
                busy=counters.get("valid", 0) / (12 * V3D_MHZ * t), idle=counters.get("idle", 0) / (12 * V3D_MHZ * t))


def main():
    if os.environ.get("PITON_COARSEN_RUNTIME") != "1":
        sys.exit("set PITON_COARSEN_RUNTIME=1: the sweep chooses k per launch")
    out_path = sys.argv[1]
    only = set(sys.argv[2:])
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["case"], r.get("bound"), r["gx"], r["k"], r["rep"]))
    runner = PiRunner()
    for name, bound, grids in PLAN:
        if only and name not in only:
            continue
        case = CASES[name]
        run = case.make("pi")
        if bound is None:
            ck, _ = suite.compile_case(case, "pi", run)
        else:
            ck = piton.compiler.compile(case.kernel, case.signature, suite.constants_for(case, "pi"),
                                        grid=(bound,), **case.options)
        assert ck.coarsen_launch is not None and ("coarsen", "coarsen0") in ck.uniforms, ck
        ks = [k for k in (1, 2, 4) if k <= ck.coarsen]
        for rep in (1, 2):
            for gx in grids:
                for k in (ks if rep == 1 else ks[::-1]):
                    if (name, bound, gx, k, rep) in done:
                        continue
                    r = launch(case, forced(ck, gx, k), run, gx, k, runner)
                    row = dict(case=name, bound=bound, gx=gx, k=k, rep=rep, programs=-(-gx // k),
                               coarsen_max=ck.coarsen, **r)
                    print(json.dumps(row), flush=True)
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
