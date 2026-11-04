"""A/B one PITON_* switch on the Pi: the same build with the switch off and on.

Every case is compiled both ways. Cases whose machine code and uniforms come
out identical are not run (ratio 1.000) unless --runtime says the switch acts
on the Pi side (launcher, generated C). The rest boot alternately off, on, on,
off, ... (--rounds pairs), best of --repeats launches per boot, sampled outputs
checked as in profile_pi.py; a case outside +-2% after --rounds gets more
pairs, up to --max-rounds. A hung or failing variant counts as a failure
(the runner power-cycles the board and goes on).

Verdict, as for the overnight loop: accept when every result is correct, some
target case is more than 2% faster, and no case is more than 2% slower.

    .venv/bin/python perf/ab.py PITON_PRESCHED
    .venv/bin/python perf/ab.py PITON_STORE_SLOTS --target copy add exp --off 0 --on 1
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf.profile_pi import RESULTS, run_case  # noqa: E402
from perf.suite import CASES, CASES_BY_NAME, compile_case  # noqa: E402
from piton.runtime.pi import PiRunner  # noqa: E402

THRESHOLD = 0.02


@contextmanager
def switch(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


def verdict(rows, targets) -> tuple[bool, str]:
    bad = [r["name"] for r in rows if r.get("failed") or r.get("errors")]
    if bad:
        return False, "wrong or failed: " + ", ".join(bad)
    slower = [f"{r['name']} {r['speedup']:.3f}x" for r in rows if r["speedup"] < 1 - THRESHOLD]
    if slower:
        return False, "slower: " + ", ".join(slower)
    faster = [r for r in rows if (not targets or r["name"] in targets) and r["speedup"] > 1 + THRESHOLD]
    if not faster:
        return False, "no target case more than 2% faster"
    return True, "faster: " + ", ".join(f"{r['name']} {r['speedup']:.3f}x" for r in faster)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env", help="the switch, e.g. PITON_PRESCHED")
    ap.add_argument("cases", nargs="*", help="cases to run (default: all)")
    ap.add_argument("--off", default="0")
    ap.add_argument("--on", default="1")
    ap.add_argument("--target", nargs="*", default=[], help="cases the change is meant to speed up")
    ap.add_argument("--runtime", action="store_true", help="the switch acts on the Pi side: run every case")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1, help="off/on pairs per case")
    ap.add_argument("--max-rounds", type=int, default=3, help="pairs when a case is outside the threshold")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    cases = [CASES_BY_NAME[n] for n in args.cases] if args.cases else CASES
    runner = PiRunner(timeout=240)
    rows = []
    for case in cases:
        compiled = {}
        try:
            for side, value in (("off", args.off), ("on", args.on)):
                with switch(args.env, value):
                    compiled[side] = compile_case(case, "pi")
        except Exception as ex:                      # the switch broke compilation: a failure
            rows.append(dict(name=case.name, instructions_off=0, instructions_on=0, speedup=0.0,
                             failed=f"compile {side}: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}"))
            print(f"{case.name}: {rows[-1]['failed']}", file=sys.stderr, flush=True)
            continue
        ck_off, ck_on = compiled["off"][0], compiled["on"][0]
        row = dict(name=case.name, instructions_off=len(ck_off.program.instrs),
                   instructions_on=len(ck_on.program.instrs))
        if not args.runtime and list(ck_off.words) == list(ck_on.words) and ck_off.uniforms == ck_on.uniforms:
            row.update(same=True, speedup=1.0)
            rows.append(row)
            print(f"{case.name}: same code", file=sys.stderr, flush=True)
            continue
        best = {"off": math.inf, "on": math.inf}
        runs = {"off": [], "on": []}
        start = time.time()
        # Boot-to-boot differences reach 5% for a few cases (sqrt, row_sum:
        # perf/loop/baseline-*.log) while one boot's repeats agree within 1%,
        # so a result outside the threshold gets more off/on pairs (alternating
        # order) before it counts.
        for r in range(args.max_rounds):
            if r >= args.rounds and (row.get("failed") or row.get("errors") or
                                     abs(best["off"] / best["on"] - 1) <= THRESHOLD):
                break
            for side in (["off", "on"] if r % 2 == 0 else ["on", "off"]):
                value = args.off if side == "off" else args.on
                try:
                    with switch(args.env, value):
                        res = run_case(case, args.repeats, runner, compiled[side])
                except Exception as ex:              # a hang or crash: the variant failed
                    row["failed"] = f"{side}: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}"
                    break
                runs[side].append(res)
                best[side] = min(best[side], res["best_us"])
                if res["errors"]:
                    row.setdefault("errors", {})[side] = res["errors"]
            if row.get("failed"):
                break
        row.update(off_us=best["off"], on_us=best["on"],
                   speedup=best["off"] / best["on"] if not row.get("failed") else 0.0,
                   off=[dict(times_us=x["times_us"], busy=x["busy"], tmu=x["tmu"], idle=x["idle"],
                             icache_miss=x["icache_miss"]) for x in runs["off"]],
                   on=[dict(times_us=x["times_us"], busy=x["busy"], tmu=x["tmu"], idle=x["idle"],
                            icache_miss=x["icache_miss"]) for x in runs["on"]])
        rows.append(row)
        status = row.get("failed") or ("WRONG" if row.get("errors") else f"{row['speedup']:.3f}x")
        print(f"{case.name}: {best['off']} -> {best['on']} us  {status}  ({time.time() - start:.0f} s)",
              file=sys.stderr, flush=True)
    ok, why = verdict(rows, set(args.target))
    print(f"\n{'case':16} {'instr off/on':>13} {'off_us':>8} {'on_us':>8} {'speedup':>8}")
    for r in rows:
        if r.get("same"):
            print(f"{r['name']:16} {r['instructions_off']:>6}/{r['instructions_on']:<6} {'':>8} {'':>8}    same")
        elif r.get("failed"):
            print(f"{r['name']:16} {r['instructions_off']:>6}/{r['instructions_on']:<6} FAILED {r['failed'][:60]}")
        else:
            flag = "  WRONG" if r.get("errors") else ""
            print(f"{r['name']:16} {r['instructions_off']:>6}/{r['instructions_on']:<6} "
                  f"{r['off_us']:>8} {r['on_us']:>8} {r['speedup']:>7.3f}x{flag}")
    print(f"\nverdict: {'ACCEPT' if ok else 'REJECT'} ({why})")
    os.makedirs(RESULTS, exist_ok=True)
    label = args.label or args.env.lower().removeprefix("piton_")
    path = os.path.join(RESULTS, f"ab-{date.today().isoformat()}-{label}.json")
    with open(path, "w") as f:
        json.dump(dict(env=args.env, off=args.off, on=args.on, targets=args.target, accept=ok, why=why,
                       rows=rows), f, indent=1)
    print(f"wrote {path}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
