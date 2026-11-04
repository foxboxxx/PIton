"""The loop's acceptance gate for one PITON_* switch, in the checkout it runs in.

  1. host tests with the switch unset and with it on
  2. hardware tests (tests/test_hw.py) with it on
  3. perf/ab.py: Pi A/B of every suite case, off vs on

Stops at the first failure. Prints one verdict line and writes the A/B JSON
under perf/results/ (ab-<date>-<label>.json) and a log under perf/loop/.

    .venv/bin/python perf/accept.py PITON_STORE_SLOTS --target copy add exp
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def step(title: str, cmd: list, env: dict, log) -> bool:
    print(f"== {title}", flush=True)
    log.write(f"\n== {title}\n$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(proc.stdout)
    log.flush()
    tail = [line for line in proc.stdout.strip().splitlines() if line.strip()][-3:]
    print("   " + "\n   ".join(tail), flush=True)
    return proc.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env")
    ap.add_argument("--on", default="1")
    ap.add_argument("--target", nargs="*", default=[])
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--skip-host", action="store_true", help="host tests already passed (e.g. by the agent)")
    args = ap.parse_args(argv)
    label = args.env.lower().removeprefix("piton_")
    os.makedirs(os.path.join(ROOT, "perf", "loop"), exist_ok=True)
    path = os.path.join(ROOT, "perf", "loop", f"accept-{datetime.now():%Y%m%d-%H%M}-{label}.log")
    base = {k: v for k, v in os.environ.items() if k != args.env and k != "PITON_HW"}
    on = dict(base, **{args.env: args.on})
    pytest = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    with open(path, "w") as log:
        ok = True
        if not args.skip_host:
            ok = step("host tests, switch unset", pytest + ["tests", "perf"], base, log) and \
                step(f"host tests, {args.env}={args.on}", pytest + ["tests", "perf"], on, log)
        ok = ok and step(f"hardware tests, {args.env}={args.on}", pytest + ["tests/test_hw.py"],
                         dict(on, PITON_HW="1"), log)
        ab = [PY, "perf/ab.py", args.env, "--on", args.on, "--label", label]
        ab += ["--target", *args.target] if args.target else []
        ab += ["--runtime"] if args.runtime else []
        ok = ok and step("Pi A/B", ab, base, log)
    print(f"{'ACCEPT' if ok else 'REJECT'} {args.env}={args.on} (log {os.path.relpath(path, ROOT)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
