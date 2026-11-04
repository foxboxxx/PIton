"""Where do the instructions go? Dynamic instruction-mix profile of the suite.

Every case runs once in the strict emulator with per-pc execution counts, so
each instruction is weighted by how often it really executes. For each kernel
(and its hottest code region) this reports:

  paired      add and mul ALU both doing work
  add / mul   only one ALU busy
  nop         neither ALU busy (split into branch-delay, thread-end and other)
  ldi / br    load-immediate and branch instructions (no ALU work)
  mov         ALU ops that only copy a value
  bound       max(add ops, mul ops) + ldi + branches: the instruction count if
              every add/mul could pair; executed / bound is the packing loss

For every add-only instruction next to a mul-only one (or the reverse) it
says why the two could not share an instruction: a data dependency, a
latency (the second reads a register file location, or rotates an
accumulator, that the instruction before the first wrote), both needing a
different register in file A (or B: a small immediate or a mul-output
rotation takes the B read port), conflicting write files, two peripheral
accesses (TMU / SFU writes, TMU loads), two signals or two flag writes.
"free" means none of these applied.

    .venv/bin/python perf/analyze.py                 # every case
    .venv/bin/python perf/analyze.py matmul_8x64x8 -v  # one case, hottest region listed
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf.suite import CASES, CASES_BY_NAME, compile_case  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.emulator import V3D  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _busy(op) -> bool:
    return op is not None and op.op != "nop"


def classify(instrs) -> list:
    """Per-instruction class name."""
    out, delay, end = [], 0, 0
    for ins in instrs:
        if ins.kind == "branch":
            cls = "branch"
        elif ins.kind in ("ldi", "sem"):
            cls = "ldi"
        else:
            add, mul = _busy(ins.add), _busy(ins.mul)
            if add and mul:
                cls = "paired"
            elif add:
                cls = "add"
            elif mul:
                cls = "mul"
            elif delay:
                cls = "nop_delay"
            elif end:
                cls = "nop_end"
            elif ins.sig != "none":
                cls = "signal"
            else:
                cls = "nop_other"
        out.append(cls)
        delay = 3 if ins.kind == "branch" else max(delay - 1, 0)
        end = 2 if ins.kind == "alu" and ins.sig == "thrend" else max(end - 1, 0)
    return out


def _reads(op) -> list:
    return [s for s in op.srcs() if s is not None] if op is not None else []


def _port_conflict(ops) -> str | None:
    """Can these ALU ops' sources share one instruction's two read ports?"""
    a, b, either = set(), set(), set()
    for op in ops:
        if op.rot is not None:
            b.add(("rot", op.rot))                          # the rotation code is raddr_b
        for s in _reads(op):
            if isinstance(s, isa.Imm):
                b.add(("imm", s.value if hasattr(s, "value") else repr(s)))
            elif s.acc is not None and s.rfile is None:
                continue                                    # accumulator: no port
            elif s.rfile == "A":
                a.add(s.raddr)
            elif s.rfile == "B":
                b.add(s.raddr)
            elif s.rfile == "AB":
                either.add(s.raddr)
    for r in either:                                        # specials fit either port
        if not a or a == {r}:
            a.add(r)
        elif not b or b == {r}:
            b.add(r)
        else:
            return "port"
    if len(a) > 1:
        return "port_A"
    if len(b) > 1:
        return "port_B"
    return None


def _periph(ins) -> int:
    """TMU / SFU writes and TMU loads: one per instruction."""
    n = int(ins.sig in ("ldtmu0", "ldtmu1"))
    for op in (ins.add, ins.mul):
        n += _busy(op) and op.dst.waddr is not None and 52 <= op.dst.waddr <= 63
    return n


def _written(ins) -> set:
    return {op.dst for op in (ins.add, ins.mul) if ins.kind != "branch" and _busy(op) and op.dst != isa.NOP}


def pair_blocker(x, y, before=None) -> str:
    """Why can't single-op instructions x (first) and y (next) merge into one?
    before: the instruction ahead of x (latencies)."""
    xo = x.add if _busy(x.add) else x.mul
    yo = y.add if _busy(y.add) else y.mul
    if xo.dst in _reads(yo) or (xo.dst.acc is not None and any(
            getattr(s, "acc", None) == xo.dst.acc for s in _reads(yo))):
        return "dependency"
    if x.setf and yo.cond != "always":
        return "dependency"                                 # y reads x's flags
    if xo.dst == yo.dst and xo.dst != isa.NOP:
        return "same_dst"
    if before is not None:
        early = _written(before)
        if any(s in early and (s.is_regfile or yo.rot is not None) for s in _reads(yo) if isinstance(s, isa.Reg)):
            return "latency"                                # merged, y would follow `before` directly
    if x.sig != "none" and y.sig != "none":
        return "signal"
    if _periph(x) + _periph(y) > 1:
        return "periph"
    # A shared instruction takes its flags from the add unit.
    if (x if _busy(x.mul) else y).setf:
        return "flags"
    add, mul = (xo, yo) if _busy(x.add) else (yo, xo)
    files = set()
    if add.dst.wfile in ("A", "B"):
        files.add(0 if add.dst.wfile == "A" else 1)
    if mul.dst.wfile in ("A", "B"):
        files.add(0 if mul.dst.wfile == "B" else 1)
    if len(files) > 1:
        return "write_file"
    return _port_conflict([xo, yo]) or "free"


def regions(program) -> list:
    """(label, start, end) for each labelled region of the program."""
    starts = sorted(set(program.labels.values()) | {0})
    names = defaultdict(list)
    for label, idx in program.labels.items():
        names[idx].append(label)
    bounds = starts + [len(program.instrs)]
    return [("/".join(names.get(s, ["entry"])), s, e) for s, e in zip(bounds, bounds[1:]) if e > s]


def summarize(instrs, counts, lo=0, hi=None) -> dict:
    hi = len(instrs) if hi is None else hi
    cls = classify(instrs)
    total = float(sum(counts[lo:hi]))
    mix = Counter()
    add_ops = mul_ops = movs = ldi = br = 0.0
    ops = Counter()
    blockers = Counter()
    for i in range(lo, hi):
        w = counts[i]
        if not w:
            continue
        ins = instrs[i]
        mix[cls[i]] += w
        if ins.kind == "branch":
            br += w
        elif ins.kind in ("ldi", "sem"):
            ldi += w
        for op, unit in ((ins.add, "add"), (ins.mul, "mul")):
            if ins.kind == "alu" and _busy(op):
                if unit == "add":
                    add_ops += w
                else:
                    mul_ops += w
                movs += w if op.op == "mov" else 0
                ops[f"{unit}:{op.op}{'.setf' if ins.setf and unit == 'add' else ''}"] += w
        if i + 1 < hi and cls[i] in ("add", "mul") and cls[i + 1] in ("add", "mul") and cls[i] != cls[i + 1]:
            before = instrs[i - 1] if i > lo else None
            blockers[pair_blocker(ins, instrs[i + 1], before)] += min(w, counts[i + 1])
    bound = max(add_ops, mul_ops) + ldi + br
    return dict(executed=total, mix={k: v / total for k, v in mix.items()} if total else {},
                add_ops=add_ops, mul_ops=mul_ops, movs=movs, ldi=ldi, branches=br,
                bound=bound, packing=bound / total if total else 0.0,
                blockers=dict(blockers), ops=dict(ops.most_common(12)))


def profile_case(case, size="emu") -> dict:
    ck, run = compile_case(case, size)
    v3d = V3D()
    v3d.pc_counts = [0] * len(ck.program.instrs)
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    stats = run_emulator(ck, run.grid, values, v3d=v3d)
    counts = v3d.pc_counts
    whole = summarize(ck.program.instrs, counts)
    regs = []
    for label, lo, hi in regions(ck.program):
        weight = sum(counts[lo:hi])
        if weight:
            r = summarize(ck.program.instrs, counts, lo, hi)
            r.update(label=label, start=lo, length=hi - lo, entries=counts[lo], share=weight / whole["executed"])
            regs.append(r)
    regs.sort(key=lambda r: -r["share"])
    return dict(name=case.name, note=case.note, instructions=len(ck.program.instrs),
                programs=int(np.prod(run.grid)), cycles=stats.get("cycles"), whole=whole,
                regions=regs, ck=ck, counts=counts)


def fmt_pct(x) -> str:
    return f"{100 * x:5.1f}"


def table(results) -> str:
    head = (f"{'case':16} {'instr':>6} {'exec/prog':>9}  {'paired':>6} {'add':>5} {'mul':>5} {'nop_d':>5} "
            f"{'nop_o':>5} {'ldi':>5} {'br':>4} {'mov':>5}  {'pack':>5}  hottest region (share, len)")
    rows = [head, "-" * len(head)]
    for r in results:
        w, m = r["whole"], r["whole"]["mix"]
        hot = r["regions"][0]
        rows.append(
            f"{r['name']:16} {r['instructions']:6} {w['executed'] / r['programs']:9.0f}  "
            f"{fmt_pct(m.get('paired', 0)):>6} {fmt_pct(m.get('add', 0))} {fmt_pct(m.get('mul', 0))} "
            f"{fmt_pct(m.get('nop_delay', 0) + m.get('nop_end', 0))} {fmt_pct(m.get('nop_other', 0) + m.get('signal', 0))} "
            f"{fmt_pct(m.get('ldi', 0))} {fmt_pct(m.get('branch', 0))[1:]} "
            f"{fmt_pct(w['movs'] / max(w['add_ops'] + w['mul_ops'], 1))}  {w['packing']:5.2f}  "
            f"{hot['label']} ({100 * hot['share']:.0f}%, {hot['length']})")
    return "\n".join(rows)


def blockers_table(results) -> str:
    keys = ["dependency", "latency", "port_A", "port_B", "port", "write_file", "periph", "flags", "signal",
            "same_dst", "free"]
    rows = [f"{'case':16} " + " ".join(f"{k:>10}" for k in keys)]
    total = Counter()
    for r in results:
        b = r["whole"]["blockers"]
        n = sum(b.values()) or 1
        total.update(b)
        rows.append(f"{r['name']:16} " + " ".join(f"{100 * b.get(k, 0) / n:9.0f}%" for k in keys))
    n = sum(total.values()) or 1
    rows.append(f"{'ALL (weighted)':16} " + " ".join(f"{100 * total.get(k, 0) / n:9.0f}%" for k in keys))
    return "\n".join(rows)


def listing(r, region) -> str:
    instrs, counts = r["ck"].program.instrs, r["counts"]
    cls = classify(instrs)
    lines = []
    for i in range(region["start"], region["start"] + region["length"]):
        lines.append(f"{counts[i]:8} {cls[i]:9} {isa.format_instr(instrs[i])}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true", help="list each case's hottest region")
    ap.add_argument("--json", default=os.path.join(RESULTS, "analysis.json"))
    args = ap.parse_args(argv)
    cases = [CASES_BY_NAME[n] for n in args.cases] if args.cases else CASES
    results = []
    for case in cases:
        print(f"profiling {case.name} ...", file=sys.stderr, flush=True)
        results.append(profile_case(case))
    print(table(results))
    print("\nWhy adjacent add-only / mul-only instructions did not pair (share of such pairs):")
    print(blockers_table(results))
    tot = Counter()
    for r in results:
        tot.update(r["whole"]["ops"])
    if args.verbose:
        for r in results:
            print(f"\n== {r['name']}: hottest region {r['regions'][0]['label']}")
            print(listing(r, r["regions"][0]))
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("ck", "counts")} for r in results], f, indent=1)
    print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
