"""Measure compiler-generated, register-resident FP32 arithmetic on the Pi.

This is a compute ceiling experiment, not a matrix-multiplication benchmark.
Four independent affine recurrences keep useful values live between iterations;
their final values are returned and checked. No input loads occur in the loop.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piton as triton
import piton.language as tl
from piton import compiler
from piton.runtime.pi import Buffer, PiRunner


@triton.jit
def compute_kernel(x_ptr, out_ptr, ITERS: tl.constexpr, UNROLL: tl.constexpr):
    offsets = tl.program_id(0) * 64 + tl.arange(0, 16)
    a = tl.load(x_ptr + offsets)
    b = tl.load(x_ptr + offsets + 16)
    c = tl.load(x_ptr + offsets + 32)
    d = tl.load(x_ptr + offsets + 48)
    for iteration in range(0, ITERS):
        for unrolled in tl.static_range(0, UNROLL):
            # Sharing the coefficient permits both ALUs to use the same
            # register read, allowing add/multiply pairs at the QPU read ports.
            a = a * 0.9999 + 0.9999
            b = b * 0.9999 + 0.9999
            c = c * 0.9999 + 0.9999
            d = d * 0.9999 + 0.9999
    tl.store(out_ptr + offsets, a)
    tl.store(out_ptr + offsets + 16, b)
    tl.store(out_ptr + offsets + 32, c)
    tl.store(out_ptr + offsets + 48, d)


def run(iterations=8192, unroll=64, repeats=5, compile_only=False):
    ck = compiler.compile(compute_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32"},
                          {"ITERS": iterations, "UNROLL": unroll}, grid=(12,))
    result = {"experiment": "register-resident-affine-recurrences", "iterations": iterations,
              "unroll": unroll, "instructions": len(ck.program.instrs), "stats": ck.stats}
    for label, start in ck.program.labels.items():
        if label.startswith("loop") and label[4:].isdigit():
            end = min((p for p in ck.program.labels.values() if p > start),
                      default=len(ck.program.instrs))
            body = ck.program.instrs[start:end]
            adds = sum(i.add is not None and i.add.op == "fadd" for i in body)
            muls = sum(i.mul is not None and i.mul.op == "fmul" for i in body)
            paired = sum(i.add is not None and i.add.op == "fadd" and
                         i.mul is not None and i.mul.op == "fmul" for i in body)
            result.update(loop_instructions=len(body), vector_adds=adds, vector_multiplies=muls,
                          paired_arithmetic_instructions=paired,
                          issue_ceiling_gflops_at_250mhz=12 * (adds + muls) / len(body))
    if result.get("vector_adds") != 4 * unroll or result.get("vector_multiplies") != 4 * unroll:
        raise AssertionError("compiled loop does not match benchmark FLOP accounting")
    if compile_only:
        return result
    x = np.linspace(-1, 1, 12 * 64, dtype=np.float32)
    buffers = [Buffer("x", x.nbytes, init=x.view(np.uint32)),
               Buffer("out", x.nbytes, init=np.full(x.size, 0x7fc00000, dtype=np.uint32)),
               Buffer("zero", 64), Buffer("scratch", 64)]
    programs = []
    for pid in range(12):
        uniforms = []
        for kind, name in ck.uniforms:
            if kind == "arg":
                uniforms.append(("buf", "x" if name == "x_ptr" else "out", 0))
            elif kind == "pid":
                uniforms.append(pid if name == "pid0" else 0)
            elif kind == "nprog":
                uniforms.append(12 if name == "nprog0" else 1)
            elif kind in ("zero", "scratch"):
                uniforms.append(("buf", kind, 0))
            elif kind == "self":
                uniforms.append(("self",))
            else:
                raise ValueError((kind, name))
        programs.append(uniforms)
    measured = PiRunner().run(ck.words, buffers, programs, ["out"], repeats=repeats)
    got = measured.outputs["out"].view(np.float32)
    alpha = float(np.float32(0.9999))
    beta = alpha
    power = alpha ** (iterations * unroll)
    reference = power * x.astype(np.float64) + beta * (1 - power) / (1 - alpha)
    max_error = float(np.max(np.abs(got - reference)))
    # Repeated toward-zero updates accumulate a small steady-state bias.
    if not np.all(np.isfinite(got)) or not np.allclose(got, reference, rtol=0.005, atol=0.003):
        raise AssertionError(f"arithmetic recurrence verification failed: max error {max_error}")
    times = measured.times_us
    if len(times) != repeats or min(times) <= 0:
        raise AssertionError(f"invalid timing samples: {times}")
    flop = 12 * 16 * 4 * 2 * iterations * unroll
    result.update(times_us=times, best_us=min(times), gflops=flop / min(times) / 1000,
                  max_abs_error=max_error, correctness="all 768 outputs checked against FP64 closed form")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8192)
    parser.add_argument("--unroll", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.unroll, args.repeats) < 1:
        parser.error("iterations, unroll and repeats must be positive")
    print(json.dumps(run(args.iterations, args.unroll, args.repeats, args.compile_only)))
