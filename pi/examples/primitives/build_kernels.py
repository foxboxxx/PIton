"""Compile every benchmarked primitive AOT and emit its .c/.h launcher.

One manifest drives both this build step and `main.c`: each entry names the
Triton kernel, its runtime signature and its compile-time constants, and the
generated `<name>_launch()` is what the benchmark calls. Run by the Makefile:

    python3 build_kernels.py --out .            # writes op_*.c/.h + ops.h
    python3 build_kernels.py --list-sources     # for COMMON_SRC
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# piton (and numpy) are imported inside build() so that --list-sources, which
# the Makefile calls on every invocation to populate COMMON_SRC, stays cheap.

# Problem sizes. VEC is the elementwise length; MAT is the square matrix side
# used by the row reductions, transpose and softmax; MM is the matmul side.
VEC = 262144            # 1 MiB per fp32 buffer
MAT = 256
MM = 256

PTR = "*fp32"
IPTR = "*i32"

# name, source file, kernel, signature, constants, grid
OPS = [
    ("op_copy", "examples/primitive_kernels.py", "copy_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, VEC // 256),
    ("op_add", "examples/kernels.py", "add_kernel",
     {"x_ptr": PTR, "y_ptr": PTR, "output_ptr": PTR, "n_elements": "i32"}, {"BLOCK_SIZE": 256},
     VEC // 256),
    ("op_mul", "examples/elementwise.py", "mul_kernel",
     {"x_ptr": PTR, "y_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, VEC // 256),
    ("op_fma", "examples/primitive_kernels.py", "fma_kernel",
     {"x_ptr": PTR, "y_ptr": PTR, "out_ptr": PTR, "n": "i32", "alpha": "fp32"}, {"BLOCK": 256},
     VEC // 256),
    ("op_relu", "examples/kernels.py", "scaled_relu_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32", "alpha": "fp32"}, {"BLOCK": 256}, VEC // 256),
    ("op_exp", "examples/sfu_tests.py", "sfu_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"OP": 4, "BLOCK": 256}, VEC // 256),
    ("op_log", "examples/primitive_kernels.py", "log_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"BLOCK": 256}, VEC // 256),
    ("op_sqrt", "examples/sfu_tests.py", "sfu_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"OP": 5, "BLOCK": 256}, VEC // 256),
    ("op_rsqrt", "examples/sfu_tests.py", "sfu_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"OP": 3, "BLOCK": 256}, VEC // 256),
    ("op_recip", "examples/sfu_tests.py", "sfu_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n": "i32"}, {"OP": 2, "BLOCK": 256}, VEC // 256),
    ("op_idiv", "examples/primitive_kernels.py", "idiv_kernel",
     {"x_ptr": IPTR, "y_ptr": IPTR, "q_ptr": IPTR, "r_ptr": IPTR, "n": "i32"}, {"BLOCK": 256},
     VEC // 256),
    ("op_rowsum", "examples/kernels.py", "row_sum_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n_cols": "i32", "stride": "i32"},
     {"BLOCK_M": 4, "BLOCK_N": 64}, MAT // 4),
    ("op_rowmax", "examples/primitive_kernels.py", "row_max_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n_cols": "i32", "stride": "i32"},
     {"BLOCK_M": 4, "BLOCK_N": 64}, MAT // 4),
    ("op_transpose", "examples/primitive_kernels.py", "transpose_kernel",
     {"x_ptr": PTR, "out_ptr": PTR, "n_rows": "i32", "n_cols": "i32", "stride_x": "i32",
      "stride_out": "i32"}, {"BLOCK_M": 16, "BLOCK_N": 16}, (MAT // 16) * (MAT // 16)),
    ("op_softmax", "examples/kernels.py", "softmax_kernel",
     {"output_ptr": PTR, "input_ptr": PTR, "input_row_stride": "i32", "output_row_stride": "i32",
      "n_cols": "i32"}, {"BLOCK_SIZE": MAT}, MAT),
    ("op_matmul", "examples/kernels.py", "matmul_kernel",
     {"a_ptr": PTR, "b_ptr": PTR, "c_ptr": PTR, "M": "i32", "N": "i32", "stride_am": "i32",
      "stride_bk": "i32", "stride_cm": "i32"},
     {"K": MM, "stride_ak": 1, "stride_bn": 1, "stride_cn": 1, "BLOCK_SIZE_M": 8,
      "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 4, "GROUP_SIZE_M": 4},
     ((MM + 7) // 8) * ((MM + 63) // 64)),
]
# extra compile() arguments: the grouped matmul requests TMU_NOSWAP (PITON_TMU_NOSWAP=1)
OPTIONS = {"op_matmul": {"tmu_noswap": True}}


def build(out_dir: str, verbose: bool = True) -> list:
    from piton import compiler
    from piton.tools.compile import _load_kernel, emit_c

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, path, kernel, sig, cons, grid in OPS:
        fn = _load_kernel(os.path.join(ROOT, path), kernel)
        ck = compiler.compile(fn, dict(sig), dict(cons), name=name, grid=(grid,), **OPTIONS.get(name, {}))
        h, c = emit_c(ck, name)
        with open(os.path.join(out_dir, name + ".h"), "w") as f:
            f.write(h)
        with open(os.path.join(out_dir, name + ".c"), "w") as f:
            f.write(c)
        with open(os.path.join(out_dir, name + ".qasm"), "w") as f:
            f.write(ck.asm["qasm"])
        rows.append((name, kernel, grid, len(ck.program.instrs)))
        if verbose:
            print(f"{name:14s} {kernel:22s} grid={grid:<6d} {len(ck.program.instrs):5d} instructions")
    # Single header so main.c does not have to list every kernel by hand.
    with open(os.path.join(out_dir, "ops.h"), "w") as f:
        f.write("// Generated by build_kernels.py: every benchmarked kernel.\n")
        f.write("#ifndef PITON_OPS_H\n#define PITON_OPS_H\n\n")
        for name, _, grid, instrs in rows:
            f.write(f'#include "{name}.h"\n')
        f.write("\n")
        f.write(f"#define OP_VEC {VEC}\n#define OP_MAT {MAT}\n#define OP_MM {MM}\n\n")
        for name, _, grid, instrs in rows:
            f.write(f"#define {name.upper()}_GRID {grid}\n")
            f.write(f"#define {name.upper()}_INSTRS {instrs}\n")
        f.write("\n#endif\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--list-sources", action="store_true",
                    help="print the generated .c files and exit (no compilation)")
    args = ap.parse_args(argv)
    if args.list_sources:
        print(" ".join(name + ".c" for name, *_ in OPS))
        return 0
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
