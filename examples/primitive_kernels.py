"""Triton kernels for the per-primitive benchmark (`pi/examples/primitives`).

Each kernel isolates one Triton primitive so its cost can be compared against
scalar ARM code on the same board. Kernels that already exist elsewhere in
`examples/` (add, mul, scaled_relu, row_sum, softmax, matmul, the SFU ops) are
not repeated here; this file only fills the gaps.
"""
import piton as triton
import piton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Load/store only: the memory-bandwidth floor for every other kernel."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)


@triton.jit
def fma_kernel(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    """saxpy: one multiply and one add per element, so both ALUs can pair."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(out_ptr + offs, x * alpha + y, mask=m)


@triton.jit
def idiv_kernel(x_ptr, y_ptr, q_ptr, r_ptr, n, BLOCK: tl.constexpr):
    """Exact 32-bit // and %: no integer divide on the QPU, no divider on VFP."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1)
    y = tl.load(y_ptr + offs, mask=m, other=1)
    tl.store(q_ptr + offs, x // y, mask=m)
    tl.store(r_ptr + offs, x % y, mask=m)


@triton.jit
def row_max_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """tl.max along a row: a cross-lane reduction via masked lane rotations."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    acc = tl.full((BLOCK_M,), -3.4028234663852886e+38, dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        x = tl.load(x_ptr + rows[:, None] * stride + (start + cols)[None, :])
        acc = tl.maximum(acc, tl.max(x, axis=1))
    tl.store(out_ptr + rows, acc)


@triton.jit
def transpose_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_x, stride_out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """tl.trans on a tile: pure data movement across the 16 SIMD lanes."""
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(n_cols, BLOCK_N)
    pid_m = pid // tiles_n
    pid_n = pid % tiles_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    tile = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=mask)
    t = tl.trans(tile)
    out_mask = (cols[:, None] < n_cols) & (rows[None, :] < n_rows)
    tl.store(out_ptr + cols[:, None] * stride_out + rows[None, :], t, mask=out_mask)


@triton.jit
def log_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Natural log. `sfu_tests.sfu_kernel` only exposes log2, and the CPU
    baselines in the benchmark compute ln, so give tl.log its own kernel."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.log(tl.load(x_ptr + offs, mask=m, other=1.0)), mask=m)
