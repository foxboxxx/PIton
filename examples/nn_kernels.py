"""Batched dense inference layers using the measured PIton matmul tiles.

Inputs and weights use ordinary contiguous FP32 row-major storage: X[M, K],
W[K, N], bias[N] and Y[M, N]. Bias and ReLU share the matrix multiplication's
launch, so hidden activations require no separate round-trip through memory.
Feature dimensions are specialized when compiling a layer. The runtime batch
size may be ragged, so the same compiled layer handles multiple batch sizes.
"""

import piton
import piton.language as tl


def dense_tiles(output_features):
    """Return (BM, BN, BK) for the hidden and final layers of the small MLP."""
    # Bias needs live registers in addition to the matmul accumulators. Halving
    # BM from the plain-matmul winner leaves room for the fused epilogue.
    return (4, 64, 8) if output_features >= 64 else (8, 16, 8)


@piton.jit
def dense_layer_kernel(
        x_ptr, w_ptr, bias_ptr, y_ptr,
        M, N: tl.constexpr, K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        ADD_BIAS: tl.constexpr, RELU: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # Mask incomplete batch rows; feature columns wrap onto real weights, as in
    # the Triton tutorial matmul. Only in-bounds tile elements are stored.
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    ks = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
    w_ptrs = w_ptr + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if K % BLOCK_SIZE_K == 0:
            x = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
            w = tl.load(w_ptrs)
        else:
            x_mask = (rows[:, None] < M) & (ks[None, :] < K - k * BLOCK_SIZE_K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=ks[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * N

    if ADD_BIAS:
        bias = tl.load(bias_ptr + cols)
        acc += bias[None, :]
    if RELU:
        acc = tl.maximum(acc, 0.0)

    output_rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = y_ptr + output_rows[:, None] * N + output_cols[None, :]
    output_mask = (output_rows[:, None] < M) & (output_cols[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)
