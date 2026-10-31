"""Matmul with B stored as contiguous column panels: [N / BN, K, BN].

The input B must have N divisible by BLOCK_SIZE_N. Packing or generating
this layout is input preparation and must be accounted for separately when
comparing a complete workload with an ordinary row-major multiplication.
"""
import numpy as np

import piton
import piton.language as tl


def pack_b_columns(b, width):
    """Return row-major B rearranged into [column panel, K, local column]."""
    b = np.asarray(b)
    if b.ndim != 2 or not isinstance(width, int) or width <= 0 or b.shape[1] % width:
        raise ValueError("B must be a matrix with columns divisible by the positive panel width")
    k, n = b.shape
    return np.ascontiguousarray(b.reshape(k, n // width, width).transpose(1, 0, 2))


@piton.jit
def matmul_packed_b_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_cm,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
    b_ptrs = (b_ptr + pid_n * K * BLOCK_SIZE_N
              + offs_k[:, None] * BLOCK_SIZE_N + offs_bn[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * BLOCK_SIZE_N

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + offs_bn
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
