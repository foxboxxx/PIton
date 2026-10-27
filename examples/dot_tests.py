import piton as triton
import piton.language as tl


@triton.jit
def dot_tile(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def dot_loop(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BK: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k in range(0, K // BK):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def load_mod(b_ptr, out_ptr, N, stride_bk, BK: tl.constexpr, BN: tl.constexpr):
    offs_k = tl.arange(0, BK)
    offs_bn = (tl.program_id(0) * BN + tl.arange(0, BN)) % N
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :])
    tl.store(out_ptr + offs_k[:, None] * BN + tl.arange(0, BN)[None, :], b)


@triton.jit
def mm_v1(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_bk, stride_cm,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MOD: tl.constexpr):
    pid = tl.program_id(0)
    offs_am = tl.arange(0, BM)
    offs_bn = tl.arange(0, BN)
    if MOD:
        offs_am = offs_am % M
        offs_bn = offs_bn % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * stride_bk
    offs_cm = tl.arange(0, BM)
    offs_cn = tl.arange(0, BN)
    tl.store(c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :], acc)


@triton.jit
def mm_v2(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_bk, stride_cm,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SWIZZLE: tl.constexpr, MASK: tl.constexpr):
    pid = tl.program_id(0)
    if SWIZZLE:
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = 4 * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * 4
        group_size_m = min(num_pid_m - first_pid_m, 4)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid
        pid_n = 0
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BK
        b_ptrs += BK * stride_bk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    if MASK:
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    else:
        tl.store(c_ptrs, acc)


@triton.jit
def pid_math(out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = 4 * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs = tl.arange(0, 16)
    v = tl.where(offs == 0, num_pid_m, tl.where(offs == 1, num_pid_n, tl.where(offs == 2, group_id,
        tl.where(offs == 3, group_size_m, tl.where(offs == 4, pid_m, tl.where(offs == 5, pid_n, -1))))))
    tl.store(out_ptr + pid * 16 + offs, v)


@triton.jit
def mm_dbg(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, stride_am, stride_bk, stride_cm, dbg_ptr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = 4 * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BK
        b_ptrs += BK * stride_bk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    o = tl.arange(0, 16)
    v = tl.where(o == 0, pid_m, tl.where(o == 1, pid_n, tl.where(o == 2, M, tl.where(o == 3, N,
        tl.where(o == 4, stride_cm, tl.where(o == 5, num_pid_m, tl.where(o == 6, group_size_m, -7)))))))
    tl.store(dbg_ptr + o, v)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def rem_test(out_ptr, M, base, BLOCK: tl.constexpr):
    offs = (base + tl.arange(0, BLOCK)) % M
    q = (base + tl.arange(0, BLOCK)) // M
    tl.store(out_ptr + tl.arange(0, BLOCK), offs)
    tl.store(out_ptr + BLOCK + tl.arange(0, BLOCK), q)
