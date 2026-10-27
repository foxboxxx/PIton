"""Regression tests for bugs found by code review (each was reproduced first)."""
import numpy as np
import pytest

import piton
import piton.language as tl
from piton.language.core import CompilationError

rng = np.random.default_rng(11)


# -- exact 32-bit integer division -------------------------------------------------

@piton.jit
def idiv_k(x_ptr, y_ptr, o_ptr, n, U: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1)
    y = tl.load(y_ptr + offs, mask=m, other=1)
    if U:
        x = x.to(tl.uint32, bitcast=True)
        y = y.to(tl.uint32, bitcast=True)
    tl.store(o_ptr + offs, (x // y).to(tl.int32, bitcast=True), mask=m)
    tl.store(o_ptr + n + offs, (x % y).to(tl.int32, bitcast=True), mask=m)


def _c_divmod(x, y):
    q = np.array([abs(int(a)) // abs(int(b)) * (1 if (a >= 0) == (b >= 0) else -1) for a, b in zip(x, y)])
    return q, np.asarray(x, np.int64) - q * np.asarray(y, np.int64)


@pytest.mark.parametrize("unsigned", [False, True])
def test_integer_division_exact(unsigned):
    """Was exact only for |a| < 2**24, and even there off by one for large quotients."""
    n = 2048
    if unsigned:
        x = rng.integers(0, 2 ** 32, n, dtype=np.uint64)
        e = rng.integers(1, 33, n).astype(np.uint64)
        y = np.maximum(1, rng.integers(0, 2 ** 32, n, dtype=np.uint64) >> (np.uint64(32) - e))
        x[:6] = [2 ** 32 - 1, 2 ** 32 - 1, 2 ** 32 - 1, 2 ** 31, 7, 14398348]
        y[:6] = [1, 2 ** 32 - 1, 2 ** 31, 2 ** 31 + 1, 2 ** 32 - 1, 1]
        q, r = x // y, x % y
        xs, ys = x.astype(np.uint32).view(np.int32), y.astype(np.uint32).view(np.int32)
        want = np.concatenate([q, r]).astype(np.uint32).view(np.int32)
    else:
        x = rng.integers(-2 ** 31, 2 ** 31, n)
        y = np.where(rng.random(n) < 0.5, rng.integers(1, 2 ** 8, n), rng.integers(1, 2 ** 31, n))
        y = y * rng.choice([-1, 1], n)
        x[:8] = [-2 ** 31, -2 ** 31, 2 ** 31 - 1, -2 ** 31, 0, 5, 14398348, 2 ** 24 - 1]
        y[:8] = [1, -1, -1, 2 ** 31 - 1, 7, -2 ** 31, 1, 3]
        q, r = _c_divmod(x, y)
        q = np.where(q == 2 ** 31, -2 ** 31, q)          # INT_MIN / -1 wraps (UB in C)
        xs, ys = x.astype(np.int32), y.astype(np.int32)
        want = np.concatenate([q, r]).astype(np.int64).astype(np.uint32).view(np.int32)
    o = np.zeros(2 * n, np.int32)
    idiv_k[(piton.cdiv(n, 64),)](xs, ys, o, n, U=int(unsigned), BLOCK=64)
    np.testing.assert_array_equal(o, want)


# -- stores ------------------------------------------------------------------------

@piton.jit
def two_stores_k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x)
    tl.store(o_ptr + BLOCK + offs, x + 1.0, mask=offs < n)


@pytest.mark.parametrize("n", [5, 40, 64])
def test_partial_store_waits_for_previous_dma(n):
    """The partial-store path rewrote VPM while the previous DMA was running
    (the emulator raises a hazard for that)."""
    x = np.arange(64, dtype=np.float32)
    o = np.zeros(128, np.float32)
    two_stores_k[(1,)](x, o, n, BLOCK=64)
    np.testing.assert_array_equal(o[:64], x)
    np.testing.assert_array_equal(o[64:64 + n], x[:n] + 1)
    assert np.all(o[64 + n:] == 0)


# -- control flow ------------------------------------------------------------------

@piton.jit
def if_nan_k(o, n, x, C: tl.constexpr, NE: tl.constexpr):
    if n == 0:
        v = 1
    else:
        v = 2
    if NE:
        c = x != C
    else:
        c = x < C
    if c:
        tl.store(o, v + 100)
    else:
        tl.store(o, v + 200)


@pytest.mark.parametrize("ne", [0, 1])
@pytest.mark.parametrize("n", [0, 1])
def test_runtime_if_on_nan_compare(ne, n):
    o = np.zeros(1, np.int32)
    if_nan_k[(1,)](o, n, 1.5, C=float("nan"), NE=ne)
    assert o[0] == (1 if n == 0 else 2) + (100 if ne else 200)


@piton.jit
def static_ret_k(o, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in tl.static_range(3):
        if n == i:
            return
        tl.store(o + i * BLOCK + offs, offs)


def test_return_inside_static_range_is_rejected():
    with pytest.raises(CompilationError, match="return inside a for loop"):
        static_ret_k[(1,)](np.zeros(64, np.int32), 1, BLOCK=16)


@piton.jit
def predcarry_k(x_ptr, o, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < 3
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(n):
        acc += tl.load(x_ptr + i * BLOCK + offs, mask=m, other=100)
        m = offs < i + 4
    tl.store(o + offs, acc)


@pytest.mark.parametrize("stages", [1, 2])
def test_pipelined_load_with_carried_mask(stages):
    x = np.arange(64, dtype=np.int32)
    o = np.zeros(16, np.int32)
    predcarry_k[(1,)](x, o, 4, BLOCK=16, num_stages=stages)
    offs = np.arange(16)
    acc, m = np.zeros(16, np.int64), offs < 3
    for i in range(4):
        acc += np.where(m, x[i * 16 + offs], 100)
        m = offs < i + 4
    np.testing.assert_array_equal(o, acc)


@piton.jit
def carry_inc_k(x_ptr, o, n, s, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    ss = tl.full((BLOCK,), s, tl.int32)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(n):
        acc += tl.load(p)
        if MODE == 0:
            p += (ss > 0).to(tl.int32) * BLOCK
        elif MODE == 1:
            p += (ss.to(tl.float32) * 1.0).to(tl.int32)
        else:
            p += tl.abs(ss)
    tl.store(o + offs, acc)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_pointer_carry_with_dense_uniform_increment(mode):
    x = np.arange(64, dtype=np.int32)
    o = np.zeros(16, np.int32)
    carry_inc_k[(1,)](x, o, 3, 16, BLOCK=16, MODE=mode)
    np.testing.assert_array_equal(o, x[0:16] + x[16:32] + x[32:48])


# -- integer arithmetic --------------------------------------------------------------

@piton.jit
def cmp_edge_k(o, y, BASE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = offs + BASE
    tl.store(o + offs, (x < y).to(tl.int32))
    tl.store(o + BLOCK + offs, (x > y).to(tl.int32))


@pytest.mark.parametrize("base", [-2 ** 31, 2 ** 31 - 16])
@pytest.mark.parametrize("y", [-2 ** 31, -2 ** 31 + 3, 0, 5, 2 ** 31 - 5, 2 ** 31 - 1])
def test_compare_narrowing_at_int32_limits(base, y):
    o = np.zeros(32, np.int32)
    cmp_edge_k[(1,)](o, y, BASE=base, BLOCK=16)
    x = np.arange(16, dtype=np.int64) + base
    np.testing.assert_array_equal(o, np.concatenate([x < y, x > y]).astype(np.int32))


@piton.jit
def uop_k(o, a, b, A: tl.constexpr, B: tl.constexpr, RT: tl.constexpr, OP: tl.constexpr):
    offs = tl.arange(0, 16)
    if RT:
        x = (a + offs * 0).to(tl.uint32, bitcast=True)
        y = (b + offs * 0).to(tl.uint32, bitcast=True)
    else:
        x = (A + offs * 0).to(tl.uint32, bitcast=True)
        y = (B + offs * 0).to(tl.uint32, bitcast=True)
    if OP == 0:
        r = tl.maximum(x, y)
    elif OP == 1:
        r = tl.minimum(x, y)
    elif OP == 2:
        r = x // y
    else:
        r = x % y
    tl.store(o + offs, r.to(tl.int32, bitcast=True))


@pytest.mark.parametrize("A,B", [(-7, 3), (5, -1), (-1, 2), (7, 3)])
@pytest.mark.parametrize("op", [0, 1, 2, 3])
@pytest.mark.parametrize("rt", [0, 1])
def test_uint32_uniform_ops(A, B, op, rt):
    ua, ub = A & 0xFFFFFFFF, B & 0xFFFFFFFF
    want = [max(ua, ub), min(ua, ub), ua // ub, ua % ub][op]
    o = np.zeros(16, np.int32)
    uop_k[(1,)](o, A, B, A=A, B=B, RT=rt, OP=op)
    assert int(o[0]) & 0xFFFFFFFF == want


@piton.jit
def ured_k(x_ptr, o, R: tl.constexpr, L: tl.constexpr, AXIS: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, L)[None, :]
    x = tl.load(x_ptr + r * L + c)
    n: tl.constexpr = L if AXIS == 0 else R
    tl.store(o + tl.arange(0, n), tl.max(x, axis=AXIS))
    tl.store(o + n + tl.arange(0, n), tl.min(x, axis=AXIS))


@pytest.mark.parametrize("R,L,axis", [(4, 16, 1), (16, 16, 0), (8, 4, 0), (8, 4, 1), (1, 16, 1)])
def test_uint32_reductions_compare_unsigned(R, L, axis):
    x = rng.integers(0, 2 ** 32, (R, L), dtype=np.uint64).astype(np.uint32)
    n = L if axis == 0 else R
    o = np.zeros(2 * n, np.uint32)
    ured_k[(1,)](x, o, R=R, L=L, AXIS=axis)
    np.testing.assert_array_equal(o, np.concatenate([x.max(axis=axis), x.min(axis=axis)]))


@piton.jit
def mul_base_k(o, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    d = (offs % 4 + 1) * (offs % 2 + 1)
    e = d - 1 + pid                  # scalar part pid - 1 is negative for pid 0
    tl.store(o + pid * BLOCK + offs, e * 1000)


def test_affine_with_dense_times_scalar():
    o = np.zeros(32, np.int32)
    mul_base_k[(2,)](o, BLOCK=16)
    offs = np.arange(16)
    want = np.concatenate([((offs % 4 + 1) * (offs % 2 + 1) - 1 + p) * 1000 for p in range(2)])
    np.testing.assert_array_equal(o, want)


@piton.jit
def bitcast_k(o, BITS: tl.constexpr):
    offs = tl.arange(0, 16)
    f = tl.full((16,), BITS, tl.int32).to(tl.float32, bitcast=True)
    tl.store(o + offs, f.to(tl.int32, bitcast=True))
    s = tl.full((), BITS, tl.int32).to(tl.float32, bitcast=True)
    tl.store(o + 16, s.to(tl.int32, bitcast=True))


@pytest.mark.parametrize("bits", [0x7F800001, 0x7FA00000, 0xFF800001, 0x7FC00001, 0x00000001, 0x3F800000])
def test_constant_bitcast_keeps_nan_payload(bits):
    sb = bits - (1 << 32) if bits >= 1 << 31 else bits
    o = np.zeros(17, np.int32)
    bitcast_k[(1,)](o, BITS=sb)
    assert [int(v) & 0xFFFFFFFF for v in (o[0], o[16])] == [bits, bits]


# -- second review round ---------------------------------------------------------------

@piton.jit
def idx_div_k(o_ptr, N, BLOCK: tl.constexpr, OP: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if OP == 0:
        v = offs // N
    else:
        v = (offs // N) * 7 + offs % N
    tl.store(o_ptr + offs, v)


@pytest.mark.parametrize("block,op", [(512, 0), (256, 1)])
def test_division_of_big_tiles_compiles(block, op):
    """The exact division used to tie every dividend register to regfile A."""
    o = np.zeros(2 * block, np.int32)
    idx_div_k[(2,)](o, 7, BLOCK=block, OP=op)
    offs = np.arange(2 * block)
    np.testing.assert_array_equal(o, offs // 7 if op == 0 else (offs // 7) * 7 + offs % 7)


@piton.jit
def div_one_k(o, a, b):
    tl.store(o, a // b)
    tl.store(o + 1, a % b)
    tl.store(o + 2, tl.cdiv(a, b))


@pytest.mark.parametrize("a", [-5, -1, 7, -2 ** 31 + 3])
def test_signed_division_by_one(a):
    o = np.zeros(3, np.int32)
    div_one_k[(1,)](o, a, 1)          # b == 1 is specialised to a constant
    assert list(o) == [a, 0, a]


@piton.jit
def const_fcmp_k(o):
    offs = tl.arange(0, 16)
    s = tl.full((16,), 2.0, tl.float32)
    tl.store(o + offs, ((s > 0.0) & (s == 2.0) & (tl.max(s, axis=0) <= s)).to(tl.int32))


def test_compare_of_constant_tiles():
    o = np.zeros(16, np.int32)
    const_fcmp_k[(1,)](o)
    assert np.all(o == 1)


@piton.jit
def other_negzero_k(x_ptr, o):
    offs = tl.arange(0, 16)
    v = tl.load(x_ptr + offs, mask=offs < 8, other=-0.0)
    tl.store(o + offs, v)


def test_load_other_negative_zero():
    o = np.zeros(16, np.float32)
    other_negzero_k[(1,)](np.ones(16, np.float32), o)
    assert np.all(o[:8] == 1) and np.all(np.signbit(o[8:]))


@piton.jit
def nan_reduce_k(x_ptr, o, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    tl.store(o, tl.min(x, axis=0))
    tl.store(o + 1, tl.max(x, axis=0))


@pytest.mark.parametrize("n", [2, 8, 32])
def test_short_nan_reductions(n):
    o = np.zeros(2, np.float32)
    nan_reduce_k[(1,)](np.full(n, np.nan, np.float32), o, N=n)
    assert np.all(np.isnan(o))


@piton.jit
def chase_k(nxt_ptr, x_ptr, o, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = offs
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(n):
        acc += tl.load(x_ptr + idx)
        idx = tl.load(nxt_ptr + idx)       # next address from data loaded in this iteration
    tl.store(o + offs, acc)


@pytest.mark.parametrize("stages", [1, 2])
def test_pipeline_with_data_dependent_addresses(stages):
    nxt = ((np.arange(64) * 5 + 3) % 64).astype(np.int32)
    x = np.arange(64, dtype=np.int32) * 10
    o = np.zeros(64, np.int32)
    chase_k[(1,)](nxt, x, o, 4, BLOCK=64, num_stages=stages)
    idx, acc = np.arange(64), np.zeros(64, np.int64)
    for _ in range(4):
        acc += x[idx]
        idx = nxt[idx]
    np.testing.assert_array_equal(o, acc)


@piton.jit
def range_k(x_ptr, o, lb, ub, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(lb, ub):
        acc += tl.load(x_ptr + i * BLOCK + offs)
    tl.store(o + offs, acc)


@pytest.mark.parametrize("stages", [1, 2])
def test_zero_trip_loop_does_not_load(stages):
    """The prefetch prologue used to run before the zero-trip check (here it would
    read far outside memory)."""
    o = np.full(16, 3.0, np.float32)
    range_k[(1,)](np.ones(32, np.float32), o, 2 ** 20, 0, BLOCK=16, num_stages=stages)
    assert np.all(o == 0)


@piton.jit
def count_k(o, lb, ub, step):
    c = 0
    for i in range(lb, ub, step):
        c += 1
    tl.store(o, c)


def test_loop_bounds_far_apart():
    o = np.zeros(1, np.int32)
    count_k[(1,)](o, -2 ** 31, 2 ** 30, 2 ** 30)
    assert o[0] == 3


@piton.jit
def big_grid_k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)


def test_program_ids_beyond_65535():
    """Program-id ranges come from the launch grid (they were assumed <= 65535)."""
    from piton.compiler import compile as pcompile
    from piton.qpu.emulator import V3D
    from piton.runtime.launcher import uniforms_for
    B, G = 16, 70000
    ck = pcompile(big_grid_k, {"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32"}, {"BLOCK": B}, grid=(G,))
    x = np.arange(G * B, dtype=np.float32)
    v = V3D(num_qpus=4, strict=True)
    addrs = {k: v.mem.alloc(G * B * 4, 64) for k in ("x_ptr", "o_ptr")}
    v.mem.write_array(addrs["x_ptr"], x)
    zero, scratch = v.mem.alloc(64, 64), v.mem.alloc(64, 64)
    pids, unifs = [0, 65535, 65536, G - 1], []
    vals = {"x_ptr": x, "o_ptr": x, "n": G * B}
    for p in pids:
        a = v.mem.alloc(4 * len(ck.uniforms), 16)
        v.mem.write_words(a, np.array(uniforms_for(ck, vals, addrs, (p, 0, 0), (G, 1, 1), zero, scratch, a),
                                      dtype=np.uint32))
        unifs.append(a)
    v.run(v.load(ck.words), unifs)
    out = v.mem.read_array(addrs["o_ptr"], (G * B,), np.float32)
    for p in pids:
        np.testing.assert_array_equal(out[p * B:(p + 1) * B], x[p * B:(p + 1) * B] + 1)


@piton.jit
def carried_detect_k(x_ptr, o, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = 0
    b = 0
    c = 0
    s = 0
    t = 0
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(n):
        if a == 1:
            c = c + 5
        a = b
        b = 1
        acc += tl.load(x_ptr + s * BLOCK + offs)
        s = t
        t = t + 1
    tl.store(o + offs, acc)
    tl.store(o + BLOCK, a)
    tl.store(o + BLOCK + 1, c)


def test_loop_carried_detection():
    """Variables whose first dry-run value matched their initial value were not carried."""
    x = np.arange(64, dtype=np.int32)
    o = np.zeros(18, np.int32)
    carried_detect_k[(1,)](x, o, 3, BLOCK=16)
    np.testing.assert_array_equal(o[:16], x[0:16] + x[0:16] + x[16:32])
    assert (o[16], o[17]) == (1, 5)


@piton.jit
def negzero_if_k(o, n):
    if n > 0:
        v = 0.0
    else:
        v = -0.0
    tl.store(o, v)


def test_negative_zero_through_if():
    o = np.ones(1, np.float32)
    negzero_if_k[(1,)](o, 0)
    assert o[0] == 0 and np.signbit(o[0])


SHADOWED = 7


@piton.jit
def shadow_k(o, n):
    for i in range(n):
        SHADOWED = i + 100
    tl.store(o, SHADOWED)


def test_name_assigned_only_in_loop_is_not_a_global():
    with pytest.raises(CompilationError, match="assigned only inside a runtime for loop"):
        shadow_k[(1,)](np.zeros(1, np.int32), 3)


# -- third review round ------------------------------------------------------------------

@piton.jit
def div_loop_k(x_ptr, o, n, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(n):
        a = tl.load(x_ptr + i * BLOCK + offs)
        acc += a // N + a % (N + 1) + (a + i) // (N + 2)
    tl.store(o + offs, acc)


def test_divisions_in_a_loop_share_constants():
    """Each division's blocks loaded their own constants; LICM hoisted every copy."""
    x = rng.integers(-1000, 1000, 3 * 64).astype(np.int32)
    o = np.zeros(64, np.int32)
    div_loop_k[(1,)](x, o, 3, 7, BLOCK=64)
    want = np.zeros(64, np.int64)
    for i in range(3):
        a = x[i * 64:(i + 1) * 64].astype(np.int64)
        want += np.trunc(a / 7).astype(np.int64) + np.fmod(a, 8) + np.trunc((a + i) / 9).astype(np.int64)
    np.testing.assert_array_equal(o, want)


@piton.jit
def nested_wide_k(o):
    c = 0
    s = 0
    for i in range(-2 ** 31, 3 * 2 ** 29, 2 ** 29):
        c += 1
        for j in range(2):
            s += j
    tl.store(o, c)
    tl.store(o + 1, s)


def test_nested_loop_keeps_outer_compare():
    o = np.zeros(2, np.int32)
    nested_wide_k[(1,)](o)
    assert list(o) == [7, 7]


@piton.jit
def iv_wrap_k(o, lb, ub, step):
    c = 0
    for i in range(lb, ub, step):
        c += 1
    tl.store(o, c)


@pytest.mark.parametrize("lb,ub,step,want", [(2147483628, 2147483646, 16, 2), (5, 7, 2147483647, 1),
                                              (2147483644, 2147483647, 2, 2)])
def test_loop_index_does_not_wrap(lb, ub, step, want):
    o = np.zeros(1, np.int32)
    iv_wrap_k[(1,)](o, lb, ub, step)
    assert o[0] == want


@piton.jit
def temps_k(x_ptr, o, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n):
        t = tl.load(x_ptr + i * BLOCK + offs)
        s = s + t
    for i in range(n):
        t = tl.load(x_ptr + i * BLOCK + offs) * 2.0      # the same temp name in a second loop
        s = s + t
    tl.store(o + offs, s)


def test_loop_local_temp_reused_in_later_loop():
    x = rng.integers(-9, 9, 3 * 16).astype(np.float32)
    o = np.zeros(16, np.float32)
    temps_k[(1,)](x, o, 3, BLOCK=16)
    np.testing.assert_array_equal(o, 3 * x.reshape(3, 16).sum(0))


@piton.jit
def number_carry_k(o, n):
    x = 0
    hit = False
    for i in range(n):
        x = 1.5
        hit = 7
    tl.store(o, x)
    tl.store(o + 1, hit)


def test_carried_numbers_widen():
    o = np.zeros(2, np.float32)
    number_carry_k[(1,)](o, 3)
    assert list(o) == [1.5, 7.0]


@piton.jit
def loopvar_k(o, n, m):
    i = -1
    s = 0
    for i in range(n):
        for i in range(m):
            pass
        s += i * 100
    tl.store(o, i)
    tl.store(o + 1, s)


def test_loop_variable_after_loop():
    o = np.zeros(2, np.int32)
    loopvar_k[(1,)](o, 2, 3)
    assert list(o) == [2, 400]
    loopvar_k[(1,)](o, 0, 3)
    assert list(o) == [-1, 0]


@piton.jit
def tuple_if_k(o, c):
    if c > 0:
        v = (0.0, 1.0)
    else:
        v = (-0.0, 1.0)
    tl.store(o, v[0])


def test_tuple_through_if_keeps_negative_zero():
    o = np.ones(1, np.float32)
    tuple_if_k[(1,)](o, 0)
    assert o[0] == 0 and np.signbit(o[0])


@piton.jit
def wide_mask_k(o, base, n, BLOCK: tl.constexpr):
    offs = base + tl.arange(0, BLOCK)
    tl.store(o + tl.arange(0, BLOCK), offs * 0 + 1, mask=offs < n)


@pytest.mark.parametrize("base,n", [(0, -2 ** 31 + 3), (-1000, 2 ** 31 - 1), (5, -2 ** 31), (0, 7)])
def test_store_mask_counts_do_not_overflow(base, n):
    o = np.zeros(16, np.int32)
    wide_mask_k[(1,)](o, base, n, BLOCK=16)
    np.testing.assert_array_equal(o, (base + np.arange(16) < n).astype(np.int32))


@piton.jit
def reduce_lanes_k(x_ptr, o):
    x = tl.load(x_ptr + tl.arange(0, 16))
    r = tl.max(x, axis=0)
    tl.store(o + tl.arange(0, 16), 1.0 / (r + tl.zeros((16,), tl.float32)))


def test_float_reduction_is_the_same_in_every_lane():
    x = np.array([0.0, -0.0] * 4 + [-1.0] * 8, np.float32)
    o = np.zeros(16, np.float32)
    reduce_lanes_k[(1,)](x, o)
    assert len(set(o.tolist())) == 1


@piton.jit
def neg_k(x_ptr, o):
    offs = tl.arange(0, 16)
    tl.store(o + offs, -tl.load(x_ptr + offs))


def test_float_negation_flips_the_sign_bit():
    x = np.array([np.nan, -np.nan, 0.0, -0.0, 1.0, -2.5, np.inf, -np.inf] * 2, np.float32)
    o = np.zeros(16, np.float32)
    neg_k[(1,)](x, o)
    np.testing.assert_array_equal(o.view(np.uint32), x.view(np.uint32) ^ np.uint32(0x80000000))


# -- fourth review round -----------------------------------------------------------------

@piton.jit
def udiv_live_k(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.uint32, bitcast=True)
    y = tl.load(y_ptr + offs).to(tl.uint32, bitcast=True)
    a0, a1, a2 = x // y, x % y, (x + 1) // y
    a3, a4, a5 = (x + 2) % y, (x ^ 5) // y, (x ^ 9) % y
    t = ((((a0 + a1) ^ a2) + a3) ^ a4) + a5
    tl.store(o_ptr + offs, t.to(tl.int32, bitcast=True))


def test_many_live_unsigned_divisions_compile():
    """At the register-capacity edge: the cold long-division path kept extra temporaries."""
    x = rng.integers(0, 2 ** 32, 128, dtype=np.uint64)
    y = np.maximum(1, rng.integers(0, 2 ** 32, 128, dtype=np.uint64) >> rng.integers(0, 32, 128).astype(np.uint64))
    o = np.zeros(128, np.int32)
    udiv_live_k[(1,)](x.astype(np.uint32).view(np.int32), y.astype(np.uint32).view(np.int32), o, BLOCK=128)
    m = 0xFFFFFFFF
    a0, a1, a2 = x // y, x % y, ((x + 1) & m) // y
    a3, a4, a5 = ((x + 2) & m) % y, (x ^ 5) // y, (x ^ 9) % y
    t = (((((a0 + a1) & m ^ a2) + a3) & m ^ a4) + a5) & m
    np.testing.assert_array_equal(o, t.astype(np.uint32).view(np.int32))


@piton.jit
def ivafter_k(o, n, KIND: tl.constexpr):
    offs = tl.arange(0, 16)
    if KIND == 0:
        i = 0.5
    elif KIND == 1:
        i = offs * 0 + 100
    else:
        i = (n * 0 + 9).to(tl.uint32)
    for i in range(n):
        pass
    tl.store(o + offs, (i + offs * 0).to(tl.float32))


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("n", [0, 3])
def test_loop_variable_after_loop_any_binding(kind, n):
    o = np.zeros(16, np.float32)
    ivafter_k[(1,)](o, n, KIND=kind)
    assert np.all(o == ([0.5, 100.0, 9.0][kind] if n == 0 else n - 1))


@piton.jit
def bigbound_k(o):
    c = 0
    for i in range(0, 5, 2 ** 32 + 1):
        c += 1
    tl.store(o, c)


def test_loop_bounds_outside_int32_are_rejected():
    with pytest.raises(CompilationError, match="does not fit"):
        bigbound_k[(1,)](np.zeros(1, np.int32))


@piton.jit
def ucarry_k(u_ptr, o, n):
    x = 0
    for i in range(n):
        x = x + tl.load(u_ptr + i)
    tl.store(o, x.to(tl.int32, bitcast=True))


def test_number_carry_joins_uint32():
    u = np.array([2 ** 31, 2 ** 30, 3], np.uint32)
    o = np.zeros(1, np.int32)
    ucarry_k[(1,)](u, o, 3)
    assert int(o[0]) & 0xFFFFFFFF == int(u.astype(np.uint64).sum()) & 0xFFFFFFFF


@piton.jit
def none_if_k(o, c, mask_ptr=None):
    res = None
    if c > 0:
        res = None
        mask_ptr = None
    tl.store(o, 1 if res is None else 0)
    tl.store(o + 1, 3 if mask_ptr is None else 0)


def test_none_values_survive_runtime_if():
    o = np.zeros(2, np.int32)
    none_if_k[(1,)](o, 2)
    assert list(o) == [1, 3]


@piton.jit
def bias_k(x_ptr, o, bias_ptr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    if bias_ptr is not None:
        x = x + tl.load(bias_ptr + offs)
    tl.store(o + offs, x)


def test_is_not_none_on_a_pointer():
    x = np.arange(16, dtype=np.float32)
    o = np.zeros(16, np.float32)
    bias_k[(1,)](x, o, np.ones(16, np.float32))
    np.testing.assert_array_equal(o, x + 1)


@piton.jit
def narrow_count_k(o, n, q):
    offs = tl.arange(0, 16)
    tl.store(o + offs, offs + 100, mask=(offs <= tl.maximum(n, 0)) & (offs < q))


@pytest.mark.parametrize("q", [0, 5])
def test_store_count_near_int_max(q):
    o = np.zeros(16, np.int32)
    narrow_count_k[(1,)](o, 2 ** 31 - 1, q)
    np.testing.assert_array_equal(o, np.where(np.arange(16) < q, np.arange(16) + 100, 0))
