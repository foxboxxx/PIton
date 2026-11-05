"""PITON_REROLL_REDUCE: row reductions rerolled into one loop per pass
(reroll._run_passes) and transposed tiles loaded through transposed pointers
(reroll.transposed_loads), in the strict emulator.

Every rerolled kernel must store the same bits as the unrolled one (switch
off) and match numpy; kernels the pass leaves alone must compile to the same
machine code with the switch on. Small kernels are rerolled by lowering the
instruction-cache threshold.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton import compiler, reroll  # noqa: E402
from piton.backend.regalloc import RegAllocError  # noqa: E402
from piton.qpu import emulator  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from kernels import softmax_kernel  # noqa: E402
from primitive_kernels import transpose_kernel  # noqa: E402
from test_store_slots import contended, launch  # noqa: E402
from test_store_slots_protocol import protocol_errors  # noqa: E402

rng = np.random.default_rng(11)
SOFTMAX = {"output_ptr": "*fp32", "input_ptr": "*fp32", "input_row_stride": "i32",
           "output_row_stride": "i32", "n_cols": "i32"}
TRANSPOSE = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_rows": "i32", "n_cols": "i32",
             "stride_x": "i32", "stride_out": "i32"}


def build(monkeypatch, kernel, sig, cons, grid, mode="1", icache=None, capacity=None):
    monkeypatch.setenv("PITON_REROLL_REDUCE", "0" if mode is None else mode)
    if icache is not None:
        monkeypatch.setattr(reroll, "ICACHE_INSTRUCTIONS", icache)
    if capacity is not None:
        monkeypatch.setattr(reroll, "ICACHE_CAPACITY", capacity)
    bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)
    return compiler.compile(kernel, sig, cons, grid=bound)


def run(ck, grid, values):
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    run_emulator(ck, grid, values)
    return values


def bits(a):
    return a.view(np.uint32)


def on_and_off(monkeypatch, kernel, sig, cons, grid, values, icache=None, capacity=None, off_fails=False):
    """(rerolled kernel, its outputs) after checking them bit for bit against the
    unrolled build (`off_fails`: which runs out of registers instead)."""
    if off_fails:
        with pytest.raises(RegAllocError):
            build(monkeypatch, kernel, sig, cons, grid, None, icache, capacity)
    else:
        off = build(monkeypatch, kernel, sig, cons, grid, None, icache, capacity)
    on = build(monkeypatch, kernel, sig, cons, grid, "1", icache, capacity)
    got = run(on, grid, values)
    if not off_fails:
        for k, v in run(off, grid, values).items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(bits(got[k]), bits(v), err_msg=k)
    return on, got


def softmax_inputs(rows, cols, stride, block):
    x = (rng.standard_normal((rows, stride)) * 3).astype(np.float32)
    out = np.full(rows * stride, np.nan, np.float32)
    return x, {"output_ptr": out, "input_ptr": x.ravel(), "input_row_stride": stride,
               "output_row_stride": stride, "n_cols": cols}


def softmax_ref(x, cols):
    e = np.exp(x[:, :cols].astype(np.float64) - x[:, :cols].max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# -- softmax (perf suite: BLOCK_SIZE=256 on the Pi) -------------------------------------


@pytest.mark.parametrize("block,cols,rows", [(256, 256, 13), (256, 200, 2), (256, 17, 1), (256, 1, 1),
                                             (128, 128, 3), (512, 300, 2)])
def test_softmax_rerolled_bit_identical(monkeypatch, block, cols, rows):
    stride = cols + 3
    x, values = softmax_inputs(rows, cols, stride, block)
    ck, got = on_and_off(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": block}, (rows,), values)
    assert ck.stats["rerolled"] and ck.stats["reroll_carry"] == 1
    assert len(ck.program.instrs) <= reroll.ICACHE_CAPACITY
    out = got["output_ptr"].reshape(rows, stride)
    np.testing.assert_allclose(out[:, :cols], softmax_ref(x, cols), rtol=2e-5, atol=1e-8)
    assert np.isnan(out[:, cols:]).all()                # masked lanes are not written


def test_softmax_suite_case_shape(monkeypatch):
    """The suite's Pi softmax (BLOCK_SIZE=256): 64-element chunks, one loop per pass."""
    ck = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 256}, (256,))
    assert ck.stats["rerolled"] == 64 and ck.stats["reroll_carry"] == 1
    assert ck.asm["ttir_opt"].count("scf.for") == 3
    assert ck.asm["ttir_opt"].count("tt.load") == 1      # the row is loaded once and carried


@pytest.mark.parametrize("name,programs", [("softmax", 32), ("transpose", None)])
def test_suite_cases_at_pi_size(monkeypatch, name, programs):
    """The suite's Pi builds of the affected cases against the suite's references
    (softmax: the first 32 of its 256 rows)."""
    from perf.suite import CASES_BY_NAME, compile_case, default_check
    monkeypatch.setenv("PITON_REROLL_REDUCE", "1")
    ck, case_run = compile_case(CASES_BY_NAME[name], "pi")
    if name == "softmax":
        assert ck.stats["rerolled"] and len(ck.program.instrs) <= reroll.ICACHE_CAPACITY
    else:
        assert "load_operand" in ck.asm["ttir_opt"]
    grid = (programs,) if programs else case_run.grid
    got = run(ck, grid, case_run.inputs.values)
    for k, want in case_run.expected.items():
        n = want.size * grid[0] // case_run.grid[0]
        assert default_check(case_run, k, got[k][:n], want[:n]) is None


@pytest.mark.parametrize("name", ["row_sum", "row_max", "dense_relu", "exp", "idiv"])
def test_other_suite_cases_unchanged(monkeypatch, name):
    """2-D tiles reduced inside a runtime loop (row_sum, row_max: ~200 instructions),
    a dot layer (its reduction is tl.dot in the K loop) and the elementwise
    kernels compile exactly as without the switch."""
    from perf.suite import CASES_BY_NAME, compile_case
    monkeypatch.setenv("PITON_REROLL_REDUCE", "0")
    off, _ = compile_case(CASES_BY_NAME[name], "pi")
    monkeypatch.setenv("PITON_REROLL_REDUCE", "1")
    on, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert on.words == off.words and on.uniforms == off.uniforms


def test_default_rerolls_rows_only(monkeypatch):
    """Unset means "rows": softmax gets its pass loops, transpose keeps the
    plain build (transposed loads measured 0.71x on the Pi)."""
    from perf.suite import CASES_BY_NAME, compile_case
    monkeypatch.delenv("PITON_REROLL_REDUCE", raising=False)
    softmax = compile_case(CASES_BY_NAME["softmax"], "pi")[0]
    monkeypatch.setenv("PITON_REROLL_REDUCE", "rows")
    # the same rewrite (with PITON_ISSUE, which port-aware allocation softmax keeps
    # depends on the order regalloc.py meets registers in sets, i.e. on heap
    # addresses, so two builds in one process can differ, as transpose's do without it)
    assert compile_case(CASES_BY_NAME["softmax"], "pi")[0].asm["ttir_opt"] == softmax.asm["ttir_opt"]
    monkeypatch.setenv("PITON_REROLL_REDUCE", "0")
    assert len(compile_case(CASES_BY_NAME["softmax"], "pi")[0].program.instrs) > len(softmax.program.instrs)
    assert softmax.stats["rerolled"]


def test_softmax_special_values(monkeypatch):
    """NaN, infinities, -0 and denormals, also in the first register (which meets the
    partial's initial value): the same bits as the unrolled reductions."""
    rows, block = 8, 256
    x = rng.standard_normal((rows, block)).astype(np.float32)
    specials = np.array([np.nan, -np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-40, -1e-40], np.float32)
    x[0, :8] = specials
    x[1, 3] = np.inf
    x[2, 0] = -np.inf
    x[3, :] = -0.0
    x[4, 0] = 1e-40
    x[4, 17] = -1e-40
    x[5, 16] = np.nan
    x[6, :] = -np.inf
    x[7, 255] = np.nan
    values = {"output_ptr": np.zeros(rows * block, np.float32), "input_ptr": x.ravel(),
              "input_row_stride": block, "output_row_stride": block, "n_cols": block}
    on_and_off(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": block}, (rows,), values)


def test_softmax_row_too_long_for_registers(monkeypatch):
    """BLOCK_SIZE=1024 does not allocate unrolled; the passes recompute their chunks."""
    with pytest.raises(RegAllocError):
        build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 1024}, (2,), None)
    x, values = softmax_inputs(2, 1000, 1000, 1024)
    ck = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 1024}, (2,))
    assert ck.stats["reroll_carry"] == 0 and len(ck.program.instrs) <= reroll.ICACHE_CAPACITY
    out = run(ck, (2,), values)["output_ptr"].reshape(2, 1000)
    np.testing.assert_allclose(out, softmax_ref(x, 1000), rtol=2e-5, atol=1e-8)


def test_softmax_many_programs(monkeypatch):
    """More programs than QPUs, each row its own program."""
    rows, block = 40, 128
    x, values = softmax_inputs(rows, block, block, block)
    _, got = on_and_off(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": block}, (rows,), values)
    np.testing.assert_allclose(got["output_ptr"].reshape(rows, block), softmax_ref(x, block), rtol=2e-5, atol=1e-8)


# -- other reductions (small rows, rerolled by a lowered threshold) -----------------------


@piton.jit
def layernorm_kernel(x_ptr, out_ptr, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + row * BLOCK + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    tl.store(out_ptr + row * BLOCK + offs, d / tl.sqrt(var + eps), mask=mask)


@pytest.mark.parametrize("block,n,slots", [(256, 256, "1"), (256, 200, "1"), (128, 100, "0"), (128, 127, "0")])
def test_layernorm_three_passes(monkeypatch, block, n, slots):
    """(BLOCK=128 with store slots stays unrolled: 463 instructions, the passes
    with the store queued after them 475, so it is rerolled with private rows.
    With PITON_ISSUE its unrolled build with private rows is 375 instructions (368
    with PITON_PAIR too, 415 with PITON_PAIR alone) and fits the cache, so a
    threshold of 360 keeps the passes; 461 without either.)"""
    monkeypatch.setenv("PITON_STORE_SLOTS", slots)
    rows = 3
    x = (rng.standard_normal((rows, block)) * 2 + 1).astype(np.float32)
    values = {"x_ptr": x.ravel(), "out_ptr": np.full(rows * block, np.nan, np.float32), "n": n, "eps": 1e-5}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "eps": "fp32"}
    ck, got = on_and_off(monkeypatch, layernorm_kernel, sig, {"BLOCK": block}, (rows,), values,
                         icache=360 if block == 128 else None)
    assert ck.stats["rerolled"] and ck.stats["reroll_carry"] == 1
    xs = x[:, :n].astype(np.float64)
    d = xs - xs.mean(axis=1, keepdims=True)
    ref = d / np.sqrt((d * d).mean(axis=1, keepdims=True) + 1e-5)
    out = got["out_ptr"].reshape(rows, block)
    np.testing.assert_allclose(out[:, :n], ref, rtol=1e-4, atol=1e-4)
    assert np.isnan(out[:, n:]).all()


@piton.jit
def minmax_kernel(x_ptr, out_ptr, stats_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    lo = tl.min(x, axis=0)
    hi = tl.max(x, axis=0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + offs, (x - lo) / (hi - lo))
    tl.store(stats_ptr + row * 3, lo)
    tl.store(stats_ptr + row * 3 + 1, hi)
    tl.store(stats_ptr + row * 3 + 2, s)


def test_independent_reductions_share_a_loop(monkeypatch):
    rows, block = 4, 256
    x = rng.standard_normal((rows, block)).astype(np.float32)
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(rows * block, np.float32),
              "stats_ptr": np.zeros(rows * 3, np.float32)}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "stats_ptr": "*fp32"}
    ck, got = on_and_off(monkeypatch, minmax_kernel, sig, {"BLOCK": block}, (rows,), values, icache=200)
    assert ck.stats["rerolled"] and ck.asm["ttir_opt"].count("scf.for") == 2
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), (x - lo) / (hi - lo), rtol=1e-5, atol=1e-6)
    stats = got["stats_ptr"].reshape(rows, 3)
    np.testing.assert_array_equal(stats[:, :2], np.concatenate([lo, hi], axis=1))
    np.testing.assert_allclose(stats[:, 2], x.astype(np.float64).sum(axis=1), rtol=1e-5, atol=1e-5)


@piton.jit
def int_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * BLOCK + offs, mask=offs < n, other=0)
    tl.store(out_ptr + row * 3, tl.sum(x, axis=0))
    tl.store(out_ptr + row * 3 + 1, tl.max(x, axis=0))
    tl.store(out_ptr + row * 3 + 2, tl.min(x, axis=0))


@pytest.mark.parametrize("ptr,dtype", [("*i32", np.int32), ("*u32", np.uint32)])
@pytest.mark.parametrize("n", [256, 45])
def test_integer_reductions(monkeypatch, ptr, dtype, n):
    """Wrapping sums and signed / unsigned max and min: scalar stores only (one loop)."""
    rows, block = 3, 256
    x = rng.integers(0, 2 ** 32, (rows, block), dtype=np.uint64).astype(np.uint32).view(dtype)
    x[0, 0] = np.iinfo(dtype).min
    x[1, 1] = np.iinfo(dtype).max
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(rows * 3, dtype), "n": n}
    sig = {"x_ptr": ptr, "out_ptr": ptr, "n": "i32"}
    ck, got = on_and_off(monkeypatch, int_reduce_kernel, sig, {"BLOCK": block}, (rows,), values, icache=100)
    assert ck.stats["rerolled"] and ck.asm["ttir_opt"].count("scf.for") == 1
    xs = np.where(np.arange(block) < n, x, 0).astype(dtype)
    total = (xs.astype(np.int64).sum(axis=1) & 0xFFFFFFFF).astype(np.uint32).view(dtype)
    ref = np.stack([total, xs.max(axis=1), xs.min(axis=1)], axis=1)
    np.testing.assert_array_equal(got["out_ptr"].reshape(rows, 3), ref)


@piton.jit
def fstats_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * BLOCK + tl.arange(0, BLOCK))
    tl.store(out_ptr + row * 3, tl.sum(x, axis=0))
    tl.store(out_ptr + row * 3 + 1, tl.max(x, axis=0))
    tl.store(out_ptr + row * 3 + 2, tl.min(x, axis=0))


def test_float_reductions_of_special_values(monkeypatch):
    """The partial's initial values (+0, and the extremes of the fmax / fmin order)
    leave every special value where the unrolled code puts it."""
    block = 256
    specials = [np.nan, -np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-40, -1e-40, 3.0, -3.0]
    rows = []
    for s in specials:
        for at in (0, 15, 16, 64, 255):
            r = rng.standard_normal(block).astype(np.float32)
            r[at] = s
            rows.append(r)
    rows.append(np.full(block, -0.0, np.float32))
    rows.append(np.full(block, -np.inf, np.float32))
    rows.append(np.array([np.inf, -np.inf] * (block // 2), np.float32))
    x = np.stack(rows)
    x.view(np.uint32)[3, 5] = 0xFFFFFFFF                 # the fmax initial value itself
    x.view(np.uint32)[4, 9] = 0x7FFFFFFF                 # and the fmin one
    n = len(x)
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(n * 3, np.float32)}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32"}
    ck, _ = on_and_off(monkeypatch, fstats_kernel, sig, {"BLOCK": block}, (n,), values, icache=100)
    assert ck.stats["rerolled"]


@piton.jit
def carried_through_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    m = tl.max(x, axis=0)
    y = tl.load(y_ptr + offs)
    pos = y > m                                          # a mask from loaded values: recomputed
    s = tl.sum(tl.where(pos, 1.0, tl.exp(y - m)), axis=0)
    tl.store(out_ptr + offs, tl.where(pos, -x, x * s + y))


@pytest.mark.parametrize("block,capacity,sub", [(64, None, 32), (128, None, 64), (128, 400, 32)])
def test_value_carried_past_a_pass(monkeypatch, block, capacity, sub):
    """x is loaded for the first pass and used again in the third: the second pass
    rotates its slots; y is first needed (and loaded) in the second pass. (At
    BLOCK=128 the store queued after the passes takes 463 instructions in
    64-element chunks and 330 in 32-element ones.)"""
    rows = 3
    x = rng.standard_normal((rows, block)).astype(np.float32)
    y = (rng.standard_normal((rows, block)) * 2).astype(np.float32)
    values = {"x_ptr": x.ravel(), "y_ptr": y.ravel(), "out_ptr": np.zeros(rows * block, np.float32)}
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"}
    ck, got = on_and_off(monkeypatch, carried_through_kernel, sig, {"BLOCK": block}, (rows,), values,
                         icache=200, capacity=capacity)
    assert ck.stats["rerolled"] == sub and ck.stats["reroll_carry"] == 1
    assert ck.asm["ttir_rerolled"].count("tt.load") == 2
    m = x.max(axis=1, keepdims=True).astype(np.float64)
    pos = y > m
    s = np.where(pos, 1.0, np.exp(y - m)).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), np.where(pos, -x, x * s + y),
                               rtol=1e-5, atol=1e-5)


@piton.jit
def int_carry_kernel(x_ptr, out_ptr, cnt_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    mx = tl.max(x, axis=0)
    eq = x == mx
    c = tl.sum(tl.where(eq, 1, 0), axis=0)
    tl.store(out_ptr + offs, tl.where(eq, x * 3 + c, x - mx))
    tl.store(cnt_ptr + row, c)


def test_integer_row_carried_through_passes(monkeypatch):
    """An int32 row carried through two reduction passes into the stores; the mask
    built from it and the reduced maximum is recomputed in each pass."""
    rows, block = 4, 256
    x = rng.integers(-5, 5, (rows, block)).astype(np.int32)
    x[0, 0] = np.iinfo(np.int32).min
    x[1, :] = np.iinfo(np.int32).min
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(rows * block, np.int32), "cnt_ptr": np.zeros(rows, np.int32)}
    sig = {"x_ptr": "*i32", "out_ptr": "*i32", "cnt_ptr": "*i32"}
    ck, got = on_and_off(monkeypatch, int_carry_kernel, sig, {"BLOCK": block}, (rows,), values, icache=100)
    assert ck.stats["rerolled"] and ck.stats["reroll_carry"] == 1
    assert ck.asm["ttir_rerolled"].count("tt.load") == 1
    mx = x.max(axis=1, keepdims=True)
    c = (x == mx).sum(axis=1, keepdims=True).astype(np.int32)
    with np.errstate(over="ignore"):
        want = np.where(x == mx, x * 3 + c, x - mx)
    np.testing.assert_array_equal(got["out_ptr"].reshape(rows, block), want)
    np.testing.assert_array_equal(got["cnt_ptr"], c.ravel())


@piton.jit
def gather_reduce_kernel(x_ptr, idx_ptr, out_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    v = tl.load(x_ptr + idx)
    tl.store(out_ptr + offs, v - tl.min(v, axis=0) + tl.max(idx, axis=0))


def test_gathered_row(monkeypatch):
    """Pointers computed from a loaded index row: the gathered values are carried
    from the reduction pass to the store pass."""
    rows, block = 3, 256
    table = rng.standard_normal(1000).astype(np.float32)
    idx = rng.integers(0, 1000, rows * block).astype(np.int32)
    values = {"x_ptr": table, "idx_ptr": idx, "out_ptr": np.zeros(rows * block, np.float32)}
    sig = {"x_ptr": "*fp32", "idx_ptr": "*i32", "out_ptr": "*fp32"}
    ck, got = on_and_off(monkeypatch, gather_reduce_kernel, sig, {"BLOCK": block}, (rows,), values, icache=100)
    assert ck.stats["rerolled"] and ck.asm["ttir_rerolled"].count("tt.load") == 2
    v = table[idx].reshape(rows, block)
    want = v - v.min(axis=1, keepdims=True) + idx.reshape(rows, block).max(axis=1, keepdims=True)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), want, rtol=1e-6, atol=1e-6)


def test_recompute_fallback(monkeypatch):
    """Without carrying (reroll.run(..., carry=False)) the passes reload their chunks;
    still the unrolled kernel's bits."""
    real = reroll.run
    monkeypatch.setattr(reroll, "run", lambda fn, sub, reductions=False, carry=True, queue=False:
                        real(fn, sub, reductions, False, queue))
    rows, block = 3, 256
    x, values = softmax_inputs(rows, 250, block, block)
    ck, got = on_and_off(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": block}, (rows,), values)
    assert ck.stats["rerolled"] and ck.asm["ttir_rerolled"].count("tt.load") == 3
    np.testing.assert_allclose(got["output_ptr"].reshape(rows, block)[:, :250], softmax_ref(x, 250),
                               rtol=2e-5, atol=1e-8)


@piton.jit
def load_after_store_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x / tl.sum(x, axis=0))
    y = tl.load(out_ptr + offs)
    tl.store(x_ptr + offs, y + tl.max(x, axis=0))


@piton.jit
def scalar_store_first_kernel(x_ptr, out_ptr, s_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)
    tl.store(s_ptr + tl.program_id(0), s)
    tl.store(out_ptr + offs, x - s)


@pytest.mark.parametrize("kernel,sig", [
    (load_after_store_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32"}),
    (scalar_store_first_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32", "s_ptr": "*fp32"}),
])
def test_reordering_memory_accesses_is_left_alone(monkeypatch, kernel, sig):
    off = build(monkeypatch, kernel, sig, {"BLOCK": 64}, (2,), None, icache=32)
    on = build(monkeypatch, kernel, sig, {"BLOCK": 64}, (2,), "1", icache=32)
    assert on.words == off.words and on.stats["rerolled"] == 0


def test_short_rows_are_left_alone(monkeypatch):
    """softmax at the suite's emulator size (BLOCK_SIZE=64) fits the cache unrolled."""
    off = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 64}, (64,), None)
    on = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 64}, (64,))
    assert on.words == off.words and on.uniforms == off.uniforms


@pytest.mark.parametrize("mode", ["rows", "tiles"])
def test_modes_enable_one_half(monkeypatch, mode):
    # (transpose's machine code is compared by its IR: Lowering.permute orders
    # rotations by id(), so two compiles in one process can schedule differently)
    sm_off = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 256}, (256,), None)
    sm = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 256}, (256,), mode)
    tr = build(monkeypatch, transpose_kernel, TRANSPOSE, {"BLOCK_M": 16, "BLOCK_N": 16}, (256,), mode)
    assert (sm.words == sm_off.words) == (mode == "tiles")
    assert ("load_operand" in tr.asm["ttir_opt"]) == (mode == "tiles")


# -- transposed loads --------------------------------------------------------------------


@pytest.mark.parametrize("bm,bn", [(16, 16), (32, 16), (16, 32)])
@pytest.mark.parametrize("n_rows,n_cols", [(64, 64), (40, 50), (17, 33), (16, 1), (1, 16)])
def test_transpose_through_transposed_loads(monkeypatch, bm, bn, n_rows, n_cols):
    """16x16 tiles: the unrolled kernel's bits; 32x16 and 16x32 tiles do not
    allocate unrolled (permutation temporaries) but do as transposed loads."""
    sx, so = n_cols + 3, n_rows + 5                     # strides wider than the matrix
    x = rng.standard_normal((n_rows, sx)).astype(np.float32)
    out = np.full((n_cols, so), np.nan, np.float32)
    grid = (piton.cdiv(n_rows, bm) * piton.cdiv(n_cols, bn),)
    values = {"x_ptr": x.ravel(), "out_ptr": out.ravel(), "n_rows": n_rows, "n_cols": n_cols,
              "stride_x": sx, "stride_out": so}
    ck, got = on_and_off(monkeypatch, transpose_kernel, TRANSPOSE, {"BLOCK_M": bm, "BLOCK_N": bn}, grid, values,
                         off_fails=bm * bn > 256)
    assert "load_operand" in ck.asm["ttir_opt"]
    res = got["out_ptr"].reshape(n_cols, so)
    np.testing.assert_array_equal(res[:, :n_rows], x[:, :n_cols].T)
    assert np.isnan(res[:, n_rows:]).all()


@piton.jit
def trans_other_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    m = (r[:, None] < n) & (r[None, :] < n - 1)
    t = tl.trans(tl.load(x_ptr + r[:, None] * BLOCK + r[None, :], mask=m, other=r[:, None] * 1.0 - r[None, :]))
    tl.store(out_ptr + r[:, None] * BLOCK + r[None, :], t * 2.0 + 1.0)


def test_transposed_load_mask_and_other(monkeypatch):
    # (port-aware allocation leaves the two builds within -1..9 instructions, -11..22
    # with PITON_ISSUE, by the order regalloc.py meets registers in sets, i.e. by heap
    # addresses; the plain allocation keeps the transposed loads ~100 shorter)
    monkeypatch.setenv("PITON_PORT_ALLOC", "0")
    block, n = 16, 11
    x = rng.standard_normal((block, block)).astype(np.float32)
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(block * block, np.float32), "n": n}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    ck, got = on_and_off(monkeypatch, trans_other_kernel, sig, {"BLOCK": block}, (1,), values, icache=64)
    assert "load_operand" in ck.asm["ttir_opt"]
    # and the transposed build through transposed_build, bit for bit (with
    # port-aware allocation compile() may keep either build: they come within a
    # few instructions of each other, and Lowering.permute orders rotations by id())
    tr = transposed_build(trans_other_kernel, sig, {"BLOCK": block}, (1,))
    assert "load_operand" in tr.asm["ttir_opt"]
    np.testing.assert_array_equal(bits(run(tr, (1,), values)["out_ptr"]), bits(got["out_ptr"]))
    i, j = np.meshgrid(np.arange(block), np.arange(block), indexing="ij")
    src = np.where((i < n) & (j < n - 1), x, (i - j).astype(np.float32))
    np.testing.assert_allclose(got["out_ptr"].reshape(block, block), src.T * 2 + 1, rtol=1e-6)


@piton.jit
def trans_in_loop_kernel(x_ptr, out_ptr, n_tiles, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)[:, None]
    c = tl.arange(0, BLOCK)[None, :]
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for t in range(0, n_tiles):
        acc += tl.trans(tl.load(x_ptr + t * BLOCK * BLOCK + r * BLOCK + c))
    tl.store(out_ptr + r * BLOCK + c, acc)


def test_transposed_load_in_a_loop(monkeypatch):
    """The permutation does not allocate inside the loop; the transposed load does."""
    block, tiles = 16, 3
    x = rng.standard_normal((tiles, block, block)).astype(np.float32)
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(block * block, np.float32), "n_tiles": tiles}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n_tiles": "i32"}
    ck, got = on_and_off(monkeypatch, trans_in_loop_kernel, sig, {"BLOCK": block}, (1,), values, off_fails=True)
    assert "load_operand" in ck.asm["ttir_opt"]
    np.testing.assert_allclose(got["out_ptr"].reshape(block, block), x.sum(axis=0).T, rtol=1e-6, atol=1e-6)


# -- the elementwise reroll (PITON_REROLL, with the switch on or off) ---------------------
#
# It emits every scalar op ahead of the chunk loop, so a scalar load after a
# tensor store, or a scalar store after a tensor load or store, would move
# before them (a read of stale memory, a store the block then overwrites).


@piton.jit
def store_then_scalar_load_kernel(x_ptr, out_ptr, first_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs)))
    tl.store(first_ptr + tl.program_id(0), tl.load(out_ptr + tl.program_id(0) * BLOCK))


@piton.jit
def store_then_scalar_store_kernel(x_ptr, out_ptr, first_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs)))
    tl.store(out_ptr + tl.program_id(0) * BLOCK, -1.0)          # element 0 of the block, last
    tl.store(first_ptr + tl.program_id(0), 2.0)


@piton.jit
def load_then_scalar_store_kernel(x_ptr, out_ptr, first_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + tl.program_id(0) * BLOCK, -1.0)            # after the block was read
    tl.store(out_ptr + offs, tl.exp(x))
    tl.store(first_ptr + tl.program_id(0), 2.0)


@piton.jit
def scalar_load_first_kernel(x_ptr, out_ptr, first_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(first_ptr + tl.program_id(0))
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs)) * scale)


@pytest.mark.parametrize("mode", [None, "1"])
@pytest.mark.parametrize("kernel,rerolled", [(store_then_scalar_load_kernel, False),
                                             (store_then_scalar_store_kernel, False),
                                             (load_then_scalar_store_kernel, False),
                                             (scalar_load_first_kernel, True)])
def test_elementwise_reroll_keeps_scalar_memory_order(monkeypatch, mode, kernel, rerolled):
    rows, block = 2, 256
    x = rng.standard_normal(rows * block).astype(np.float32)
    first = np.full(rows, 0.5, np.float32)
    values = {"x_ptr": x, "out_ptr": np.zeros(rows * block, np.float32), "first_ptr": first}
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "first_ptr": "*fp32"}
    ck = build(monkeypatch, kernel, sig, {"BLOCK": block}, (rows,), mode, icache=20)
    assert bool(ck.stats["rerolled"]) == rerolled
    got = run(ck, (rows,), values)
    e = np.exp(x.astype(np.float64)).reshape(rows, block)
    out = got["out_ptr"].reshape(rows, block)
    if kernel is scalar_load_first_kernel:
        np.testing.assert_allclose(out, e * 0.5, rtol=1e-6)
        return
    if kernel is store_then_scalar_load_kernel:
        np.testing.assert_array_equal(got["first_ptr"], out[:, 0])
    else:
        np.testing.assert_array_equal(got["first_ptr"], [2.0, 2.0])
    if kernel is store_then_scalar_store_kernel:
        assert (out[:, 0] == -1.0).all()
        out = out[:, 1:]
        e = e[:, 1:]
    if kernel is load_then_scalar_store_kernel:
        assert (got["x_ptr"].reshape(rows, block)[:, 0] == -1.0).all()
    np.testing.assert_allclose(out, e, rtol=1e-6)


# -- review: carried slots read by pipelined loads, and odd transposed tiles -------------


@piton.jit
def rot3_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    m = tl.max(x, axis=0)
    s = tl.sum(x - m, axis=0)
    v = tl.sum((x - m) * (x - m) + s, axis=0)
    tl.store(out_ptr + offs, x * v + s)


def test_row_rotated_through_two_passes(monkeypatch):
    """x is read by all four passes: the two middle ones rotate its slots fully."""
    rows, block = 3, 256
    x = rng.standard_normal((rows, block)).astype(np.float32)
    values = {"x_ptr": x.ravel(), "out_ptr": np.zeros(rows * block, np.float32)}
    ck, got = on_and_off(monkeypatch, rot3_kernel, {"x_ptr": "*fp32", "out_ptr": "*fp32"}, {"BLOCK": block},
                         (rows,), values, icache=60)
    assert ck.stats["rerolled"] and ck.stats["reroll_carry"] == 1 and ck.asm["ttir_opt"].count("scf.for") == 4
    xd = x.astype(np.float64)
    m = xd.max(1, keepdims=True)
    s = (xd - m).sum(1, keepdims=True)
    v = ((xd - m) ** 2 + s).sum(1, keepdims=True)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), xd * v + s, rtol=1e-3, atol=1e-2)


@piton.jit
def slot_gather_kernel(idx_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + offs)
    k = tl.max(idx, axis=0)
    v = tl.load(x_ptr + tl.minimum(idx, k))             # loaded in the second pass, through idx's slots
    w = idx * 1000003 + 7                               # takes over idx's slots there: no valid index
    s = tl.sum(v, axis=0) + tl.sum(w, axis=0).to(tl.float32)
    tl.store(out_ptr + offs, w.to(tl.float32) + s)


def test_prefetch_through_handed_over_slots(monkeypatch):
    """The second pass's loads are prefetched through slot 0 of idx while w shifts
    into its slots: the last prefetch must reuse the current chunk's addresses (an
    out-of-range read would fault in the emulator)."""
    rows, block = 2, 256
    table = rng.standard_normal(300).astype(np.float32)
    idx = rng.integers(0, 300, rows * block).astype(np.int32)
    values = {"idx_ptr": idx, "x_ptr": table, "out_ptr": np.zeros(rows * block, np.float32)}
    sig = {"idx_ptr": "*i32", "x_ptr": "*fp32", "out_ptr": "*fp32"}
    ck, got = on_and_off(monkeypatch, slot_gather_kernel, sig, {"BLOCK": block}, (rows,), values, icache=60,
                         off_fails=True)
    assert ck.stats["rerolled"] and ck.stats["reroll_carry"] == 1
    ii = idx.reshape(rows, block).astype(np.int64)
    w = (ii * 1000003 + 7 + 2 ** 31) % 2 ** 32 - 2 ** 31
    s = table[idx].reshape(rows, block).astype(np.float64).sum(1, keepdims=True) + \
        ((w.sum(1, keepdims=True) + 2 ** 31) % 2 ** 32 - 2 ** 31)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block), w.astype(np.float32) + s, rtol=1e-5)


def transposed_build(kernel, sig, cons, grid):
    """The transposed-load build even where compile() keeps the unrolled one."""
    fn = compiler._pyfn(kernel)
    pid_max = tuple(g - 1 for g in grid) + (0,) * (3 - len(grid))
    tile, prog, mf, lw, stats, asm = compiler._compile_tile(fn, sig, cons, kernel.__name__, pid_max,
                                                            {"num_stages": 2, "store_mutex": True}, transposed=True)
    return compiler.CompiledKernel(tile.name, prog, list(lw.uniforms), dict(sig), cons, asm, stats)


@piton.jit
def trans_dense_mask_kernel(x_ptr, m_ptr, out_ptr, n, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, BM)[:, None]
    c = tl.arange(0, BN)[None, :]
    mask = ((tl.load(m_ptr + r * BN + c) > 0) | (r >= n)) & ~(c == 1)
    t = tl.trans(tl.load(x_ptr + r * BN + c, mask=mask, other=float("nan")))
    tl.store(out_ptr + tl.arange(0, BN)[:, None] * BM + tl.arange(0, BM)[None, :], t)


@pytest.mark.parametrize("bm,bn", [(16, 16), (8, 4), (2, 8)])
def test_transposed_load_of_a_loaded_mask(monkeypatch, bm, bn):
    """A mask computed from loaded values (or-ed, negated): its registers are permuted."""
    n = bm // 2
    x = rng.standard_normal((bm, bn)).astype(np.float32)
    mv = rng.integers(-1, 2, (bm, bn)).astype(np.int32)
    values = {"x_ptr": x.ravel(), "m_ptr": mv.ravel(), "out_ptr": np.zeros(bm * bn, np.float32), "n": n}
    sig = {"x_ptr": "*fp32", "m_ptr": "*i32", "out_ptr": "*fp32", "n": "i32"}
    off = run(build(monkeypatch, trans_dense_mask_kernel, sig, {"BM": bm, "BN": bn}, (1,), None), (1,), values)
    ck = transposed_build(trans_dense_mask_kernel, sig, {"BM": bm, "BN": bn}, (1,))
    assert "load_operand" in ck.asm["ttir_opt"]
    got = run(ck, (1,), values)
    np.testing.assert_array_equal(bits(got["out_ptr"]), bits(off["out_ptr"]))
    r, c = np.arange(bm)[:, None], np.arange(bn)[None, :]
    want = np.where(((mv > 0) | (r >= n)) & (c != 1), x, np.nan).T
    np.testing.assert_array_equal(got["out_ptr"].reshape(bn, bm), want)


@piton.jit
def trans_wide_stride_kernel(x_ptr, out_ptr, n_rows, n_cols, sx, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    m = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    t = tl.trans(tl.load(x_ptr + rows[:, None] * sx + cols[None, :], mask=m, other=-7))
    tl.store(out_ptr + cols[:, None] * BM + rows[None, :], t)


def test_transposed_load_across_16_mb(monkeypatch):
    """Rows 8.4 MB apart: the per-lane offsets of a transposed load exceed 24 bits."""
    bm, bn, n_rows, n_cols, sx = 16, 16, 3, 13, 2_100_000
    x = np.zeros(sx * (n_rows - 1) + bn, np.int32)
    src = rng.integers(-100, 100, (n_rows, bn)).astype(np.int32)
    for i in range(n_rows):
        x[i * sx:i * sx + bn] = src[i]
    values = {"x_ptr": x, "out_ptr": np.zeros(bm * bn, np.int32), "n_rows": n_rows, "n_cols": n_cols, "sx": sx}
    sig = {"x_ptr": "*i32", "out_ptr": "*i32", "n_rows": "i32", "n_cols": "i32", "sx": "i32"}
    ck, got = on_and_off(monkeypatch, trans_wide_stride_kernel, sig, {"BM": bm, "BN": bn}, (1,), values, icache=10)
    assert "load_operand" in ck.asm["ttir_opt"]
    want = np.full((bm, bn), -7, np.int32)
    want[:n_rows, :n_cols] = src[:, :n_cols]
    np.testing.assert_array_equal(got["out_ptr"].reshape(bn, bm), want.T)


# -- with store slots (PITON_STORE_SLOTS, on by default): stores after the passes ----------


def dma_sizes(monkeypatch):
    """Bytes of every VDW DMA the emulator starts from now on."""
    sizes, real = [], emulator.QPU.start_vdw

    def start_vdw(self, addr):
        before = self.stats["vdw_bytes"]
        real(self, addr)
        sizes.append(self.stats["vdw_bytes"] - before)
    monkeypatch.setattr(emulator.QPU, "start_vdw", start_vdw)
    return sizes


def slot_on_and_off(monkeypatch, kernel, sig, cons, grid, values, seed=0, icache=None, capacity=None):
    """(build with the switch and store slots on, its outputs launched with contended
    DMAs) after checking the protocol on every path and the outputs bit for bit
    against the unrolled slot build and the passes with private VPM rows."""
    monkeypatch.setenv("PITON_STORE_SLOTS", "0")
    private = build(monkeypatch, kernel, sig, cons, grid, "1", icache, capacity)
    monkeypatch.setenv("PITON_STORE_SLOTS", "1")
    off = build(monkeypatch, kernel, sig, cons, grid, None, icache, capacity)
    on = build(monkeypatch, kernel, sig, cons, grid, "1", icache, capacity)
    assert on.stats.get("store_slots") == 1 and protocol_errors(on.program) == []
    wants = [run(off, grid, values), run(private, grid, values)]
    got = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}
    launch(on, grid, got, contended(seed))
    for want in wants:
        for k, v in want.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(bits(got[k]), bits(v), err_msg=k)
    return on, got


@pytest.mark.parametrize("cols,rows", [(256, 13), (200, 3), (1, 1)])
def test_softmax_store_queued_after_the_passes(monkeypatch, cols, rows):
    """The suite's Pi softmax (BLOCK_SIZE=256): the division pass carries its
    quotients out in the slots of the exponentials, and the row is stored after
    the last loop through a store slot, one 1 KiB DMA (not four of 256 bytes);
    64-element chunks, since 32-element ones (8 chunks) would lose the queue."""
    stride = cols + 3
    x, values = softmax_inputs(rows, cols, stride, 256)
    ck, got = slot_on_and_off(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 256}, (rows,), values, cols)
    assert ck.stats["rerolled"] == 64 and ck.stats["reroll_carry"] == 1
    assert len(ck.program.instrs) <= reroll.ICACHE_CAPACITY
    ttir = ck.asm["ttir_opt"]
    assert ttir.count("scf.for") == 3 and ttir.count("tt.load") == 1 and ttir.count("tt.store") == 1
    assert ttir.index("piton.concat") > ttir.rindex("scf.yield") and ttir.index("tt.store") > ttir.rindex("scf.yield")
    if cols == 256:
        sizes = dma_sizes(monkeypatch)
        run(ck, (rows,), values)
        assert sizes == [1024] * rows
    out = got["output_ptr"].reshape(rows, stride)
    np.testing.assert_allclose(out[:, :cols], softmax_ref(x, cols), rtol=2e-5, atol=1e-8)
    assert np.isnan(out[:, cols:]).all()


def test_transpose_stores_through_a_slot(monkeypatch):
    """The transposed-load tile is stored through a store slot: one 1 KiB DMA per tile."""
    n = 48
    t = rng.standard_normal((n, n)).astype(np.float32)
    values = {"x_ptr": t.ravel(), "out_ptr": np.full(n * n, np.nan, np.float32), "n_rows": n, "n_cols": n,
              "stride_x": n, "stride_out": n}
    grid = ((n // 16) ** 2,)
    ck, got = slot_on_and_off(monkeypatch, transpose_kernel, TRANSPOSE, {"BLOCK_M": 16, "BLOCK_N": 16}, grid, values)
    assert "load_operand" in ck.asm["ttir_opt"]
    sizes = dma_sizes(monkeypatch)
    run(ck, grid, values)
    assert sizes == [1024] * grid[0]
    np.testing.assert_array_equal(got["out_ptr"].reshape(n, n), t.T)


@piton.jit
def stored_and_divided_kernel(x_ptr, e_ptr, out_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + row * BLOCK + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    s = tl.sum(e, axis=0)
    tl.store(e_ptr + row * BLOCK + offs, e, mask=mask)         # e as the second pass computed it
    tl.store(out_ptr + row * BLOCK + offs, e / s, mask=mask)


@piton.jit
def stored_as_carried_kernel(x_ptr, e_ptr, s_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + row * BLOCK + offs, mask=mask, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    s = tl.sum(e, axis=0)
    tl.store(e_ptr + row * BLOCK + offs, e, mask=mask)         # nothing left to compute per chunk
    tl.store(s_ptr + row, s)


@pytest.mark.parametrize("n", [256, 77])
def test_queued_stores_of_carried_values(monkeypatch, n):
    """A stored value an earlier pass computed: rotated through the store loop when
    the loop also needs it (and the quotient gets fresh slots; two queued stores
    take 685 instructions in 64-element chunks, so the cache limit is raised), or
    stored from its slots with no store loop at all; the scalar store stays last."""
    rows, block = 3, 256
    x = (rng.standard_normal((rows, block)) * 3).astype(np.float32)
    e = np.exp(x[:, :n].astype(np.float64) - x[:, :n].max(axis=1, keepdims=True))
    values = {"x_ptr": x.ravel(), "e_ptr": np.full(rows * block, np.nan, np.float32),
              "out_ptr": np.full(rows * block, np.nan, np.float32), "n": n}
    sig = {"x_ptr": "*fp32", "e_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}
    ck, got = slot_on_and_off(monkeypatch, stored_and_divided_kernel, sig, {"BLOCK": block}, (rows,), values, n,
                              capacity=700)
    assert ck.stats["rerolled"] == 64 and ck.asm["ttir_opt"].count("scf.for") == 3
    assert ck.asm["ttir_rerolled"].count("piton.concat") == 2
    np.testing.assert_allclose(got["e_ptr"].reshape(rows, block)[:, :n], e, rtol=4e-6, atol=1e-30)
    np.testing.assert_allclose(got["out_ptr"].reshape(rows, block)[:, :n], e / e.sum(axis=1, keepdims=True),
                               rtol=2e-5, atol=1e-8)
    values = {"x_ptr": x.ravel(), "e_ptr": np.full(rows * block, np.nan, np.float32),
              "s_ptr": np.zeros(rows, np.float32), "n": n}
    sig = {"x_ptr": "*fp32", "e_ptr": "*fp32", "s_ptr": "*fp32", "n": "i32"}
    ck, got = slot_on_and_off(monkeypatch, stored_as_carried_kernel, sig, {"BLOCK": block}, (rows,), values, n + 1)
    ttir = ck.asm["ttir_rerolled"]
    assert ck.stats["rerolled"] == 64 and ttir.count("scf.for") == 2 and "piton.concat" in ttir
    assert ttir.rindex("tt.store") > ttir.index("piton.concat")
    np.testing.assert_allclose(got["e_ptr"].reshape(rows, block)[:, :n], e, rtol=4e-6, atol=1e-30)
    np.testing.assert_allclose(got["s_ptr"], e.sum(axis=1), rtol=2e-5)


def test_slot_fallback_keeps_the_rewrite(monkeypatch):
    """When a store slot's registers (or the queued store's) do not fit, the build
    falls back to private VPM rows (compiler._compile_tile): the fallback keeps
    the passes and the transposed loads of the build it replaces."""
    real = compiler._build

    # pass on the pre-scheduling variant (_build's alphas and flags, PITON_ISSUE)
    def no_slots(tile, ranges, options, interleave, *args, **kw):
        if options.get("store_slots"):
            raise RegAllocError("store slot registers (test)")
        return real(tile, ranges, options, interleave, *args, **kw)
    monkeypatch.setattr(compiler, "_build", no_slots)
    ck = build(monkeypatch, softmax_kernel, SOFTMAX, {"BLOCK_SIZE": 256}, (13,))
    assert ck.stats["rerolled"] == 64 and ck.stats["reroll_carry"] == 1 and not ck.stats.get("store_slots")
    assert ck.asm["ttir_opt"].count("scf.for") == 3 and "piton.concat" not in ck.asm["ttir_opt"]
    x, values = softmax_inputs(13, 250, 256, 256)
    out = run(ck, (13,), values)["output_ptr"].reshape(13, 256)
    np.testing.assert_allclose(out[:, :250], softmax_ref(x, 250), rtol=2e-5, atol=1e-8)
    ck = build(monkeypatch, transpose_kernel, TRANSPOSE, {"BLOCK_M": 16, "BLOCK_N": 16}, (9,))
    assert "load_operand" in ck.asm["ttir_opt"] and not ck.stats.get("store_slots")


@pytest.mark.parametrize("name", ["softmax", "transpose"])
def test_suite_pi_builds_assemble_with_vc4asm(monkeypatch, name):
    """The Pi-size binaries the switch changes (store slots and port-aware
    allocation on): vc4asm assembles the .qasm to the same words and its
    verifier (-V) reports nothing; the slot protocol holds on every path."""
    from perf.suite import CASES_BY_NAME, compile_case
    from piton.qpu.asm import find_vc4asm, vc4asm_assemble
    monkeypatch.setenv("PITON_REROLL_REDUCE", "1")
    ck, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert ck.stats.get("store_slots") == 1 and protocol_errors(ck.program) == []
    if find_vc4asm() is None:
        pytest.skip("vc4asm not installed")
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert words == [int(w) for w in ck.words]
    assert not warnings, warnings[:5]
