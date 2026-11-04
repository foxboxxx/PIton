"""Port-aware register allocation (PITON_PORT_ALLOC=1, regalloc.py "Port-aware
file choice"): the read-port model, the fallback to the plain allocation, and
the affected kernels in the strict emulator with the switch on, including
ragged tails and tiny / large grids."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, ROOT)
import piton  # noqa: E402
import piton.language as tl  # noqa: E402
from piton import compiler  # noqa: E402
from piton.backend import regalloc  # noqa: E402
from piton.backend.mir import MOp, VReg  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from piton.qpu.emulator import V3D  # noqa: E402
from piton.runtime.launcher import Launcher, run_emulator  # noqa: E402
import kernels as K  # noqa: E402
import primitive_kernels as P  # noqa: E402
from sfu_tests import sfu_kernel  # noqa: E402
from mandelbrot import TARGET, escape_counts, reference_counts  # noqa: E402
from perf.suite import CASES_BY_NAME, compile_case, default_check  # noqa: E402

rng = np.random.default_rng(11)

TARGETS = ["mandel_32x16", "mandel_16x16x2", "softmax", "compute_peak", "exp", "log", "sqrt", "idiv",
           "matmul_8x64x8"]


@pytest.fixture
def port_alloc(monkeypatch):
    monkeypatch.setenv("PITON_PORT_ALLOC", "1")


def _launch(kernel, grid, *args, extra=None, **constants):
    """Compile afresh (the launch cache does not know PITON_* switches) and run."""
    kernel._cache.clear()
    try:
        return Launcher(kernel, grid, extra)(*args, **constants)
    finally:
        kernel._cache.clear()


def _dynamic(case, env, monkeypatch):
    """(executed instructions, stats) of a suite case, checked against numpy."""
    monkeypatch.setenv("PITON_PORT_ALLOC", env)
    ck, run = compile_case(case, "emu")
    v3d = V3D()
    v3d.pc_counts = [0] * len(ck.program.instrs)
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    run_emulator(ck, run.grid, values, v3d=v3d)
    for name, expected in run.expected.items():
        error = default_check(run, name, values[name], expected)
        assert error is None, f"{case.name}.{name} (PITON_PORT_ALLOC={env}): {error}"
    return sum(v3d.pc_counts), ck.stats


# ---------------------------------------------------------------------------
# The read-port model
# ---------------------------------------------------------------------------


def _where(files):
    return lambda v: files.get(v, "acc")


def test_two_ops_reading_a_and_b_registers_cannot_pair():
    a, b, c, d = (VReg(n) for n in "abcd")
    g = regalloc.port_group([("add", MOp("fadd", VReg("x"), [a, b])), ("mul", MOp("fmul", VReg("y"), [c, d]))])
    assert not regalloc.port_fits(g, _where({a: "A", b: "B", c: "A", d: "B"}))
    assert not regalloc.port_fits(g, _where({a: "A", b: "B", c: "B", d: "A"}))
    assert not regalloc.port_fits(g, _where({a: "A", b: "B", d: "B"}))     # c in an accumulator
    assert regalloc.port_fits(g, _where({a: "A", b: "B"}))                  # c, d in accumulators
    assert regalloc.port_fits(g, _where({a: "A", d: "B"}))


def test_shared_operands_small_immediates_and_write_swap():
    a, b, x, y = (VReg(n) for n in "abxy")
    shared = regalloc.port_group([("add", MOp("fadd", x, [a, b])), ("mul", MOp("fmul", y, [a, b]))])
    assert regalloc.port_fits(shared, _where({a: "A", b: "B"}))
    # a small immediate takes the B read port
    imm = regalloc.port_group([("add", MOp("add", x, [a, isa.Imm(1)]))])
    assert regalloc.port_fits(imm, _where({a: "A"}))
    assert not regalloc.port_fits(imm, _where({a: "B"}))
    assert regalloc.port_group([("add", MOp("add", x, [a, isa.Imm(1)])),
                                ("mul", MOp("fmul", y, [b, isa.Imm(2)]))]) is None
    # one write-swap bit: add and mul cannot both write file A
    pair = regalloc.port_group([("add", MOp("fadd", x, [a, a])), ("mul", MOp("fmul", y, [b, b]))])
    assert not regalloc.port_fits(pair, _where({a: "A", b: "B", x: "A", y: "A"}))
    assert regalloc.port_fits(pair, _where({a: "A", b: "B", x: "A", y: "B"}))
    assert regalloc.port_fits(pair, _where({a: "A", b: "B", x: "B", y: "A"}))


def test_clone_keeps_ids_and_leaves_the_original_unallocated():
    case = CASES_BY_NAME["mandel_32x16"]
    from piton import frontend, passes
    from piton.backend import opt
    from piton.backend.lower import Lowering
    tile = frontend.generate(compiler._pyfn(case.kernel), case.signature, case.constants, name=None)
    ranges = passes.optimize(tile, None)
    mf = Lowering(tile, ranges, {"num_stages": 2, "store_mutex": True, "cache_dot_a": True}).run()
    opt.run(mf)
    clone = regalloc.clone_function(mf)
    assert clone.dump() == mf.dump()
    regalloc.allocate(clone, hints=regalloc.PortHints())
    vregs = {x for b in mf.blocks for ins in b.insts for op in ins.ops for x in [op.dst] + op.srcs
             if isinstance(x, VReg)}
    assert vregs and all(v.phys is None for v in vregs)
    assert clone.dump() != mf.dump()


# ---------------------------------------------------------------------------
# Switch and fallback
# ---------------------------------------------------------------------------


def test_switch_defaults_on(monkeypatch):
    case = CASES_BY_NAME["mandel_32x16"]
    monkeypatch.delenv("PITON_PORT_ALLOC", raising=False)
    default = compile_case(case, "emu")[0]
    monkeypatch.setenv("PITON_PORT_ALLOC", "1")
    assert list(compile_case(case, "emu")[0].words) == list(default.words)
    monkeypatch.setenv("PITON_PORT_ALLOC", "0")
    assert "port_alloc" not in compile_case(CASES_BY_NAME["add"], "emu")[0].stats


@pytest.mark.parametrize("error", [regalloc.RegAllocError, RuntimeError])
def test_failed_port_aware_attempt_keeps_the_plain_allocation(monkeypatch, error):
    case = CASES_BY_NAME["mandel_32x16"]
    monkeypatch.setenv("PITON_PORT_ALLOC", "0")
    plain, run = compile_case(case, "emu")
    real = compiler.allocate

    def failing(fn, verbose=False, hints=None):
        if hints is not None:
            raise error("forced")
        return real(fn, verbose)
    monkeypatch.setattr(compiler, "allocate", failing)
    monkeypatch.setenv("PITON_PORT_ALLOC", "1")
    ck, _ = compile_case(case, "emu", run)
    assert ck.stats["port_alloc"] == 0
    assert ck.words == plain.words and ck.uniforms == plain.uniforms


# ---------------------------------------------------------------------------
# Affected suite kernels: correct, and never more instructions executed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TARGETS)
def test_suite_case_with_port_alloc(name, monkeypatch):
    # the plain Mandelbrot loop (PITON_MANDEL_FAST restructures it; there
    # port-aware allocation saves ~13% instead of ~20%)
    monkeypatch.setenv("PITON_MANDEL_FAST", "0")
    case = CASES_BY_NAME[name]
    off, _ = _dynamic(case, "0", monkeypatch)
    on, stats = _dynamic(case, "1", monkeypatch)
    assert stats["port_alloc"] > 0, "no port-aware allocation was chosen"
    assert on <= off, f"{name}: {on} instructions executed with PITON_PORT_ALLOC=1, {off} without"
    if name.startswith("mandel"):
        # read ports were the Mandelbrot loop's limit (perf/README.md, finding 2)
        assert on < 0.85 * off


# ---------------------------------------------------------------------------
# Edge cases with the switch on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,block", [(1, 256), (1000, 256), (4097, 256), (16 * 300 + 5, 16)])
def test_add_ragged_tails_and_grids(port_alloc, n, block):
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.full(n, np.nan, np.float32)
    ck = _launch(K.add_kernel, (piton.cdiv(n, block),), x, y, out, n, BLOCK_SIZE=block)
    assert "port_alloc" in ck.stats
    np.testing.assert_allclose(out, x + y, rtol=1e-6)      # fadd rounds toward zero


@pytest.mark.parametrize("op,fn,lo,hi", [(4, np.exp, -8.0, 8.0), (5, np.sqrt, 0.01, 100.0),
                                         (2, lambda v: 1.0 / v, 0.01, 100.0)])
@pytest.mark.parametrize("n", [5, 777, 4096 + 300])
def test_sfu_ragged(port_alloc, op, fn, lo, hi, n):
    x = rng.uniform(lo, hi, n).astype(np.float32)
    out = np.full(n, np.nan, np.float32)
    ck = _launch(sfu_kernel, (piton.cdiv(n, 256),), x, out, n, OP=op, BLOCK=256)
    assert "port_alloc" in ck.stats
    np.testing.assert_allclose(out, fn(x.astype(np.float64)), rtol=4e-6, atol=1e-30)


def test_log_ragged(port_alloc):
    n = 1000
    x = rng.uniform(0.01, 100.0, n).astype(np.float32)
    out = np.full(n, np.nan, np.float32)
    _launch(P.log_kernel, (piton.cdiv(n, 256),), x, out, n, BLOCK=256)
    np.testing.assert_allclose(out, np.log(x.astype(np.float64)), rtol=4e-6, atol=4e-7)


def test_idiv_ragged(port_alloc):
    n = 700
    a = rng.integers(-2 ** 31, 2 ** 31 - 1, n, dtype=np.int64).astype(np.int32)
    b = (rng.integers(2, 1 << 20, n) * rng.choice([-1, 1], n)).astype(np.int32)
    q = np.zeros(n, np.int32)
    r = np.zeros(n, np.int32)
    ck = _launch(P.idiv_kernel, (piton.cdiv(n, 256),), a, b, q, r, n, BLOCK=256)
    assert ck.stats["port_alloc"] > 0
    want = (np.sign(a.astype(np.int64)) * np.sign(b) * (np.abs(a.astype(np.int64)) // np.abs(b))).astype(np.int32)
    np.testing.assert_array_equal(q, want)
    np.testing.assert_array_equal(r, (a.astype(np.int64) - want.astype(np.int64) * b).astype(np.int32))


@pytest.mark.parametrize("rows,cols,block", [(1, 64, 64), (3, 100, 128), (5, 16, 16), (2, 250, 256)])
def test_softmax_masked_rows(port_alloc, rows, cols, block):
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 4
    y = np.zeros_like(x)
    _launch(K.softmax_kernel, (rows,), y, x, cols, cols, cols, BLOCK_SIZE=block)
    e = np.exp(x.astype(np.float64) - x.max(1, keepdims=True))
    np.testing.assert_allclose(y, e / e.sum(1, keepdims=True), rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("M,N,Kd", [(8, 64, 8), (40, 70, 40), (72, 100, 64)])
def test_matmul_ragged(port_alloc, M, N, Kd):
    a = rng.standard_normal((M, Kd)).astype(np.float32)
    b = rng.standard_normal((Kd, N)).astype(np.float32)
    c = np.full((M, N), np.nan, np.float32)
    grid = (piton.cdiv(M, 8) * piton.cdiv(N, 64),)
    # K fixed at compile time, as in the suite (a runtime K needs more registers)
    _launch(K.matmul_kernel, grid, a, b, c, M, N, Kd, Kd, 1, N, 1, N, 1, extra={"K": Kd},
            BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=8, GROUP_SIZE_M=4)
    np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("block,unroll,tile_h", [(16, 16, 2), (32, 16, 1), (16, 8, 1)])
def test_mandelbrot_partial_last_block(port_alloc, monkeypatch, block, unroll, tile_h):
    width, height, iters = 40, 8, 96         # 40 is not a multiple of the block
    step = np.float32(3.5 / 100 / 640)
    x0 = np.float32(TARGET[0] - width / 2 * float(step))
    y0 = np.float32(TARGET[1] - height / 2 * float(step))
    from mandelbrot import mandelbrot_kernel
    mandelbrot_kernel._cache.clear()
    try:
        got = escape_counts(x0, y0, step, width, height, iters, block, unroll, tile_h)
    finally:
        mandelbrot_kernel._cache.clear()
    assert (got != reference_counts(x0, y0, step, width, height, iters)).mean() < 0.15
    # identical to the plain allocation: the escape counts are exact integers
    monkeypatch.setenv("PITON_PORT_ALLOC", "0")
    try:
        plain = escape_counts(x0, y0, step, width, height, iters, block, unroll, tile_h)
    finally:
        mandelbrot_kernel._cache.clear()
    np.testing.assert_array_equal(got, plain)


def test_compute_peak_one_program(port_alloc):
    case = CASES_BY_NAME["compute_peak"]
    x = rng.uniform(-1, 1, 64).astype(np.float32)
    out = np.zeros(64, np.float32)
    cons = dict(case.constants, ITERS=3)
    ck = piton.compiler.compile(case.kernel, case.signature, cons, grid=(1,))
    assert ck.stats["port_alloc"] > 0
    run_emulator(ck, (1,), {"x_ptr": x, "out_ptr": out})
    ref = 9999.0 + (x.astype(np.float64) - 9999.0) * 0.9999 ** (3 * 16)
    np.testing.assert_allclose(out, ref, rtol=5e-3)


# ---------------------------------------------------------------------------
# Adversarial kernels: the port-aware build must compute exactly what the
# plain build computes (register choices never change arithmetic)
# ---------------------------------------------------------------------------


@piton.jit
def _nested_k(x_ptr, out_ptr, n, a, b, c, d, m_outer, m_inner, BLOCK: tl.constexpr):
    """Nested runtime loops: loop invariants (uniforms, immediates) read at
    depths 1 and 2, loop-carried values, a select."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    acc = tl.zeros([BLOCK], tl.float32)
    t = x
    for i in range(m_outer):
        s = t * a + b
        for j in range(m_inner):
            s = s * c + d
            s = tl.where(s > 3.0, s - 2.5, s)
            acc += s * 0.5
        t = t + 1.0
    tl.store(out_ptr + offs, acc, mask=mask)


@piton.jit
def _intops_k(x_ptr, y_ptr, out_ptr, n, k, sh, BLOCK: tl.constexpr):
    """Integer shifts, bit ops, compares and selects with small immediates."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=-1)
    r = (x ^ y) + (x >> 3) - (y << 2)
    r = tl.where(x < y, r, -r)
    r = tl.maximum(tl.minimum(r, x), y & 127)
    for i in range(k):
        r = r + (r >> sh) - (x | 5)
        r = tl.where(r > y, r - y, r + 7)
        r = r ^ (y >> 1)
    tl.store(out_ptr + offs, r, mask=mask)


@piton.jit
def _floats_k(x_ptr, out_ptr, n, k, BLOCK: tl.constexpr):
    """inf, NaN, -0.0 and denormal inputs through compares, min/max and reductions."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.max(tl.where(mask, x, -1e30), axis=0)
    acc = x
    for i in range(k):
        acc = tl.where(acc != acc, 0.0, acc)
        acc = tl.maximum(tl.minimum(acc * 1.5 - m, 1e30), -1e30) + x * 0.0
    tl.store(out_ptr + offs, acc + tl.sum(tl.where(mask, x, 0.0), axis=0), mask=mask)


def _nested_case(n, block):
    x = rng.standard_normal(n).astype(np.float32)
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "a": "fp32", "b": "fp32", "c": "fp32",
           "d": "fp32", "m_outer": "i32", "m_inner": "i32"}
    return _nested_k, sig, {"BLOCK": block}, dict(x_ptr=x, out_ptr=np.full(n, np.nan, np.float32), n=n,
                                                  a=0.75, b=0.1, c=0.9, d=0.3, m_outer=3, m_inner=4)


def _intops_case(n, block):
    x = rng.integers(-2 ** 31, 2 ** 31 - 1, n, dtype=np.int64).astype(np.int32)
    y = rng.integers(-2 ** 31, 2 ** 31 - 1, n, dtype=np.int64).astype(np.int32)
    x[:4] = [-2 ** 31, 2 ** 31 - 1, 0, -1]
    y[:4] = [2 ** 31 - 1, -2 ** 31, -1, 0]
    sig = {"x_ptr": "*i32", "y_ptr": "*i32", "out_ptr": "*i32", "n": "i32", "k": "i32", "sh": "i32"}
    return _intops_k, sig, {"BLOCK": block}, dict(x_ptr=x, y_ptr=y, out_ptr=np.zeros(n, np.int32), n=n,
                                                  k=5, sh=3)


def _floats_case(n, block):
    x = rng.standard_normal(n).astype(np.float32)
    x[:5] = [np.inf, -np.inf, np.nan, -0.0, 1e-40]
    sig = {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "k": "i32"}
    return _floats_k, sig, {"BLOCK": block}, dict(x_ptr=x, out_ptr=np.full(n, np.nan, np.float32), n=n, k=3)


@pytest.mark.parametrize("make,n,block", [(_nested_case, 197, 64), (_nested_case, 5, 16),
                                          (_intops_case, 197, 64), (_intops_case, 21, 16),
                                          (_floats_case, 53, 16), (_floats_case, 100, 64)])
def test_outputs_bit_identical_to_the_plain_allocation(monkeypatch, make, n, block):
    kernel, sig, cons, vals = make(n, block)
    grid = (piton.cdiv(n, block),)
    bound = tuple(1 << max(0, int(g) - 1).bit_length() for g in grid)
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PORT_ALLOC", env)
        ck = compiler.compile(kernel, sig, cons, grid=bound)
        v = {k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in vals.items()}
        run_emulator(ck, grid, v)
        out[env] = (ck, v)
    (plain, v0), (port, v1) = out["0"], out["1"]
    assert port.stats["port_alloc"] > 0 and list(port.words) != list(plain.words)
    assert port.uniforms == plain.uniforms
    for name, val in vals.items():
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(v1[name].view(np.uint32), v0[name].view(np.uint32), err_msg=name)


@pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")
@pytest.mark.parametrize("name", ["exp", "idiv", "softmax", "mandel_16x16x2", "matmul_8x64x8"])
def test_port_aware_pi_builds_assemble_with_vc4asm(monkeypatch, name):
    """The Pi-size binaries: vc4asm assembles our .qasm to the same words and
    its verifier (-V) reports nothing."""
    monkeypatch.setenv("PITON_PORT_ALLOC", "1")
    ck, _ = compile_case(CASES_BY_NAME[name], "pi")
    assert ck.stats["port_alloc"] > 0
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert words == [int(w) for w in ck.words]
    assert not warnings, warnings[:5]
