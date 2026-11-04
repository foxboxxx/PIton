"""Pairing schedules (PITON_PAIR=1, sched.py "Pairing"): the move to the mul
ALU and what may not move, schedules never longer than the plain list
schedule and free of latency violations, the dot products' split
multiply-add pairs, padding loop-entry hazards in the preheader, and the
suite kernels in the strict emulator with outputs bit-identical to the
switch off, including ragged tails, small dots and partial Mandelbrot
blocks."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, ROOT)
import piton  # noqa: E402
from piton import compiler  # noqa: E402
from piton.backend import sched  # noqa: E402
from piton.backend.mir import MInst, MOp  # noqa: E402
from piton.qpu import isa  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from piton.qpu.emulator import V3D  # noqa: E402
from piton.qpu.isa import AluOp, Imm, Instr  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
import kernels as K  # noqa: E402
from nn_kernels import dense_layer_kernel, dense_tiles  # noqa: E402
from mandelbrot import TARGET, escape_counts, mandelbrot_kernel  # noqa: E402
from perf.suite import CASES, CASES_BY_NAME, compile_case, default_check  # noqa: E402

rng = np.random.default_rng(23)

TARGETS = ["matmul_8x64x8", "matmul_8x64x4", "dense_relu", "mandel_32x16", "mandel_16x16x2", "compute_peak",
           "softmax", "transpose"]


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setenv("PITON_PAIR", "1")


def _node(op, *srcs, dst=isa.R0, cond="always", setf=False, sig=None, rot=None):
    return sched._node_for(0, MInst(ops=[MOp(op, dst, list(srcs), cond=cond, setf=setf, rot=rot)], sig=sig))


# ---------------------------------------------------------------------------
# The switch and Bundle.try_add's move to the mul ALU
# ---------------------------------------------------------------------------


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("PITON_PAIR", raising=False)
    assert not sched.pair_enabled()
    monkeypatch.setenv("PITON_PAIR", "0")
    assert not sched.pair_enabled()
    monkeypatch.setenv("PITON_PAIR", "1")
    assert sched.pair_enabled()


def test_move_steps_over_to_the_mul_alu():
    move = _node("mov", isa.RA[1], dst=isa.R1)
    add = _node("add", isa.R2, isa.RB[3], dst=isa.RA[4])
    plain = sched.Bundle()
    assert plain.try_add(move) and not plain.try_add(add)
    bundle = sched.Bundle(swap=True)
    assert bundle.try_add(move) and bundle.try_add(add)
    ins = bundle.to_instr()
    assert ins.add.op == "add" and ins.mul.op == "mov" and ins.mul.dst == isa.R1
    # the mul ALU's move is `v8min x, x`: every bit of the source
    decoded = isa.decode(isa.encode(isa.canonicalize(ins)))
    assert decoded.mul.op == "v8min" and decoded.mul.a == decoded.mul.b == isa.RA[1]
    v3d = V3D(num_qpus=1)
    q = v3d.qpus[0]
    q.last_written_now, q.last_acc_now, q.last_r5_now = set(), set(), False
    q.ra[1] = np.array([0xFFFFFFFF, 0x80000000, 0x7FC00001, 1] * 4, dtype=np.uint32)
    q.acc[2] = np.arange(16, dtype=np.uint32)
    q.rb[3] = np.full(16, 5, dtype=np.uint32)
    writes = []
    q.exec_alu(decoded, writes)
    for dst, file, value, mask in writes:
        q.commit(dst, file, value, mask)
    np.testing.assert_array_equal(q.acc[1], q.ra[1])
    np.testing.assert_array_equal(q.ra[4], np.arange(16) + 5)


def test_conditional_move_and_tmu_write_may_move():
    for move in (_node("mov", isa.RA[1], dst=isa.RB[2], cond="ifz"), _node("mov", isa.R1, dst=isa.TMU0_S)):
        bundle = sched.Bundle(swap=True)
        assert bundle.try_add(move) and bundle.try_add(_node("sub", isa.R2, isa.R3, dst=isa.R0))
        assert bundle.mul is move.alu_ops[0]


@pytest.mark.parametrize("move", [
    _node("mov", Imm(5), dst=isa.R1),                        # vc4asm: other bits on the mul ALU
    _node("mov", isa.VW_WAIT, dst=isa.NOP),                  # a discarded result
    _node("mov", isa.RA[1], dst=isa.R1, setf=True),          # flags come from the add ALU
    _node("mov", isa.RA[1], dst=isa.VPM),
    _node("mov", isa.RA[1], dst=isa.UNIF_ADDR),
    _node("mov", isa.RA[1], dst=isa.VW_SETUP),
])
def test_what_does_not_move(move):
    bundle = sched.Bundle(swap=True)
    assert bundle.try_add(move)
    before = (bundle.add, bundle.mul)
    assert not bundle.try_add(_node("add", isa.R2, isa.R3, dst=isa.R0))
    assert (bundle.add, bundle.mul) == before


def test_move_is_undone_when_the_pair_does_not_encode():
    move = _node("mov", isa.RA[1], dst=isa.R1)
    add = _node("add", isa.RA[2], isa.RB[3], dst=isa.R0)        # a second regfile-A read
    bundle = sched.Bundle(swap=True)
    assert bundle.try_add(move) and not bundle.try_add(add)
    assert bundle.add is move.alu_ops[0] and bundle.mul is None
    assert add in bundle.rejected
    same_dst = _node("add", isa.R2, isa.R3, dst=isa.R1)
    assert not bundle.try_add(same_dst)
    assert bundle.add is move.alu_ops[0] and bundle.mul is None


# ---------------------------------------------------------------------------
# Block schedules: never longer, every latency kept
# ---------------------------------------------------------------------------


def _violations(instrs):
    """(i, j) pairs closer than the resource model's required distance."""
    res = [sched.instr_res(ins) for ins in instrs]
    return [(i, j) for i in range(len(res)) for j in range(i + 1, min(i + 5, len(res)))
            if sched.required_distance(res[i], res[j]) > j - i]


@pytest.mark.parametrize("name", ["exp", "idiv", "softmax", "transpose", "matmul_8x64x4", "dense_relu",
                                  "mandel_16x16x2", "compute_peak"])
def test_block_schedules_never_longer_and_latency_clean(name, monkeypatch):
    blocks = []
    original = sched.schedule_block

    def record(minsts, *args, **kwargs):
        blocks.append(list(minsts))
        return original(minsts, *args, **kwargs)
    monkeypatch.setattr(sched, "schedule_block", record)
    monkeypatch.setenv("PITON_PAIR", "1")
    compile_case(CASES_BY_NAME[name], "emu")
    monkeypatch.setattr(sched, "schedule_block", original)
    assert blocks
    longer = shorter = 0
    for minsts in blocks[-40:]:
        monkeypatch.setenv("PITON_PAIR", "0")
        plain = sched.schedule_block(minsts)
        monkeypatch.setenv("PITON_PAIR", "1")
        paired = sched.schedule_block(minsts)
        longer += len(paired) > len(plain)
        shorter += len(paired) < len(plain)
        assert not _violations(paired), name
        assert sched._lower_bound(sched.build_dag(minsts)) <= len(paired)
        for ins in paired:
            isa.encode(isa.canonicalize(ins))
    assert not longer
    assert shorter, f"{name}: no block scheduled shorter"


def test_missed_pairs_weighted_by_schedules_built(pair):
    ops = [MInst(ops=[MOp("fmul", isa.RA[i], [isa.RA[i], isa.RB[i]])]) for i in range(4)]
    ops += [MInst(ops=[MOp("fadd", isa.RB[i + 8], [isa.RA[i + 8], isa.RB[i + 4]])]) for i in range(4)]
    missed = []
    sched.schedule_block(ops, missed=missed, weight=100)
    assert missed and all(0 < w <= 100 for w, _ in missed)


def test_kept_pairs_are_those_of_the_schedule_returned(pair):
    """`kept` (compiler.py's second port-aware search) gets the pairs the
    shortest schedule (the earlier one on ties) missed, at their own weight."""
    ops = [MInst(ops=[MOp("fmul", isa.RA[i], [isa.RA[i], isa.RB[i]])]) for i in range(4)]
    ops += [MInst(ops=[MOp("fadd", isa.RB[i + 8], [isa.RA[i + 8], isa.RB[i + 4]])]) for i in range(4)]
    kept = []
    out = sched.schedule_block(ops, weight=100, kept=kept)
    nodes = sched.build_dag(ops)
    best = None
    for swap, balance in sched.PAIR_MODES:
        pairs = []
        body = sched._list_schedule(nodes, pairs, 100, swap, balance)
        if best is None or len(body) < len(best[0]):
            best = (body, pairs)
    assert len(out) == len(best[0])
    assert kept and kept == best[1]


def test_second_port_aware_search_only_with_the_switch(monkeypatch):
    calls = {}
    original = compiler._port_aware

    def record(*args, **kwargs):
        calls[env].append(kwargs.get("kept", False))
        return original(*args, **kwargs)
    monkeypatch.setattr(compiler, "_port_aware", record)
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        calls[env] = []
        compiler.compile(K.add_kernel, sig, {"BLOCK_SIZE": 64}, grid=(4,))
    assert calls["0"] and not any(calls["0"])
    assert calls["1"] == [False, True] * (len(calls["1"]) // 2)


# ---------------------------------------------------------------------------
# Dot products: the add before each broadcast may leave its pair
# ---------------------------------------------------------------------------


def test_dot_marks_the_pair_after_each_broadcast(monkeypatch):
    seen = {}
    original = sched.schedule_block

    def record(minsts, *args, **kwargs):
        seen.setdefault(env, []).append(list(minsts))
        return original(minsts, *args, **kwargs)
    monkeypatch.setattr(sched, "schedule_block", record)
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        compile_case(CASES_BY_NAME["matmul_8x64x4"], "emu")
    assert not any(m.split for block in seen["0"] for m in block)
    splits = 0
    for block in seen["1"]:
        last = None
        for m in block:
            if m.split:
                # the multiply-add pair right after a broadcast, nothing else
                assert m.atomic and [op.op for op in m.ops] == ["fadd", "fmul"] and last == "bcast"
                splits += 1
            elif m.tag == "mac" and m.atomic:
                assert last == "mac"
            if m.tag in ("mac", "bcast"):
                last = m.tag
    assert splits


def test_split_pair_joins_the_broadcast(pair):
    ck, _ = compile_case(CASES_BY_NAME["matmul_8x64x8"], "emu")
    joined = [ins for ins in ck.program.instrs if ins.kind == "alu" and ins.mul is not None
              and ins.mul.rot is not None and ins.add is not None and ins.add.op == "fadd"]
    assert joined, "no broadcast shares its instruction with a product's add"


def test_split_ops_keep_their_order():
    """Scheduled apart, the add still reads the product before the multiply
    overwrites its accumulator (a WAR edge of latency 0)."""
    acc, prod = isa.RA[3], isa.R1
    pair_ = MInst(ops=[MOp("fadd", acc, [acc, prod]), MOp("fmul", prod, [isa.R5, isa.RB[2]])],
                  atomic=True, split=True)
    nodes = sched.build_dag([pair_], split=True)
    assert len(nodes) == 2 and [n.mops[0].op for n in nodes] == ["fadd", "fmul"]
    assert any(p is nodes[0] and lat == 0 for p, lat in nodes[1].preds)
    assert len(sched.build_dag([pair_])) == 1


# ---------------------------------------------------------------------------
# Block boundaries: loop-entry hazards padded in the preheader
# ---------------------------------------------------------------------------


def _loop_blocks(pre_branch=False):
    pre = [Instr(add=AluOp("mov", isa.RA[1], isa.R0))]
    if pre_branch:
        # the write in the branch's last delay slot: on the way to both successors
        pre = [Instr(kind="branch", bcond="anyz", target="out"), Instr.nop(), Instr.nop()] + pre
    loop = [Instr(add=AluOp("add", isa.R1, isa.RA[1], isa.RB[2])),
            Instr(add=AluOp("sub", isa.NOP, isa.R1, isa.RB[3]), setf=True),
            Instr(kind="branch", bcond="anyn", target="loop"), Instr.nop(), Instr.nop(), Instr.nop()]
    blocks = {"pre": pre, "loop": loop, "out": [Instr(add=AluOp("mov", isa.R2, isa.R1))]}
    preds = {"pre": [], "loop": ["pre", "loop"], "out": ["loop"] + (["pre"] if pre_branch else [])}
    return blocks, preds, ["pre", "loop", "out"]


def test_entry_hazard_padded_in_the_preheader(monkeypatch):
    monkeypatch.setenv("PITON_PAIR", "0")
    blocks, preds, order = _loop_blocks()
    sched.fix_block_boundaries(blocks, preds, order)
    assert blocks["loop"][0].comment == "boundary hazard" and len(blocks["pre"]) == 1
    monkeypatch.setenv("PITON_PAIR", "1")
    blocks, preds, order = _loop_blocks()
    loop = list(blocks["loop"])
    sched.fix_block_boundaries(blocks, preds, order)
    assert blocks["loop"] == loop
    assert len(blocks["pre"]) == 2 and blocks["pre"][1].comment == "boundary hazard"


def test_preheader_with_two_successors_pads_the_loop_head(pair):
    blocks, preds, order = _loop_blocks(pre_branch=True)
    sched.fix_block_boundaries(blocks, preds, order)
    assert blocks["loop"][0].comment == "boundary hazard"


# ---------------------------------------------------------------------------
# Kernels in the strict emulator: correct, bit-identical to the switch off
# ---------------------------------------------------------------------------


def _run(case, monkeypatch, env):
    monkeypatch.setenv("PITON_PAIR", env)
    ck, run = compile_case(case, "emu")
    v3d = V3D()
    v3d.pc_counts = [0] * len(ck.program.instrs)
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    run_emulator(ck, run.grid, values, v3d=v3d)
    for name, expected in run.expected.items():
        error = default_check(run, name, values[name], expected)
        assert error is None, f"{case.name}.{name} (PITON_PAIR={env}): {error}"
    return ck, values, sum(v3d.pc_counts)


@pytest.mark.parametrize("name", [c.name for c in CASES])
def test_suite_case_bit_identical_and_never_longer(name, monkeypatch):
    case = CASES_BY_NAME[name]
    off, v_off, n_off = _run(case, monkeypatch, "0")
    on, v_on, n_on = _run(case, monkeypatch, "1")
    assert on.uniforms == off.uniforms and on.coarsen == off.coarsen
    for k, v in v_off.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v_on[k].view(np.uint32), v.view(np.uint32), err_msg=k)
    assert n_on <= n_off, f"{name}: {n_on} instructions executed with PITON_PAIR=1, {n_off} without"
    if name in TARGETS and name != "compute_peak":
        assert n_on < n_off


@pytest.mark.parametrize("name", ["mandel_32x16", "mandel_16x16x2"])
def test_mandel_pi_build_on_the_emulator_image(name, monkeypatch):
    """The Pi builds (their program-id bound changes what the port-aware
    allocation finds) on the emulator-size image: counts bit-identical to
    the switch off and at least 3% fewer instructions (measured 0.948 and
    0.932; mandel_32x16 0.983 with only the first port-aware search)."""
    case = CASES_BY_NAME[name]
    run = case.make("emu")
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        ck, _ = compile_case(case, "pi")
        v3d = V3D()
        v3d.pc_counts = [0] * len(ck.program.instrs)
        values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
        run_emulator(ck, run.grid, values, v3d=v3d)
        for key, expected in run.expected.items():
            error = default_check(run, key, values[key], expected)
            assert error is None, f"{name}.{key} (PITON_PAIR={env}): {error}"
        out[env] = values, sum(v3d.pc_counts)
    for k, v in out["0"][0].items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(out["1"][0][k].view(np.uint32), v.view(np.uint32), err_msg=k)
    assert out["1"][1] <= 0.97 * out["0"][1], (out["1"][1], out["0"][1])


def _bitwise(make, monkeypatch):
    """make() -> (compiled kernel, grid, values); outputs with the switch on
    and off, compared bit for bit (returns the switch-on values)."""
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        ck, grid, vals = make()
        run_emulator(ck, grid, vals)
        out[env] = vals
    for k, v in out["0"].items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(out["1"][k].view(np.uint32), v.view(np.uint32), err_msg=k)
    return out["1"]


@pytest.mark.parametrize("M,N,Kd,bk", [(8, 64, 8, 8), (40, 70, 40, 8), (72, 100, 64, 4), (3, 16, 5, 4)])
def test_matmul_ragged(monkeypatch, M, N, Kd, bk):
    a = rng.standard_normal((M, Kd)).astype(np.float32)
    b = rng.standard_normal((Kd, N)).astype(np.float32)
    sig = {n: ("*fp32" if n.endswith("ptr") else "i32") for n in K.matmul_kernel.arg_names}
    cons = dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=bk, GROUP_SIZE_M=4, K=Kd,
                stride_ak=1, stride_bn=1, stride_cn=1)
    for k in cons:
        sig.pop(k, None)
    grid = (piton.cdiv(M, 8) * piton.cdiv(N, 64),)

    def make():
        ck = compiler.compile(K.matmul_kernel, sig, cons, grid=grid)
        vals = dict(a_ptr=a.copy(), b_ptr=b.copy(), c_ptr=np.full(M * N, np.nan, np.float32), M=M, N=N,
                    stride_am=Kd, stride_bk=N, stride_cm=N)
        return ck, grid, vals
    got = _bitwise(make, monkeypatch)
    np.testing.assert_allclose(got["c_ptr"].reshape(M, N), a @ b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("m,n,k,relu", [(8, 128, 64, True), (9, 17, 11, False), (5, 64, 128, True),
                                        (12, 16, 64, False)])
def test_dense_layers_ragged(monkeypatch, m, n, k, relu):
    x = rng.uniform(0, 1, (m, k)).astype(np.float32)
    w = rng.uniform(-0.3, 0.3, (k, n)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    bm, bn, bk = dense_tiles(n)
    grid = ((m + bm - 1) // bm * ((n + bn - 1) // bn),)
    sig = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    sig["M"] = "i32"
    cons = dict(N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                ADD_BIAS=True, RELU=relu)

    def make():
        ck = compiler.compile(dense_layer_kernel, sig, cons, grid=grid)
        return ck, grid, dict(x_ptr=x.copy(), w_ptr=w.copy(), bias_ptr=bias.copy(),
                              y_ptr=np.full(m * n + 32, -1.5, np.float32), M=m)
    got = _bitwise(make, monkeypatch)
    want = x.astype(np.float64) @ w + bias
    want = np.maximum(want, 0) if relu else want
    np.testing.assert_allclose(got["y_ptr"][:m * n].reshape(m, n), want, rtol=1e-4, atol=1e-4)
    assert (got["y_ptr"][m * n:] == np.float32(-1.5)).all()


@pytest.mark.parametrize("M,N,Kd", [(4, 8, 8), (16, 2, 4), (4, 8, 32)])
def test_small_dots(monkeypatch, M, N, Kd):
    sys.path.insert(0, os.path.dirname(__file__))
    import test_tiles as T
    a = rng.integers(-4, 4, (M, Kd)).astype(np.float32)
    b = rng.integers(-4, 4, (Kd, N)).astype(np.float32)
    c = rng.integers(-4, 4, (M, N)).astype(np.float32)
    sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32"}

    def make():
        ck = compiler.compile(T.dot_k, sig, dict(M=M, N=N, K=Kd, ACC=True))
        return ck, (1,), dict(a_ptr=a.copy(), b_ptr=b.copy(), c_ptr=c.copy(), o_ptr=np.zeros(M * N, np.float32))
    got = _bitwise(make, monkeypatch)
    np.testing.assert_array_equal(got["o_ptr"], (a @ b + c).ravel())


@pytest.mark.parametrize("n,block", [(1, 256), (1000, 256), (16 * 300 + 5, 16)])
def test_add_ragged(monkeypatch, n, block):
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"}
    grid = (piton.cdiv(n, block),)

    def make():
        ck = compiler.compile(K.add_kernel, sig, {"BLOCK_SIZE": block}, grid=grid)
        return ck, grid, dict(x_ptr=x.copy(), y_ptr=y.copy(), output_ptr=np.full(n, np.nan, np.float32),
                              n_elements=n)
    got = _bitwise(make, monkeypatch)
    np.testing.assert_allclose(got["output_ptr"], x + y, rtol=1e-6)


@pytest.mark.parametrize("rows,cols,block", [(1, 64, 64), (3, 100, 128), (2, 250, 256)])
def test_softmax_masked_rows(monkeypatch, rows, cols, block):
    x = (rng.standard_normal((rows, cols)) * 4).astype(np.float32)
    sig = {"output_ptr": "*fp32", "input_ptr": "*fp32", "input_row_stride": "i32",
           "output_row_stride": "i32", "n_cols": "i32"}

    def make():
        ck = compiler.compile(K.softmax_kernel, sig, {"BLOCK_SIZE": block}, grid=(rows,))
        return ck, (rows,), dict(output_ptr=np.zeros(rows * cols, np.float32), input_ptr=x.ravel().copy(),
                                 input_row_stride=cols, output_row_stride=cols, n_cols=cols)
    got = _bitwise(make, monkeypatch)
    e = np.exp(x.astype(np.float64) - x.max(1, keepdims=True))
    np.testing.assert_allclose(got["output_ptr"].reshape(rows, cols), e / e.sum(1, keepdims=True),
                               rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("block,unroll,tile_h", [(16, 16, 2), (32, 16, 1), (16, 8, 1)])
def test_mandelbrot_partial_last_block(monkeypatch, block, unroll, tile_h):
    width, height, iters = 40, 8, 96         # 40 is not a multiple of the block
    step = np.float32(3.5 / 100 / 640)
    x0 = np.float32(TARGET[0] - width / 2 * float(step))
    y0 = np.float32(TARGET[1] - height / 2 * float(step))
    counts = {}
    for env in ("0", "1"):
        monkeypatch.setenv("PITON_PAIR", env)
        mandelbrot_kernel._cache.clear()
        try:
            counts[env] = escape_counts(x0, y0, step, width, height, iters, block, unroll, tile_h)
        finally:
            mandelbrot_kernel._cache.clear()
    np.testing.assert_array_equal(counts["1"], counts["0"])


def test_compute_peak_one_program(monkeypatch):
    case = CASES_BY_NAME["compute_peak"]
    x = rng.uniform(-1, 1, 64).astype(np.float32)

    def make():
        ck = compiler.compile(case.kernel, case.signature, dict(case.constants, ITERS=3), grid=(1,))
        return ck, (1,), dict(x_ptr=x.copy(), out_ptr=np.zeros(64, np.float32))
    got = _bitwise(make, monkeypatch)
    ref = 9999.0 + (x.astype(np.float64) - 9999.0) * 0.9999 ** (3 * 16)
    np.testing.assert_allclose(got["out_ptr"], ref, rtol=5e-3)


@pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")
@pytest.mark.parametrize("name", [c.name for c in CASES])
def test_pi_builds_assemble_with_vc4asm(pair, name):
    """vc4asm assembles every Pi-size .qasm to our words (moves on the mul
    ALU included) and its verifier (-V) reports nothing."""
    ck, _ = compile_case(CASES_BY_NAME[name], "pi")
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    assert words == [int(w) for w in ck.words]
    assert not warnings, warnings[:5]
