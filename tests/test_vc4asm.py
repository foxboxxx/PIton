"""The emitted .qasm must assemble with vc4asm to exactly our encoding."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton.compiler as pc  # noqa: E402
from piton.qpu.asm import find_vc4asm, vc4asm_assemble  # noqa: E402
from kernels import add_kernel, matmul_kernel  # noqa: E402

pytestmark = pytest.mark.skipif(find_vc4asm() is None, reason="vc4asm not installed")


def check(ck):
    words, warnings = vc4asm_assemble(ck.asm["qasm"], verify=True)
    ours = ck.words
    assert len(words) == len(ours)
    bad = [(i, f"{a:016x}", f"{b:016x}") for i, (a, b) in enumerate(zip(words, ours)) if a != b]
    assert not bad, bad[:5]
    return warnings


def test_vector_add_qasm():
    ck = pc.compile(add_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32",
                                 "n_elements": "i32"}, {"BLOCK_SIZE": 256})
    check(ck)


def test_matmul_qasm():
    sig = {n: ("*fp32" if n.endswith("ptr") else "i32") for n in matmul_kernel.arg_names}
    cons = dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=4, GROUP_SIZE_M=4, K=64,
                stride_ak=1, stride_bn=1, stride_cn=1)
    for k in cons:
        sig.pop(k, None)
    check(pc.compile(matmul_kernel, sig, cons))


def test_matmul_runtime_k_qasm():
    sig = {n: ("*fp32" if n.endswith("ptr") else "i32") for n in matmul_kernel.arg_names}
    cons = dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=4, GROUP_SIZE_M=4,
                stride_ak=1, stride_bn=1, stride_cn=1)
    for k in cons:
        sig.pop(k, None)
    check(pc.compile(matmul_kernel, sig, cons))


def test_new_patterns_qasm():
    """Lane permutations, ALU exp/log/div/sqrt, IEEE compares, per-row DMA stores."""
    sys.path.insert(0, os.path.dirname(__file__))
    import test_tiles as T
    import test_hw as H
    f32 = {"x_ptr": "*fp32", "o_ptr": "*fp32"}
    check(pc.compile(T.reduce2d, f32, dict(R=16, L=4, AXIS=1, KIND=0)))
    check(pc.compile(T.trans_k, f32, dict(M=8, N=8)))
    check(pc.compile(T.dot_k, {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32"},
                     dict(M=4, N=8, K=8, ACC=True)))
    for F in range(4):
        check(pc.compile(T.explog_k, {**f32, "n": "i32"}, dict(F=F, BLOCK=128)))
    for F in range(3):
        check(pc.compile(T.divsqrt_k, {"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32"},
                         dict(F=F, BLOCK=128)))
    check(pc.compile(T.fcmp_k, {"x_ptr": "*fp32", "y_ptr": "*fp32", "o": "*i32"}, dict(C=0.0, MODE=0, BLOCK=16)))
    check(pc.compile(H._strided_store, {"x_ptr": "*fp32", "out_ptr": "*fp32", "base": "i32", "stride": "i32"},
                     dict(R=4, L=16)))
