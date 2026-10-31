"""The packed input layout must preserve the ordinary matmul's numerical order."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton.compiler as pc
from kernels import matmul_kernel
from packed_matmul import matmul_packed_b_kernel, pack_b_columns
from piton.runtime.launcher import run_emulator
from piton.runtime.pi import Buffer


@pytest.mark.parametrize("stages", [1, 2])
def test_packed_matmul_ragged_bit_exact(stages):
    m, n, k, bm, bn, bk = 5, 32, 7, 4, 16, 4
    rng = np.random.default_rng(340)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
    packed = pack_b_columns(b, bn)
    np.testing.assert_array_equal(packed.transpose(1, 0, 2).reshape(k, n), b)
    got = np.full((m, n), np.nan, np.float32)
    expected = got.copy()
    sig = {name: "*fp32" for name in ("a_ptr", "b_ptr", "c_ptr")}
    sig.update({name: "i32" for name in ("M", "N", "stride_am", "stride_cm")})
    constants = dict(K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=4)
    grid = (4,)
    ck = pc.compile(matmul_packed_b_kernel, sig, constants, grid=grid, num_stages=stages)
    args = dict(a_ptr=a, b_ptr=packed, c_ptr=got, M=m, N=n, stride_am=k, stride_cm=n)
    run_emulator(ck, grid, args)
    sig["stride_bk"] = "i32"
    constants.update(stride_ak=1, stride_bn=1, stride_cn=1)
    baseline = pc.compile(matmul_kernel, sig, constants, grid=grid, num_stages=stages)
    args.update(b_ptr=b, c_ptr=expected, stride_bk=n)
    run_emulator(baseline, grid, args)
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_allclose(got, a.astype(np.float64) @ b.astype(np.float64), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kwargs", [
    dict(column_block=(12, 5)),
    dict(column_block=(0, 4)),
    dict(column_block=(12, 4), lcg_seed=None),
    dict(column_block=(12, 4), nbytes=83 * 4),
    dict(column_block=(12, 4), init=np.zeros(84, np.uint32)),
])
def test_reject_invalid_packed_generation(kwargs):
    args = dict(name="b", nbytes=84 * 4, lcg_seed=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        Buffer(**args)
