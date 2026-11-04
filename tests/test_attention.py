"""Fused attention on the strict emulator against a float64 NumPy reference."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import piton.compiler as pc
from attention import attention_kernel
from piton.runtime.launcher import run_emulator


def reference(q, k, v, scale, causal):
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * scale
    if causal:
        s = np.where(np.tril(np.ones(s.shape, bool)), s, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=1, keepdims=True)) @ v.astype(np.float64)


def run_attention(q, k, v, causal, bm=4, bn=16):
    seq, d = q.shape
    scale = float(d) ** -0.5
    grid = ((seq + bm - 1) // bm,)
    constants = dict(HEAD_DIM=d, BLOCK_M=bm, BLOCK_N=bn, CAUSAL=causal)
    signature = dict.fromkeys(("q_ptr", "k_ptr", "v_ptr", "o_ptr"), "*fp32")
    signature.update(seq="i32", sm_scale="fp32")
    compiled = pc.compile(attention_kernel, signature, constants, grid=grid)
    out = np.full(seq * d + 32, np.float32(-1234.5), dtype=np.float32)
    args = dict(q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=out, seq=seq, sm_scale=np.float32(scale))
    run_emulator(compiled, grid, args, strict=True)
    np.testing.assert_array_equal(out[seq * d:], np.float32(-1234.5))
    return out[:seq * d].reshape(seq, d), scale


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq,d,bm,bn", [(16, 16, 4, 16), (21, 16, 4, 16), (8, 32, 2, 16), (19, 32, 2, 16), (20, 64, 1, 16)])
def test_fused_attention(seq, d, bm, bn, causal):
    rng = np.random.default_rng(seq * 7 + d)
    q, k, v = (rng.normal(0, 1, (seq, d)).astype(np.float32) for _ in range(3))
    actual, scale = run_attention(q, k, v, causal, bm, bn)
    np.testing.assert_allclose(actual, reference(q, k, v, scale, causal), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_online_softmax_rescale(causal):
    """Large scores, with a later K tile dominating, force real rescaling."""
    rng = np.random.default_rng(5)
    q, k, v = (rng.normal(0, 1, (40, 16)).astype(np.float32) for _ in range(3))
    q *= 6
    k[35:] *= 5
    actual, scale = run_attention(q, k, v, causal)
    np.testing.assert_allclose(actual, reference(q, k, v, scale, causal), rtol=1e-3, atol=1e-4)
