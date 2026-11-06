"""Fused attention on the strict emulator against a float64 NumPy reference."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from attention import attention_kernel

import piton.compiler as pc
from piton.runtime.launcher import run_emulator

GUARD = np.float32(-1234.5)


def reference(q, k, v, scale, causal):
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * scale
    if causal:
        s = np.where(np.tril(np.ones(s.shape, bool)), s, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=1, keepdims=True)) @ v.astype(np.float64)


def run_attention(q, k, v, causal, bm=4, bn=16, heads=1, scale=None):
    """q/k/v are [seq, heads * d]; head h reads columns h*d .. (h+1)*d."""
    seq, width = q.shape
    d = width // heads
    scale = float(d) ** -0.5 if scale is None else scale
    grid = ((seq + bm - 1) // bm, heads)
    constants = {"HEAD_DIM": d, "BLOCK_M": bm, "BLOCK_N": bn, "CAUSAL": causal, "STRIDE": width}
    signature = dict.fromkeys(("q_ptr", "k_ptr", "v_ptr", "o_ptr"), "*fp32")
    signature.update(seq="i32", sm_scale="fp32")
    compiled = pc.compile(attention_kernel, signature, constants, grid=grid)
    out = np.full(seq * width + 32, GUARD, dtype=np.float32)
    args = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out,
            "seq": seq, "sm_scale": np.float32(scale)}
    run_emulator(compiled, grid, args, strict=True)
    np.testing.assert_array_equal(out[seq * width:], GUARD)
    return out[:seq * width].reshape(seq, width), scale


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


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads,seq,bm", [(2, 21, 4), (4, 9, 2)])
def test_multi_head_strided(heads, seq, bm, causal):
    """Heads interleaved in one [seq, heads*16] buffer, as a projection writes them.

    STRIDE=64 with BLOCK_M=4 exceeds the register file, so 4 heads use BLOCK_M=2.
    """
    rng = np.random.default_rng(heads * 31 + seq)
    q, k, v = (rng.normal(0, 1, (seq, heads * 16)).astype(np.float32) for _ in range(3))
    actual, scale = run_attention(q, k, v, causal, bm=bm, heads=heads)
    for h in range(heads):
        cols = slice(h * 16, (h + 1) * 16)
        np.testing.assert_allclose(actual[:, cols], reference(q[:, cols], k[:, cols], v[:, cols], scale, causal),
                                   rtol=1e-3, atol=1e-4)


def test_sm_scale_is_honoured():
    rng = np.random.default_rng(11)
    q, k, v = (rng.normal(0, 1, (12, 16)).astype(np.float32) for _ in range(3))
    actual, scale = run_attention(q, k, v, False, scale=0.6)
    np.testing.assert_allclose(actual, reference(q, k, v, scale, False), rtol=1e-3, atol=1e-4)
