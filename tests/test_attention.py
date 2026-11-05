"""Flash-attention example (examples/attention.py) on the strict emulator, and
the Pi harness (examples/attention_pi.py) without a Pi."""
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from attention import (attention_reference, attn_fwd_kernel, compile_attention,  # noqa: E402
                       pattern_qk, run_attention)
import attention_pi  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402

# Round-toward-zero fp32 against float64: the emulator measures <= 5e-6 at the
# Pi sizes (N 512) for outputs of magnitude <= 1; small sizes stay far below.
ATOL = 2e-5
# PIton's x / y is within 2 ulp (x / 1.0 lands 2 ulp toward zero), so outputs
# that are exact up to the final normalisation are compared to that.
DIV_ULP = 2


def qkv(n, d, seed=0, gain=1.0, heads=None):
    rng = np.random.default_rng(seed)
    shape = (n, d) if heads is None else (heads, n, d)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    return (q * np.float32(gain)).astype(np.float32), k, v


@pytest.mark.parametrize("n,d,block_m,block_n,causal", [
    (64, 16, 2, 16, False),
    (64, 16, 2, 16, True),
    (64, 32, 2, 32, False),
    (64, 32, 2, 32, True),
    (64, 32, 4, 16, False),
    (32, 64, 2, 16, True),
])
def test_matches_float64(n, d, block_m, block_n, causal):
    q, k, v = qkv(n, d, seed=n + d + block_m)
    out, _, _ = run_attention(q, k, v, causal, block_m, block_n)
    ref = attention_reference(q, k, v, causal)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=0, atol=ATOL)


def run_padded(q, k, v, causal, block_m, block_n, extra_rows=3):
    """Run with an output buffer extra_rows longer than N_CTX, filled with a
    sentinel, so stores past the last query row would show."""
    n, d = q.shape
    ck = compile_attention(d, block_m, block_n, causal, even_n=n % block_n == 0, max_n_ctx=n)
    sentinel = np.float32(-1234.5)
    out = np.full((n + extra_rows, d), sentinel, np.float32)
    values = dict(Q=q, K=k, V=v, Out=out, sm_scale=float(1 / np.sqrt(d)), N_CTX=n,
                  stride_qh=0, stride_kh=0, stride_vh=0, stride_oh=0)
    run_emulator(ck, (-(-n // block_m), 1), {k_: v_ for k_, v_ in values.items() if k_ in ck.signature})
    np.testing.assert_array_equal(out[n:], sentinel)
    return out[:n]


@pytest.mark.parametrize("n,causal", [(37, False), (37, True), (1, False), (17, True)])
def test_ragged_sequence_lengths(n, causal):
    # N_CTX divides neither BLOCK_M nor BLOCK_N: masked query rows (no store
    # past the end), masked keys in the tail block, and N_CTX = 1.
    q, k, v = qkv(n, 16, seed=n)
    out = run_padded(q, k, v, causal, 2, 16)
    np.testing.assert_allclose(out, attention_reference(q, k, v, causal), rtol=0, atol=ATOL)
    if n == 1:      # one key: weight exp2(0) = 1, then out = v / 1
        np.testing.assert_array_max_ulp(out, v, maxulp=DIV_ULP)


def test_multi_head_runtime_strides_through_the_launcher():
    # kernel[grid](...) as in Triton: padded rows (stride_qm = 24 for HEAD_DIM 16),
    # two heads on grid axis 1, every stride a runtime argument (stride_kk = 1
    # is specialised, as Triton does for integers equal to 1).
    heads, n, d, pad = 2, 48, 16, 8
    rng = np.random.default_rng(5)
    q_store = rng.standard_normal((heads, n, d + pad)).astype(np.float32)
    k, v = (rng.standard_normal((heads, n, d)).astype(np.float32) for _ in range(2))
    out = np.zeros((heads, n, d), np.float32)
    grid = (n // 2, heads)
    attn_fwd_kernel[grid](q_store, k, v, out, float(1 / np.sqrt(d)),
                          n * (d + pad), d + pad, n * d, d, 1, n * d, d, n * d, d, n,
                          HEAD_DIM=d, BLOCK_M=2, BLOCK_N=16, CAUSAL=True, EVEN_N=True)
    ref = attention_reference(q_store[:, :, :d], k, v, causal=True)
    np.testing.assert_allclose(out, ref, rtol=0, atol=ATOL)


def test_transposed_k_layout_is_bit_identical():
    # K stored [HEAD_DIM, N_CTX] loads the same values in the same order
    q, k, v = qkv(32, 32, seed=11)
    nd, _, _ = run_attention(q, k, v, False, 2, 16, k_layout="nd")
    dn, _, _ = run_attention(q, k, v, False, 2, 16, k_layout="dn")
    np.testing.assert_array_equal(nd.view(np.uint32), dn.view(np.uint32))


def test_hard_attention_retrieves_values_exactly():
    # Each query matches one key by 1000 / sqrt(d): every other weight is
    # exp2(-255), which flushes to 0, and the match gets exp2(0) = 1, so the
    # accumulator holds the matched value row bit for bit -- also when the match
    # comes in the second key block and the first block's sum is rescaled by 0 --
    # and only the final division by l = 1 rounds.
    n = d = 32
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    k = np.eye(n, dtype=np.float32)
    q = (1000.0 * np.eye(n, dtype=np.float32)[perm]).astype(np.float32)
    v = rng.standard_normal((n, d)).astype(np.float32)
    out, _, _ = run_attention(q, k, v, False, 2, 16)
    np.testing.assert_array_max_ulp(out, v[perm], maxulp=DIV_ULP)


def test_causal_rows_ignore_future_keys():
    # Changing keys and values after position 31 leaves rows 0-31 bit-identical;
    # row 0 sees only key 0 and equals V[0] up to the division by 1.
    n, d = 48, 16
    q, k, v = qkv(n, d, seed=8)
    a, ck, _ = run_attention(q, k, v, True, 2, 16)
    k2, v2 = k.copy(), v.copy()
    rng = np.random.default_rng(9)
    k2[32:] = rng.standard_normal((n - 32, d)) * 10
    v2[32:] = rng.standard_normal((n - 32, d)) * 10
    b, _, _ = run_attention(q, k2, v2, True, 2, 16, compiled=ck)
    np.testing.assert_array_equal(a[:32].view(np.uint32), b[:32].view(np.uint32))
    np.testing.assert_array_max_ulp(a[0], v[0], maxulp=DIV_ULP)
    assert not np.array_equal(a[32:], b[32:])


def test_large_scores_stay_finite():
    # Scores of +-100: without the running maximum exp would overflow
    q, k, v = qkv(48, 16, seed=4, gain=25.0)
    out, _, _ = run_attention(q, k, v, False, 2, 16)
    ref = attention_reference(q, k, v)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=0, atol=ATOL)


def test_identity_values_give_the_attention_matrix():
    # The Pi's picture: with V = I the output is softmax(Q K^T / sqrt(d))
    q, k = pattern_qk(32, 32)
    eye = np.eye(32, dtype=np.float32)
    p, _, _ = run_attention(q, k, eye, True, 2, 16)
    ref = attention_reference(q, k, eye, True)
    np.testing.assert_allclose(p, ref, rtol=0, atol=ATOL)
    np.testing.assert_array_equal(p[np.triu_indices(32, 1)], 0)      # causal: exactly 0
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)


# -- the Pi build -------------------------------------------------------------------

KERNELS, PROBLEMS, CASES = attention_pi.plan()


@pytest.mark.parametrize("spec", KERNELS, ids=[k["name"] for k in KERNELS])
def test_pi_kernels_compile(spec):
    # every kernel the Pi firmware runs, at its size (grid for N_CTX 512)
    ck = compile_attention(spec["head_dim"], spec["block_m"], spec["block_n"], spec["causal"],
                           spec["even_n"], spec["k_layout"], n_ctx=attention_pi.MAX_N,
                           max_n_ctx=attention_pi.MAX_N)
    assert 0 < len(ck.words) < 4096
    assert set(ck.signature) == {"Q", "K", "V", "Out", "sm_scale", "N_CTX",
                                 "stride_qh", "stride_kh", "stride_vh", "stride_oh"}


@pytest.fixture(scope="module")
def synthetic():
    refs = attention_pi.references(PROBLEMS)
    log = attention_pi.synthetic_log(KERNELS, PROBLEMS, CASES, 5, 3, refs)
    return log, refs


def parse(log, refs):
    return attention_pi.parse_log(log, KERNELS, PROBLEMS, CASES, 5, 3, refs)


def test_pi_plan_covers_the_sizes():
    sizes = {(p["n_ctx"], p["head_dim"], p["causal"]) for p in PROBLEMS}
    for n in attention_pi.SIZES:
        for d in (32, 64):
            assert (n, d, False) in sizes and (n, d, True) in sizes
    assert any(p["n_ctx"] % 32 for p in PROBLEMS)                  # a ragged length
    assert any(KERNELS[k]["k_layout"] == "dn" for k, _ in CASES)   # pre-transposed K


def test_pi_inputs_mirror_the_board_lcg():
    # piton_rt.h's LCG, as main.c draws it: Q = 4 * lcg(1000 + d), K, V
    q, k, v = attention_pi.inputs_for(dict(n_ctx=4, head_dim=32, causal=False, inputs=attention_pi.LCG))
    s = 1032
    first = []
    for _ in range(2):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        first.append(np.float32(s >> 8) * np.float32(1 / 8388608) - np.float32(1))
    np.testing.assert_array_equal(q.ravel()[:2], np.float32(4) * np.array(first, np.float32))
    assert q.dtype == k.dtype == v.dtype == np.float32 and q.shape == (4, 32)


def test_pi_log_parses(synthetic):
    log, refs = synthetic
    result = parse(log, refs)
    assert len(result["cases"]) == len(CASES)
    best = attention_pi.best_rows(result["cases"])
    assert len(best) == 12
    assert all(r["gflops"] > 0 and r["speedup_vs_arm"] > 0 for r in result["cases"])
    pattern = attention_pi.pattern_panels(result, KERNELS, PROBLEMS, CASES)
    assert [p.shape for p in pattern] == [(64, 64), (64, 64)]


def _replace_first(log, pattern, new):
    out, count = re.subn(pattern, new, log, count=1, flags=re.MULTILINE)
    assert count == 1
    return out


@pytest.mark.parametrize("pattern,new", [
    (r"^ATTN_TIME case=3 us=\d+\n", ""),                                  # a missing timing
    (r"failures=0 nonfinite=0", "failures=2 nonfinite=0"),                 # a board check failed
    (r"^ATTN_CHECK_OK$", "ATTN_CHECK_FAILED"),
    (r"^PITON_DONE\n", ""),                                                # incomplete run
    (r"^ATTN_ARM problem=2 us=\d+\n", ""),
    (r"differing=0", "differing=1"),
    (r"^(ATTN_SAMPLE case=5 stride=\d+ count=\d+\n)[0-9a-f]+", r"\g<1>3f800000"),   # a wrong output
])
def test_pi_log_rejects_bad_runs(synthetic, pattern, new):
    log, refs = synthetic
    with pytest.raises(ValueError):
        parse(_replace_first(log, pattern, new), refs)
