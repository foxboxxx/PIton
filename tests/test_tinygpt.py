"""TinyGPT: trainer gradients, the new kernels, and QPU inference vs the NumPy reference.

All QPU runs are on the strict emulator. Inference is checked teacher-forced
(logits for a fixed token sequence) rather than by free-running generation:
the QPU rounds toward zero, so a near-tie argmax may legitimately differ and
then the continuations diverge.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from tinygpt import TinyGPT, sample
from tinygpt_model import (
    Config,
    forward,
    gelu,
    generate_reference,
    init_params,
    layernorm,
    load_model,
    loss_and_grads,
)

# Fixed text (not live README slices) so editing the docs cannot change the tests.
TEXT = ("Loads in loops whose addresses depend only on loop-carried pointers are\n"
        "**software pipelined**: every TMU receive refills the FIFO (`tl.dot`).")


@pytest.fixture(scope="module")
def trained():
    return load_model()


@pytest.fixture(scope="module")
def qpu_model(trained):
    cfg, _, params, _ = trained
    return TinyGPT(cfg, params)


def test_backprop_matches_finite_differences():
    cfg = Config(vocab=7, ctx=5, d_model=8, n_head=2, n_layer=2, d_mlp=12)
    rng = np.random.default_rng(0)
    params = {k: v + rng.normal(0, 0.1, v.shape) for k, v in init_params(cfg, seed=3).items()}
    ids = rng.integers(0, 7, (3, 5)).astype(np.int32)
    targets = rng.integers(0, 7, (3, 5)).astype(np.int32)
    _, grads = loss_and_grads(cfg, params, ids, targets)
    for name, value in params.items():
        # Every element of small tensors; a fixed sample of the larger ones.
        flat = value.reshape(-1)
        picks = range(flat.size) if flat.size <= 16 else rng.choice(flat.size, 16, replace=False)
        for i in picks:
            old = flat[i]
            flat[i] = old + 1e-6
            plus, _ = loss_and_grads(cfg, params, ids, targets)
            flat[i] = old - 1e-6
            minus, _ = loss_and_grads(cfg, params, ids, targets)
            flat[i] = old
            numeric = (plus - minus) / 2e-6
            # The key bias cancels in softmax, so its true gradient is 0.
            assert abs(numeric - grads[name].reshape(-1)[i]) < 1e-6 + 1e-5 * abs(numeric), (name, i)


def test_checkpoint_and_tokenizer(trained):
    cfg, tok, params, meta = trained
    assert cfg.vocab == len(tok.chars) and cfg.d_model == cfg.n_head * cfg.head_dim
    assert all(v.dtype == np.float32 for v in params.values())
    assert meta["val_loss"] < 2.0          # vs ln(vocab) ~ 4.5 for a uniform guess
    assert tok.decode(tok.encode(TEXT)) == TEXT    # every character is in the vocabulary


@pytest.mark.parametrize("t", [5, 32])
def test_embed_and_layernorm_kernels(qpu_model, trained, t):
    cfg, _, params, _ = trained
    ids = np.random.default_rng(t).integers(0, cfg.vocab, t).astype(np.int32)
    x = qpu_model.embed(ids)
    # The gather is exact; the add rounds toward zero (<= 1 ulp from NumPy).
    np.testing.assert_allclose(x, params["tok_emb"][ids] + params["pos_emb"][:t], rtol=2.5e-7, atol=0)
    y = qpu_model.layernorm(x, "h0.ln1")
    expected, _ = layernorm(x.astype(np.float64), params["h0.ln1_g"], params["h0.ln1_b"])
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("m", [3, 32])
@pytest.mark.parametrize("w,b,act,res", [
    ("h0.w1", "h0.b1", True, False),        # 32 -> 128, fused GELU
    ("h1.w2", "h1.b2", False, True),        # 128 -> 32, fused residual
    ("w_head", "b_head", False, False),     # 32 -> vocab (not a multiple of the tile)
])
def test_linear_kernel_epilogues(qpu_model, trained, m, w, b, act, res):
    _, _, params, _ = trained
    rng = np.random.default_rng(m)
    x = rng.normal(0, 1, (m, params[w].shape[0])).astype(np.float32)
    r = rng.normal(0, 1, (m, params[w].shape[1])).astype(np.float32) if res else None
    y = qpu_model.linear(x, w, b, gelu=act, residual=r)
    expected = x.astype(np.float64) @ params[w] + params[b]
    if act:
        expected = gelu(expected)
    if res:
        expected += r
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("t", [1, 9, 32])
def test_forward_matches_reference(qpu_model, trained, t):
    cfg, tok, params, _ = trained
    ids = tok.encode(TEXT[:t])
    assert len(ids) == t
    logits = qpu_model.forward(ids)
    np.testing.assert_allclose(logits, forward(cfg, params, ids), rtol=1e-3, atol=1e-3)
    last = qpu_model.forward(ids, last_only=True)
    np.testing.assert_allclose(last, logits[-1:], rtol=1e-5, atol=1e-5)


def test_short_greedy_generation_matches_reference(qpu_model, trained):
    cfg, tok, params, _ = trained
    prompt = tok.encode("The QPU ")
    ours = qpu_model.generate(prompt, 3)
    theirs = generate_reference(cfg, params, prompt, 3)
    for n in range(len(prompt), len(theirs)):
        top2 = np.sort(forward(cfg, params, theirs[:n])[-1])[-2:]
        if top2[1] - top2[0] < 1e-3:       # a near tie may round either way on the QPU
            break
        assert ours[n] == theirs[n]


def test_sampling():
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 5.0, 1.0])
    assert sample(logits, 0.0, rng) == 1
    draws = [sample(logits, 1.0, np.random.default_rng(s)) for s in range(200)]
    assert draws.count(1) > 150 and set(draws) <= {0, 1, 2}


def test_rejects_unsupported_inputs(trained):
    cfg, _, params, _ = trained
    with pytest.raises(ValueError, match="unsupported"):
        TinyGPT(Config(vocab=cfg.vocab, d_model=24, n_head=2), params)
    model = TinyGPT(cfg, params)
    with pytest.raises(ValueError, match="empty prompt"):
        model.generate([], 1)
    with pytest.raises(AssertionError):
        model.forward(np.zeros(cfg.ctx + 1, np.int32))
