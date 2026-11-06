"""TinyGPT: a character-level transformer small enough for the Pi Zero's QPUs.

This module is the host side of the model: the corpus, a NumPy reference
forward pass, and a NumPy trainer with hand-written backpropagation and Adam.
The QPU inference lives in tinygpt.py. Both share these definitions, so the
formulas below are exactly what the kernels compute:

* pre-LayerNorm blocks, LayerNorm(x) = (x - mean) * rsqrt(var + EPS) * g + b
* causal multi-head attention, separate Q/K/V/O projections with biases
* MLP with GELU approximated as u * sigmoid(1.702 u) (the QPU has sigmoid
  but no tanh/erf)
* untied output head

The training corpus is the repository's own README.md and STATUS.md, so the
model learns to babble about QPUs, registers and matmuls. Train once with

    .venv/bin/python examples/tinygpt_model.py --train

which writes examples/data/tinygpt/model.npz. Tests only load that file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "examples" / "data" / "tinygpt"
MODEL_PATH = DATA_DIR / "model.npz"
CORPUS_FILES = ("README.md", "STATUS.md")
# The larger sizes also read the compiler's own source, so they see ~15x more
# text than the README alone and learn to write PIton-flavoured Python.
CODE_CORPUS_GLOBS = ("README.md", "STATUS.md", "perf/README.md", "piton/**/*.py", "examples/*.py")
EPS = 1e-5
GELU_K = 1.702


@dataclass(frozen=True)
class Config:
    vocab: int
    ctx: int = 32
    d_model: int = 32
    n_head: int = 2
    n_layer: int = 2
    d_mlp: int = 128

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# Model sizes. "tiny" is the original checkpoint (model.npz, README + STATUS);
# the others are trained on CODE_CORPUS_GLOBS. Every shape here compiles for
# the QPU kernels (examples/tinygpt_pi.py picks the tiles per size).
SIZES = {
    "tiny": {"ctx": 32, "d_model": 32, "n_head": 2, "n_layer": 2, "d_mlp": 128},
    "small": {"ctx": 64, "d_model": 64, "n_head": 4, "n_layer": 4, "d_mlp": 256},
    "medium": {"ctx": 128, "d_model": 128, "n_head": 4, "n_layer": 4, "d_mlp": 512},
}
TRAIN_STEPS = {"tiny": 4000, "small": 6000, "medium": 6000}


def model_path(size: str = "tiny") -> Path:
    if size not in SIZES:
        raise ValueError(f"unknown TinyGPT size {size!r}; choose from {', '.join(SIZES)}")
    return MODEL_PATH if size == "tiny" else DATA_DIR / f"model_{size}.npz"


# ----------------------------------------------------------------------------
# Corpus and tokenizer

def corpus_files(size: str = "tiny", root: Path = ROOT) -> list[str]:
    if size == "tiny":
        return list(CORPUS_FILES)
    names = set()
    for pattern in CODE_CORPUS_GLOBS:
        names.update(str(path.relative_to(root)) for path in root.glob(pattern) if path.is_file())
    return sorted(names)


def load_corpus(root: Path = ROOT, size: str = "tiny") -> str:
    """The size's corpus files, reduced to printable ASCII and newlines."""
    text = "\n\n".join((root / name).read_text(encoding="utf-8") for name in corpus_files(size, root))
    return "".join(c for c in text if c == "\n" or " " <= c <= "~")


class Tokenizer:
    def __init__(self, chars: str):
        self.chars = chars
        self.index = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> Tokenizer:
        return cls("".join(sorted(set(text))))

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.index[c] for c in text if c in self.index], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


# ----------------------------------------------------------------------------
# Parameters

def layer_names(layer: int) -> list[str]:
    p = f"h{layer}."
    return [p + n for n in ("ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
                            "ln2_g", "ln2_b", "w1", "b1", "w2", "b2")]


def init_params(cfg: Config, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d, f = cfg.d_model, cfg.d_mlp

    def normal(shape, std):
        return rng.normal(0.0, std, shape)

    resid_std = 1.0 / np.sqrt(d) / np.sqrt(2 * cfg.n_layer)
    params = {"tok_emb": normal((cfg.vocab, d), 0.1), "pos_emb": normal((cfg.ctx, d), 0.1)}
    for layer in range(cfg.n_layer):
        p = f"h{layer}."
        params.update({
            p + "ln1_g": np.ones(d), p + "ln1_b": np.zeros(d),
            p + "wq": normal((d, d), d ** -0.5), p + "bq": np.zeros(d),
            p + "wk": normal((d, d), d ** -0.5), p + "bk": np.zeros(d),
            p + "wv": normal((d, d), d ** -0.5), p + "bv": np.zeros(d),
            p + "wo": normal((d, d), resid_std), p + "bo": np.zeros(d),
            p + "ln2_g": np.ones(d), p + "ln2_b": np.zeros(d),
            p + "w1": normal((d, f), d ** -0.5), p + "b1": np.zeros(f),
            p + "w2": normal((f, d), f ** -0.5 / np.sqrt(2 * cfg.n_layer)), p + "b2": np.zeros(d),
        })
    params.update({"lnf_g": np.ones(d), "lnf_b": np.zeros(d),
                   "w_head": normal((d, cfg.vocab), d ** -0.5), "b_head": np.zeros(cfg.vocab)})
    return params


def save_model(path: Path, cfg: Config, tok: Tokenizer, params: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.astype(np.float32) for k, v in params.items()}
    header = json.dumps({"config": asdict(cfg), "chars": tok.chars, **meta})
    np.savez_compressed(path, __meta__=np.array(header), **arrays)


def load_model(path: Path | str = MODEL_PATH) -> tuple[Config, Tokenizer, dict[str, np.ndarray], dict]:
    if isinstance(path, str) and path in SIZES:
        path = model_path(path)
    with np.load(path) as z:
        meta = json.loads(str(z["__meta__"]))
        params = {k: z[k] for k in z.files if k != "__meta__"}
    cfg = Config(**meta.pop("config"))
    return cfg, Tokenizer(meta.pop("chars")), params, meta


# ----------------------------------------------------------------------------
# Forward pass (shared by the reference and the trainer)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gelu(u):
    return u * sigmoid(GELU_K * u)


def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + EPS)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd)


def forward(cfg: Config, params: dict, ids: np.ndarray, *, cache: bool = False, dtype=np.float64):
    """ids [B, T] (or [T]) -> logits [B, T, vocab]; optionally the backprop cache."""
    squeeze = ids.ndim == 1
    ids = np.atleast_2d(ids)
    bsz, t = ids.shape
    assert t <= cfg.ctx
    P = {k: v.astype(dtype, copy=False) for k, v in params.items()}
    h, hd = cfg.n_head, cfg.head_dim
    scale = hd ** -0.5
    mask = np.tril(np.ones((t, t), bool))
    x = P["tok_emb"][ids] + P["pos_emb"][:t]
    tape = []
    for layer in range(cfg.n_layer):
        p = f"h{layer}."
        x_in = x
        a, ln1 = layernorm(x, P[p + "ln1_g"], P[p + "ln1_b"])
        q, k, v = (a @ P[p + w] + P[p + b] for w, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
        qh, kh, vh = (z.reshape(bsz, t, h, hd).transpose(0, 2, 1, 3) for z in (q, k, v))
        s = (qh @ kh.transpose(0, 1, 3, 2)) * scale
        s = np.where(mask, s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        att = np.exp(s)
        att /= att.sum(-1, keepdims=True)
        yh = att @ vh
        y = yh.transpose(0, 2, 1, 3).reshape(bsz, t, cfg.d_model)
        x = x + y @ P[p + "wo"] + P[p + "bo"]
        x_mid = x
        a2, ln2 = layernorm(x, P[p + "ln2_g"], P[p + "ln2_b"])
        u = a2 @ P[p + "w1"] + P[p + "b1"]
        gu = gelu(u)
        x = x + gu @ P[p + "w2"] + P[p + "b2"]
        if cache:
            tape.append({"x_in": x_in, "a": a, "ln1": ln1, "qh": qh, "kh": kh, "vh": vh, "att": att,
                         "y": y, "x_mid": x_mid, "a2": a2, "ln2": ln2, "u": u, "gu": gu})
    af, lnf = layernorm(x, P["lnf_g"], P["lnf_b"])
    logits = af @ P["w_head"] + P["b_head"]
    if cache:
        return logits, {"ids": ids, "tape": tape, "af": af, "lnf": lnf, "P": P}
    return logits[0] if squeeze else logits


# ----------------------------------------------------------------------------
# Training

def _layernorm_back(dy, g, saved):
    xhat, rstd = saved
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, (dy * xhat).reshape(-1, g.size).sum(0), dy.reshape(-1, g.size).sum(0)


def _linear_back(dy, x, w, grads, wname, bname):
    grads[wname] = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    grads[bname] = dy.reshape(-1, dy.shape[-1]).sum(0)
    return dy @ w.T


def loss_and_grads(cfg: Config, params: dict, ids: np.ndarray, targets: np.ndarray, dtype=np.float64):
    """Mean cross-entropy over all positions, and its gradient for every parameter."""
    logits, c = forward(cfg, params, ids, cache=True, dtype=dtype)
    P, tape = c["P"], c["tape"]
    bsz, t = ids.shape
    h, hd = cfg.n_head, cfg.head_dim
    scale = hd ** -0.5
    z = logits - logits.max(-1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(-1, keepdims=True)
    n = bsz * t
    flat = prob.reshape(n, -1)
    loss = -np.log(flat[np.arange(n), targets.ravel()] + 1e-30).mean()
    dlogits = prob.copy().reshape(n, -1)
    dlogits[np.arange(n), targets.ravel()] -= 1.0
    dlogits = (dlogits / n).reshape(logits.shape)

    g = {}
    daf = _linear_back(dlogits, c["af"], P["w_head"], g, "w_head", "b_head")
    dx, g["lnf_g"], g["lnf_b"] = _layernorm_back(daf, P["lnf_g"], c["lnf"])
    for layer in reversed(range(cfg.n_layer)):
        p, s = f"h{layer}.", tape[layer]
        dgu = _linear_back(dx, s["gu"], P[p + "w2"], g, p + "w2", p + "b2")
        sg = sigmoid(GELU_K * s["u"])
        du = dgu * (sg + GELU_K * s["u"] * sg * (1.0 - sg))
        da2 = _linear_back(du, s["a2"], P[p + "w1"], g, p + "w1", p + "b1")
        dln, g[p + "ln2_g"], g[p + "ln2_b"] = _layernorm_back(da2, P[p + "ln2_g"], s["ln2"])
        dx = dx + dln
        dy = _linear_back(dx, s["y"], P[p + "wo"], g, p + "wo", p + "bo")
        dyh = dy.reshape(bsz, t, h, hd).transpose(0, 2, 1, 3)
        att, qh, kh, vh = s["att"], s["qh"], s["kh"], s["vh"]
        datt = dyh @ vh.transpose(0, 1, 3, 2)
        dvh = att.transpose(0, 1, 3, 2) @ dyh
        ds = att * (datt - (datt * att).sum(-1, keepdims=True)) * scale
        dqh = ds @ kh
        dkh = ds.transpose(0, 1, 3, 2) @ qh
        da = 0.0
        for dz, w, b in ((dqh, "wq", "bq"), (dkh, "wk", "bk"), (dvh, "wv", "bv")):
            dz = dz.transpose(0, 2, 1, 3).reshape(bsz, t, cfg.d_model)
            da = da + _linear_back(dz, s["a"], P[p + w], g, p + w, p + b)
        dln, g[p + "ln1_g"], g[p + "ln1_b"] = _layernorm_back(da, P[p + "ln1_g"], s["ln1"])
        dx = dx + dln
    g["pos_emb"] = np.zeros_like(P["pos_emb"])
    g["pos_emb"][:t] = dx.sum(0)
    g["tok_emb"] = np.zeros_like(P["tok_emb"])
    np.add.at(g["tok_emb"], c["ids"].ravel(), dx.reshape(n, -1))
    return loss, g


def train(steps: int = 4000, batch: int = 32, lr: float = 3e-3, seed: int = 0, log_every: int = 250,
          size: str = "tiny"):
    """Adam with cosine decay. Larger sizes keep the checkpoint with the best
    validation loss (their corpus is small enough to overfit)."""
    text = load_corpus(size=size)
    tok = Tokenizer.from_text(text)
    data = tok.encode(text)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    cfg = Config(vocab=len(tok.chars), **SIZES[size])
    params = {k: v.astype(np.float32) for k, v in init_params(cfg, seed).items()}
    best = (np.inf, params, 0)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(seed + 1)
    b1, b2 = 0.9, 0.99

    def batch_of(src, count):
        starts = rng.integers(0, len(src) - cfg.ctx - 1, count)
        x = np.stack([src[s:s + cfg.ctx] for s in starts])
        y = np.stack([src[s + 1:s + cfg.ctx + 1] for s in starts])
        return x, y

    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = batch_of(train_data, batch)
        loss, grads = loss_and_grads(cfg, params, x, y, dtype=np.float32)
        rate = lr * 0.5 * (1 + np.cos(np.pi * step / steps)) * min(1.0, step / 100)
        for k, gk in grads.items():
            m[k] = b1 * m[k] + (1 - b1) * gk
            v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
            update = (m[k] / (1 - b1 ** step)) / (np.sqrt(v2[k] / (1 - b2 ** step)) + 1e-8)
            decay = 0.01 * params[k] if params[k].ndim == 2 else 0.0
            params[k] = (params[k] - rate * (update + decay)).astype(np.float32)
        if step % log_every == 0 or step == steps:
            xv, yv = batch_of(val_data, 64)
            val, _ = loss_and_grads(cfg, params, xv, yv, dtype=np.float32)
            print(f"step {step:5d}  train {loss:.3f}  val {val:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if size != "tiny" and val < best[0]:
                best = (val, dict(params), step)
    if size != "tiny":
        params = best[1]
    xv, yv = batch_of(val_data, 256)
    val, _ = loss_and_grads(cfg, params, xv, yv, dtype=np.float32)
    meta = {"steps": steps, "batch": batch, "lr": lr, "seed": seed, "val_loss": round(float(val), 4),
            "corpus": list(CORPUS_FILES) if size == "tiny" else list(CODE_CORPUS_GLOBS),
            "corpus_chars": len(text), "size": size, "train_seconds": round(time.time() - t0)}
    if size != "tiny":
        meta["kept_step"] = best[2]
    return cfg, tok, params, meta


def generate_reference(cfg: Config, params: dict, prompt_ids: np.ndarray, n_new: int) -> np.ndarray:
    """Greedy decoding with the float64 reference (sliding window of cfg.ctx tokens)."""
    ids = list(prompt_ids)
    for _ in range(n_new):
        logits = forward(cfg, params, np.array(ids[-cfg.ctx:], dtype=np.int32))
        ids.append(int(np.argmax(logits[-1])))
    return np.array(ids, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="train and overwrite the saved model")
    ap.add_argument("--size", default="tiny", choices=tuple(SIZES))
    ap.add_argument("--steps", type=int, default=None, help="default: TRAIN_STEPS[size]")
    ap.add_argument("--prompt", default="The QPU ")
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()
    if args.train:
        cfg, tok, params, meta = train(steps=args.steps or TRAIN_STEPS[args.size], size=args.size)
        save_model(model_path(args.size), cfg, tok, params, meta)
        n_params = sum(v.size for v in params.values())
        print(f"saved {model_path(args.size)} ({n_params} parameters, val loss {meta['val_loss']})")
    cfg, tok, params, _ = load_model(model_path(args.size))
    out = generate_reference(cfg, params, tok.encode(args.prompt), args.tokens)
    print(tok.decode(out))


if __name__ == "__main__":
    main()
