"""TinyGPT inference on the QPUs: every layer of the transformer is a PIton kernel.

    .venv/bin/python examples/tinygpt.py --prompt "The QPU " --tokens 40

One forward pass is 1 + 9 * n_layer + 2 launches:

    embed            token gather + position embedding           embed_kernel
    per block:  LN1 -> Q, K, V projections                       layernorm_kernel, linear_kernel x3
                causal multi-head flash attention                attention_kernel (attention.py)
                output projection + residual                     linear_kernel (RESIDUAL)
                LN2 -> FC1 + GELU -> FC2 + residual              layernorm_kernel, linear_kernel x2
    final LN -> vocabulary head                                  layernorm_kernel, linear_kernel

The host only picks the next token (greedy argmax) and slides the context
window; activations are produced and consumed by kernels. While generating,
the final LayerNorm and head run on the last position only.

Runs on the emulator by default. `device="pi"` uses the same launches through
piton.runtime.pi, which is unverified here (no Pi attached). Cycle counts are
the emulator's rough timing model (1 QPU cycle = 4 clocks at 250 MHz), not
measurements.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention import attention_kernel
from tinygpt_model import EPS, GELU_K, Config, load_model

import piton
import piton.language as tl
from piton import compiler
from piton.runtime.launcher import run_emulator

QPU_HZ = 250e6
CLOCKS_PER_CYCLE = 4


@piton.jit
def embed_kernel(ids_ptr, tok_ptr, pos_ptr, x_ptr, T,
                 D: tl.constexpr, BLOCK_M: tl.constexpr):
    """x[t] = tok_emb[ids[t]] + pos_emb[t] (a TMU gather by token id)."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    live = rows < T
    ids = tl.load(ids_ptr + rows, mask=live, other=0)
    tok = tl.load(tok_ptr + ids[:, None] * D + d[None, :])
    pos = tl.load(pos_ptr + rows[:, None] * D + d[None, :], mask=live[:, None], other=0.0)
    tl.store(x_ptr + rows[:, None] * D + d[None, :], tok + pos, mask=live[:, None])


@piton.jit
def layernorm_kernel(x_ptr, g_ptr, b_ptr, y_ptr, M, eps,
                     D: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    live = rows[:, None] < M
    x = tl.load(x_ptr + rows[:, None] * D + d[None, :], mask=live, other=0.0)
    mean = tl.sum(x, axis=1) / D
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=1) / D + eps)
    g = tl.load(g_ptr + d)
    b = tl.load(b_ptr + d)
    y = xc * rstd[:, None] * g[None, :] + b[None, :]
    tl.store(y_ptr + rows[:, None] * D + d[None, :], y, mask=live)


@piton.jit
def linear_kernel(x_ptr, w_ptr, bias_ptr, res_ptr, y_ptr, M,
                  N: tl.constexpr, K: tl.constexpr,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr):
    """y = [res +] act(x @ w + bias); x [M, K], w [K, N], all row-major FP32."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Columns wrap onto real weights (as in dense_layer_kernel); only in-bounds
    # elements are stored.
    cols = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    ks = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
    w_ptrs = w_ptr + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(K // BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * N
    acc += tl.load(bias_ptr + cols)[None, :]
    if GELU:
        acc = acc * tl.sigmoid(acc * 1.702)
    out_rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = out_rows[:, None] * N + out_cols[None, :]
    mask = (out_rows[:, None] < M) & (out_cols[None, :] < N)
    if RESIDUAL:
        acc += tl.load(res_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, acc, mask=mask)


assert GELU_K == 1.702, "linear_kernel hard-codes the GELU constant"

# (BLOCK_M, BLOCK_N, BLOCK_K) per projection shape; chosen to fit the
# register file with the fused epilogues (see tests/test_tinygpt.py).
LINEAR_TILES = {"narrow": (8, 16, 8), "wide": (4, 64, 8)}
LN_BLOCK_M = 4
EMBED_BLOCK_M = 4
ATTN_BLOCK_M = 4


def linear_tiles(n: int) -> tuple[int, int, int]:
    return LINEAR_TILES["wide" if n >= 64 else "narrow"]


class TinyGPT:
    """QPU inference for a TinyGPT checkpoint.

    `cycles` sums the emulator's per-launch cycle estimates; on the Pi,
    `pi_us` sums the measured per-launch times that run_on_pi reports.
    """

    def __init__(self, cfg: Config, params: dict, device: str = "emu"):
        def pow2(n):
            return n > 0 and n & (n - 1) == 0

        # tl.arange needs powers of two; linear_kernel needs K % BLOCK_K == 0.
        if not (pow2(cfg.d_model) and pow2(cfg.head_dim) and cfg.d_model % 8 == 0 and cfg.d_mlp % 8 == 0):
            raise ValueError(f"unsupported TinyGPT shape for the QPU kernels: {cfg}")
        self.cfg, self.device = cfg, device
        self.p = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in params.items()}
        self.launches = 0
        self.cycles = 0
        self.pi_us = 0.0
        self._attn = None

    # -- single launches ---------------------------------------------------
    def _record(self, ck):
        self.launches += 1
        stats = getattr(ck, "last_stats", None) or {}
        self.cycles += int(stats.get("cycles", 0))
        self.pi_us += float(sum(stats.get("times_us", ())))

    def embed(self, ids: np.ndarray) -> np.ndarray:
        t, d = len(ids), self.cfg.d_model
        x = np.zeros((t, d), np.float32)
        grid = (triton_cdiv(t, EMBED_BLOCK_M),)
        ck = embed_kernel[grid](np.ascontiguousarray(ids, np.int32), self.p["tok_emb"], self.p["pos_emb"],
                                x, t, D=d, BLOCK_M=EMBED_BLOCK_M, device=self.device)
        self._record(ck)
        return x

    def layernorm(self, x: np.ndarray, name: str) -> np.ndarray:
        m, d = x.shape
        y = np.zeros_like(x)
        ck = layernorm_kernel[(triton_cdiv(m, LN_BLOCK_M),)](
            x, self.p[name + "_g"], self.p[name + "_b"], y, m, EPS,
            D=d, BLOCK_M=LN_BLOCK_M, device=self.device)
        self._record(ck)
        return y

    def linear(self, x: np.ndarray, w: str, b: str, *, gelu=False, residual=None) -> np.ndarray:
        m, k = x.shape
        n = self.p[w].shape[1]
        bm, bn, bk = linear_tiles(n)
        if k % bk:
            raise ValueError(f"linear_kernel needs K % {bk} == 0, got K={k}")
        y = np.zeros((m, n), np.float32)
        res = residual if residual is not None else y
        grid = (triton_cdiv(m, bm) * triton_cdiv(n, bn),)
        ck = linear_kernel[grid](x, self.p[w], self.p[b], res, y, m,
                                 N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                                 GELU=gelu, RESIDUAL=residual is not None, device=self.device)
        self._record(ck)
        return y

    def _attention_kernel(self):
        # Compiled once for the full-context grid and launched with the actual
        # one. kernel[grid] would recompile per power-of-two grid and specialise
        # seq == 1; some of those variants exceed the register file (the
        # (2, 2) grid at BLOCK_M=4 does), while the full-context one fits.
        if self._attn is None:
            cfg = self.cfg
            signature = dict.fromkeys(("q_ptr", "k_ptr", "v_ptr", "o_ptr"), "*fp32")
            signature.update(seq="i32", sm_scale="fp32")
            constants = {"HEAD_DIM": cfg.head_dim, "BLOCK_M": ATTN_BLOCK_M, "BLOCK_N": 16,
                         "CAUSAL": True, "STRIDE": cfg.d_model}
            self._attn = compiler.compile(attention_kernel, signature, constants,
                                          grid=(triton_cdiv(cfg.ctx, ATTN_BLOCK_M), cfg.n_head))
        return self._attn

    def attention(self, q, k, v) -> np.ndarray:
        t = q.shape[0]
        cfg = self.cfg
        if not 0 < t <= cfg.ctx:      # the compiled program-id ranges cover ctx rows
            raise ValueError(f"attention over {t} positions; the kernel is compiled for 1..{cfg.ctx}")
        o = np.zeros_like(q)
        ck = self._attention_kernel()
        grid = (triton_cdiv(t, ATTN_BLOCK_M), cfg.n_head)
        values = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": o,
                  "seq": t, "sm_scale": np.float32(float(cfg.head_dim) ** -0.5)}
        if self.device == "emu":
            ck.last_stats = run_emulator(ck, grid, values)
        else:
            from piton.runtime.pi import run_on_pi
            ck.last_stats = run_on_pi(ck, grid, values)
        self._record(ck)
        return o

    # -- model ---------------------------------------------------------------
    def forward(self, ids: np.ndarray, last_only: bool = False) -> np.ndarray:
        """ids [T] -> logits [T, vocab] (or [1, vocab] for the last position)."""
        assert 0 < len(ids) <= self.cfg.ctx
        x = self.embed(ids)
        for layer in range(self.cfg.n_layer):
            p = f"h{layer}."
            a = self.layernorm(x, p + "ln1")
            q, k, v = (self.linear(a, p + w, p + b) for w, b in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")))
            y = self.attention(q, k, v)
            x = self.linear(y, p + "wo", p + "bo", residual=x)
            a = self.layernorm(x, p + "ln2")
            u = self.linear(a, p + "w1", p + "b1", gelu=True)
            x = self.linear(u, p + "w2", p + "b2", residual=x)
        if last_only:
            x = np.ascontiguousarray(x[-1:])
        return self.linear(self.layernorm(x, "lnf"), "w_head", "b_head")

    def generate(self, prompt_ids, n_new: int, temperature: float = 0.0, seed: int = 0,
                 on_token=None) -> np.ndarray:
        """Greedy (temperature 0) or seeded temperature sampling over a sliding window."""
        rng = np.random.default_rng(seed)
        ids = [int(i) for i in prompt_ids]
        if not ids:
            raise ValueError("empty prompt (no characters from the model's vocabulary)")
        for _ in range(n_new):
            logits = self.forward(np.array(ids[-self.cfg.ctx:], np.int32), last_only=True)[0]
            ids.append(sample(logits, temperature, rng))
            if on_token:
                on_token(ids[-1])
        return np.array(ids, np.int32)

    def estimated_seconds(self) -> float:
        return self.cycles * CLOCKS_PER_CYCLE / QPU_HZ


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def sample(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    return int(rng.choice(len(p), p=p / p.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="The QPU ")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="emu", choices=("emu", "pi"))
    args = ap.parse_args()
    cfg, tok, params, meta = load_model()
    model = TinyGPT(cfg, params, device=args.device)
    n_params = sum(v.size for v in params.values())
    print(f"TinyGPT: {n_params} params, d={cfg.d_model}, {cfg.n_layer} layers x {cfg.n_head} heads, "
          f"ctx {cfg.ctx}, vocab {cfg.vocab} (val loss {meta.get('val_loss')})")
    print(tok.decode(tok.encode(args.prompt)), end="", flush=True)
    t0 = time.time()
    model.generate(tok.encode(args.prompt), args.tokens, args.temperature, args.seed,
                   on_token=lambda i: print(tok.decode([i]), end="", flush=True))
    wall = time.time() - t0
    print()
    if args.device == "pi":
        qpu_s, how = model.pi_us * 1e-6, "measured by run_on_pi"
    else:
        qpu_s, how = model.estimated_seconds(), (f"{model.cycles} cycles, emulator timing-model estimate "
                                                 "at 250 MHz excluding launch/copy overhead")
    rate = f" = {args.tokens / qpu_s:.0f} tokens/s" if qpu_s > 0 and args.tokens else ""
    print(f"{model.launches} launches, {qpu_s * 1e3:.1f} ms of QPU time{rate} ({how}); "
          f"host wall time {wall:.1f}s")


if __name__ == "__main__":
    main()
