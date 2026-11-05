"""Scaled dot-product attention for one transformer head, flash-attention style.

softmax(Q K^T / sqrt(d)) V, written like the forward pass of Triton tutorial 06
("Fused Attention"), simplified to one head per grid row: each program owns
BLOCK_M query rows, keeps them in registers and walks the keys and values in
blocks of BLOCK_N.  A block costs two `tl.dot`s, Q K^T on a transposed load of
K and P V.  The softmax is computed online: a running row maximum `m_i` and
row sum `l_i` rescale the accumulator whenever the maximum grows, so the
N x N score matrix never reaches memory, and the division by `l_i` at the end
normalises.  As in the tutorial the key loop runs in two stages: blocks that
need no mask, then (only where needed) the blocks on the causal diagonal or
past a ragged N_CTX, with the masks applied.

Q, V and Out are [N_CTX, HEAD_DIM] fp32 with a row stride; K is read through
two strides, so it can be stored [N_CTX, HEAD_DIM] (stride_kn=HEAD_DIM,
stride_kk=1: the K^T tile is a gather) or pre-transposed [HEAD_DIM, N_CTX]
(stride_kn=1, stride_kk=N_CTX: contiguous rows).  Grid axis 1 walks heads
through the `stride_*h` arguments.  EVEN_N promises N_CTX % BLOCK_N == 0.

    grid = (triton.cdiv(N_CTX, BLOCK_M), n_heads)
    attn_fwd_kernel[grid](q, k, v, out, 1 / sqrt(d), ..., HEAD_DIM=d, ...)

    .venv/bin/python examples/attention.py       # emulator preview: the attention matrix as a PNG
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import piton as triton  # noqa: E402
import piton.language as tl  # noqa: E402


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K, V, lo, hi, qk_scale, N_CTX,
                    stride_kn, stride_kk, stride_vn, offs_m, offs_n, offs_d,
                    BLOCK_N: tl.constexpr, MASK_KEYS: tl.constexpr, CAUSAL_MASK: tl.constexpr):
    # K^T as a [HEAD_DIM, BLOCK_N] tile: one key per column
    k_ptrs = K + (lo + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + (lo + offs_n)[:, None] * stride_vn + offs_d[None, :]
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        if MASK_KEYS:
            k = tl.load(k_ptrs, mask=cols[None, :] < N_CTX, other=0.0)
        else:
            k = tl.load(k_ptrs)
        qk = tl.dot(q, k) * qk_scale
        if MASK_KEYS:
            qk = tl.where(cols[None, :] < N_CTX, qk, float("-inf"))
        if CAUSAL_MASK:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, float("-inf"))
        # online softmax: new running max, rescale what was summed so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        if MASK_KEYS:
            v = tl.load(v_ptrs, mask=cols[:, None] < N_CTX, other=0.0)
        else:
            v = tl.load(v_ptrs)
        acc = tl.dot(p, v, acc * alpha[:, None])
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def attn_fwd_kernel(Q, K, V, Out, sm_scale,
                    stride_qh, stride_qm, stride_kh, stride_kn, stride_kk,
                    stride_vh, stride_vn, stride_oh, stride_om,
                    N_CTX,
                    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    CAUSAL: tl.constexpr, EVEN_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_base = K + off_h * stride_kh
    v_base = V + off_h * stride_vh

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504      # exp(x) = exp2(x * log2(e))

    if CAUSAL:
        # stage 1: key blocks wholly before this program's first query row
        lo = (start_m * BLOCK_M) // BLOCK_N * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, 0, lo, qk_scale, N_CTX,
                                        stride_kn, stride_kk, stride_vn, offs_m, offs_n, offs_d,
                                        BLOCK_N, False, False)
        # stage 2: the diagonal, masked
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, lo, hi, qk_scale, N_CTX,
                                        stride_kn, stride_kk, stride_vn, offs_m, offs_n, offs_d,
                                        BLOCK_N, not EVEN_N, True)
    else:
        n_full = N_CTX // BLOCK_N * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, 0, n_full, qk_scale, N_CTX,
                                        stride_kn, stride_kk, stride_vn, offs_m, offs_n, offs_d,
                                        BLOCK_N, False, False)
        if not EVEN_N:
            # ragged tail: the last, partial key block
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_base, v_base, n_full, N_CTX, qk_scale,
                                            N_CTX, stride_kn, stride_kk, stride_vn, offs_m, offs_n,
                                            offs_d, BLOCK_N, True, False)

    acc = acc / l_i[:, None]
    o_ptrs = Out + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ---------------------------------------------------------------------------
# Host side: compiling, references, the emulator, pictures
# ---------------------------------------------------------------------------

import inspect  # noqa: E402
import struct  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

PARAMS = list(inspect.signature(attn_fwd_kernel.fn).parameters)

# Largest tiles (BLOCK_M x BLOCK_N) that fit the registers (4 accumulators, 2 x 31
# register-file entries; PIton does not spill) in the contiguous build
# (`contiguous_strides`), measured with this compiler:
#   HEAD_DIM 32: 4x16, 2x64, 1x64 (causal 2x32, 1x64); 8x16, 4x32 and 1x128 do not fit
#   HEAD_DIM 64: 2x32, 1x64 (causal 2x16, 1x64);       4x16 and 2x64 do not fit
# The tiles examples/attention_pi.py times on the Pi, per (HEAD_DIM, CAUSAL):
PI_TILES = {
    (32, False): [(2, 32), (2, 16), (4, 16), (2, 64)],
    (32, True): [(2, 32), (2, 16)],
    (64, False): [(2, 16), (2, 32)],
    (64, True): [(2, 16)],
}


def contiguous_strides(head_dim: int, k_layout: str = "nd", n_ctx: int | None = None) -> dict:
    """Row strides of contiguous [N_CTX, HEAD_DIM] Q, V and Out, with K stored
    [N_CTX, HEAD_DIM] (k_layout "nd") or pre-transposed [HEAD_DIM, N_CTX] ("dn")."""
    strides = dict(stride_qm=head_dim, stride_vn=head_dim, stride_om=head_dim)
    if k_layout == "nd":
        strides.update(stride_kn=head_dim, stride_kk=1)
    elif k_layout == "dn":
        if n_ctx is None:
            raise ValueError("a transposed K needs n_ctx (its row stride)")
        strides.update(stride_kn=1, stride_kk=n_ctx)
    else:
        raise ValueError(f"k_layout {k_layout!r}: expected 'nd' or 'dn'")
    return strides


def signature_for(constants: dict) -> dict:
    """PIton signature of every argument that is not a compile-time constant."""
    sig = {}
    for name in PARAMS:
        if name in constants:
            continue
        sig[name] = "*fp32" if name in ("Q", "K", "V", "Out") else "fp32" if name == "sm_scale" else "i32"
    return sig


def compile_attention(head_dim, block_m, block_n, causal=False, even_n=True, k_layout="nd",
                      n_ctx=None, max_n_ctx=512, heads=1, num_stages=2):
    """attn_fwd_kernel specialised for contiguous rows (strides are constants,
    N_CTX and the head strides stay runtime arguments), for launches of up to
    max_n_ctx tokens and `heads` heads."""
    from piton import compiler
    constants = dict(HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=bool(causal),
                     EVEN_N=bool(even_n), **contiguous_strides(head_dim, k_layout, n_ctx))
    grid = (-(-max_n_ctx // block_m), heads)
    return compiler.compile(attn_fwd_kernel, signature_for(constants), constants,
                            num_stages=num_stages, grid=grid)


def attention_reference(q, k, v, causal=False, sm_scale=None):
    """softmax(q k^T * sm_scale) v in float64, for [N, D] or [H, N, D] inputs."""
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    sm_scale = 1.0 / np.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    s = np.einsum("...md,...nd->...mn", q, k) * sm_scale
    if causal:
        n = s.shape[-1]
        s = np.where(np.tril(np.ones((s.shape[-2], n), bool)), s, -np.inf)
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def attention_flops(n_ctx, head_dim, causal=False, heads=1):
    """FLOPs of the two matrix products, counted as in Triton tutorial 06
    (4 N^2 D per head, half of that when causal); the softmax is not counted."""
    return 4.0 * n_ctx * n_ctx * head_dim * heads * (0.5 if causal else 1.0)


def run_attention(q, k, v, causal=False, block_m=2, block_n=16, compiled=None, k_layout="nd"):
    """Run the contiguous build on the strict emulator for [N, D] or [H, N, D]
    float32 q, k, v; returns (out, compiled kernel, emulator stats).  k_layout
    "dn" passes K pre-transposed ([H, D, N])."""
    from piton.runtime.launcher import run_emulator
    q, k, v = (np.ascontiguousarray(x, dtype=np.float32) for x in (q, k, v))
    squeeze = q.ndim == 2
    if squeeze:
        q, k, v = q[None], k[None], v[None]
    heads, n_ctx, head_dim = q.shape
    even_n = n_ctx % block_n == 0
    if compiled is None:
        compiled = compile_attention(head_dim, block_m, block_n, causal, even_n, k_layout,
                                     n_ctx=n_ctx, max_n_ctx=n_ctx, heads=heads)
    k_arg = np.ascontiguousarray(k.transpose(0, 2, 1)) if k_layout == "dn" else k
    out = np.full_like(q, np.nan)
    values = dict(Q=q, K=k_arg, V=v, Out=out, sm_scale=float(1.0 / np.sqrt(head_dim)), N_CTX=n_ctx,
                  stride_qh=n_ctx * head_dim, stride_kh=n_ctx * head_dim,
                  stride_vh=n_ctx * head_dim, stride_oh=n_ctx * head_dim)
    values = {name: value for name, value in values.items() if name in compiled.signature}
    grid = (-(-n_ctx // block_m), heads)
    stats = run_emulator(compiled, grid, values)
    return (out[0] if squeeze else out), compiled, stats


def pattern_qk(n_ctx=64, head_dim=64, vocab=8, seed=3):
    """Queries and keys with a visible attention pattern: half of each vector
    is a sinusoidal position encoding (q_i . k_j falls off with |i - j|: the
    diagonal band), the other half the embedding of one of `vocab` random
    tokens (orthonormal, x4: tokens attend to earlier and later copies of
    themselves, the off-diagonal dots).  Q = K."""
    half = head_dim // 2
    pos = np.arange(n_ctx)[:, None]
    freq = 1.0 / 16.0 ** (np.arange(0, half, 2) / half)
    pe = np.zeros((n_ctx, half))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, n_ctx)
    embed = np.linalg.qr(rng.standard_normal((half, half)))[0][:vocab]
    x = np.concatenate([pe, 4.0 * embed[tokens]], axis=1).astype(np.float32)
    return x, x.copy()


# -- pictures (zlib + struct, no imaging library needed) ---------------------

_MAGMA = np.array([[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
                   [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]], np.float64)


def colormap(values, lo=0.0, hi=1.0):
    """Map values in [lo, hi] onto a magma-like ramp; returns uint8 RGB."""
    t = np.clip((np.asarray(values, np.float64) - lo) / (hi - lo), 0.0, 1.0) * (len(_MAGMA) - 1)
    i = np.minimum(t.astype(int), len(_MAGMA) - 2)
    f = (t - i)[..., None]
    return np.round(_MAGMA[i] * (1 - f) + _MAGMA[i + 1] * f).astype(np.uint8)


def write_png(path, rgb):
    """Write an [H, W, 3] uint8 image as a PNG."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def weights_image(panels, scale=4, gap=8):
    """Attention matrices side by side, each pixel scale x scale; every row is
    shown relative to its largest weight, on a square-root scale."""
    def shade(p):
        p = np.nan_to_num(np.asarray(p, np.float64), nan=0.0)
        top = np.maximum(p.max(axis=1, keepdims=True), 1e-30)
        return colormap(np.sqrt(np.clip(p / top, 0, 1)))
    tiles = [np.kron(shade(p), np.ones((scale, scale, 1), np.uint8)) for p in panels]
    h = max(t.shape[0] for t in tiles)
    out = np.full((h, sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1), 3), 255, np.uint8)
    x = 0
    for t in tiles:
        out[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    return out


def main():
    """Preview in the emulator: with V = I the output is the attention matrix
    itself; writes docs/images/attention-weights.png (full and causal)."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    n = d = 64
    q, k = pattern_qk(n, d)
    eye = np.eye(n, dtype=np.float32)
    panels = []
    for causal in (False, True):
        p, ck, stats = run_attention(q, k, eye, causal=causal, block_m=2, block_n=16)
        ref = attention_reference(q, k, eye, causal)
        print(f"causal={causal}: {len(ck.words)} instructions, max |error| {np.abs(p - ref).max():.2e}, "
              f"rows sum to 1 within {np.abs(p.sum(1) - 1).max():.1e}")
        panels.append(p)
    path = os.path.join(root, "docs", "images", "attention-weights.png")
    write_png(path, weights_image(panels))
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
