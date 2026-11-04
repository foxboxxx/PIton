"""Fused (flash-style) attention: softmax(Q K^T * scale [+ causal mask]) V.

Q, K, V, O are contiguous FP32 [SEQ, HEAD_DIM] (one head). Each program owns
BLOCK_M query rows and streams K/V in BLOCK_N tiles with an online softmax, so
the SEQ x SEQ score matrix is never written to memory.

HEAD_DIM and BLOCK_M must be powers of two. Register pressure limits the tile:
roughly BLOCK_M * HEAD_DIM <= 64 (d=16: BLOCK_M<=4, d=32: 2, d=64: 1) with
BLOCK_N=16; larger tiles raise RegAllocError.
"""
import piton
import piton.language as tl


@piton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, seq, sm_scale,
                     HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    cols = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + rows[:, None] * HEAD_DIM + d[None, :], mask=rows[:, None] < seq, other=0.0)
    q = q * sm_scale

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - 1.0e30
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # Causal tiles past the diagonal are masked, not skipped: a data-dependent
    # loop bound pushes register pressure over the limit at BLOCK_M=4.
    for start in range(0, seq, BLOCK_N):
        kc = start + cols
        kt = tl.load(k_ptr + kc[None, :] * HEAD_DIM + d[:, None], mask=kc[None, :] < seq, other=0.0)
        s = tl.dot(q, kt)
        valid = kc[None, :] < seq
        if CAUSAL:
            valid = valid & (kc[None, :] <= rows[:, None])
        s = tl.where(valid, s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kc[:, None] * HEAD_DIM + d[None, :], mask=kc[:, None] < seq, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + rows[:, None] * HEAD_DIM + d[None, :], acc, mask=rows[:, None] < seq)
