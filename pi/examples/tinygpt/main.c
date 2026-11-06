// TinyGPT, resident on the Pi: a character-level transformer whose every layer
// is a PIton-compiled Triton kernel on the 12 QPUs. The ARM only samples.
//
// Built by examples/tinygpt_pi.py, which generates tg_model.h (config, fp16
// weights, float64 reference) and tg_kernels.h (launchers + tile sizes).
//
//   prefill(T tokens)   embed -> per layer: LN, Q/K/V (K/V into the cache),
//                       causal attention, O + residual, LN, FC1+GELU, FC2 +
//                       residual -> final LN + head on the last row
//   decode(1 token)     the same with M = 1 against the KV cache
//
// When the context fills, the last TG_CTX/2 tokens are prefilled again
// (positions are absolute). arm_* is the same algorithm in C on the ARM1176
// (VFP, caches on), used as the baseline and as a second on-device check.
//
// Every launcher allocates its uniforms (a mailbox call) on first use, so
// both paths run once at the largest grids before the ARM caches are turned
// on. GPU memory lies above CACHED_LIMIT (arm_sgemm.c), so ARM accesses to it
// stay uncached and coherent with the QPUs.
#include "rpi.h"
#include "piton_rt.h"
#include "tg_model.h"
#include "tg_kernels.h"

#define CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// MMU + ARM caches, as pi/examples/matmul/arm_sgemm.c (copied so that this
// program needs none of its VFP assembly)

static uint32_t page_table[4096] __attribute__((aligned(16384)));
static uint32_t saved_fpscr;

#define CACHED_LIMIT 0x08000000u        // program, bss, stack (STACK_ADDR = 0x8000000)

static inline void sync_barrier(void) { asm volatile("mcr p15, 0, %0, c7, c10, 4" :: "r"(0) : "memory"); }
static inline void prefetch_flush_(void) { asm volatile("mcr p15, 0, %0, c7, c5, 4" :: "r"(0) : "memory"); }

static void arm_caches_on(void) {
    for (uint32_t i = 0; i < 4096; i++) {
        uint32_t base = i << 20;
        uint32_t d = base | (3u << 10) | 2u;          // section, AP = full access, domain 0
        if (base < CACHED_LIMIT)
            d |= (1u << 12) | (1u << 3) | (1u << 2);  // TEX=001 C=1 B=1: write-back, write-allocate
        else
            d |= 1u << 4;                             // TEX=000 C=0 B=0: strongly ordered, no execute
        page_table[i] = d;
    }
    sync_barrier();
    asm volatile("mcr p15, 0, %0, c2, c0, 2" :: "r"(0));                // TTBCR: TTBR0 only
    asm volatile("mcr p15, 0, %0, c2, c0, 0" :: "r"(page_table));       // TTBR0 (walks uncached)
    asm volatile("mcr p15, 0, %0, c3, c0, 0" :: "r"(3));                // domain 0: manager
    asm volatile("mcr p15, 0, %0, c8, c7, 0" :: "r"(0));                // invalidate TLBs
    asm volatile("mcr p15, 0, %0, c7, c7, 0" :: "r"(0));                // invalidate I + D caches
    sync_barrier();
    uint32_t r;
    asm volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(r));
    r |= (1u << 23) | (1u << 12) | (1u << 11) | (1u << 2) | 1u;         // XP, I, Z, C, M
    asm volatile("mcr p15, 0, %0, c1, c0, 0" :: "r"(r) : "memory");
    prefetch_flush_();
    // VFP RunFast mode (flush-to-zero, default NaN): no support-code traps
    asm volatile("vmrs %0, fpscr" : "=r"(saved_fpscr));
    asm volatile("vmsr fpscr, %0" :: "r"(saved_fpscr | (1u << 24) | (1u << 25)));
}

static void arm_caches_off(void) {
    asm volatile("vmsr fpscr, %0" :: "r"(saved_fpscr));
    asm volatile("mcr p15, 0, %0, c7, c14, 0" :: "r"(0) : "memory");   // clean + invalidate D
    sync_barrier();
    uint32_t r;
    asm volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(r));
    r &= ~((1u << 12) | (1u << 11) | (1u << 2) | 1u);
    asm volatile("mcr p15, 0, %0, c1, c0, 0" :: "r"(r) : "memory");
    prefetch_flush_();
    asm volatile("mcr p15, 0, %0, c7, c7, 0" :: "r"(0));                // invalidate I + D
    asm volatile("mcr p15, 0, %0, c8, c7, 0" :: "r"(0));                // invalidate TLBs
    sync_barrier();
}

// ---------------------------------------------------------------------------
// small float helpers (freestanding: no libm)

static inline float tg_sqrtf(float x) {
    float r;
    asm("vsqrt.f32 %0, %1" : "=w"(r) : "w"(x));
    return r;
}

static float tg_expf(float x) {
    // exp(x) = 2^n * exp(r), |r| <= ln2/2, degree-6 Taylor (rel. error ~2e-7)
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float t = x * 1.44269504f;
    int n = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
    float r = x - (float)n * 0.693145752f - (float)n * 1.42860677e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f +
              r * (0.00833333377f + r * 0.00138888892f)))));
    union { float f; uint32_t u; } s = { .u = (uint32_t)(n + 127) << 23 };
    return p * s.f;
}

static float absf(float x) { return x < 0.0f ? -x : x; }

// libpi's printk has no %f: x as [-]I.FFFFFFe[+-]EE (7 significant digits)
static void print_float(float x) {
    if (x != x) { printk("nan"); return; }
    if (x < 0.0f) { printk("-"); x = -x; }
    if (x > 3.4e38f) { printk("inf"); return; }
    int e = 0;
    if (x != 0.0f) {
        while (x >= 10.0f) { x *= 0.1f; e++; }
        while (x < 1.0f) { x *= 10.0f; e--; }
    }
    uint32_t m = (uint32_t)(x * 1000000.0f + 0.5f);
    if (m >= 10000000u) { m /= 10u; e++; }
    printk("%d.%d%d%d%d%d%de%s%d", m / 1000000u, m / 100000u % 10u, m / 10000u % 10u, m / 1000u % 10u,
           m / 100u % 10u, m / 10u % 10u, m % 10u, e < 0 ? "-" : "+", e < 0 ? -e : e);
}

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {                                // fp16 subnormal -> fp32 normal
            e = 127 - 15 + 1;
            while (!(m & 1024)) { m <<= 1; e--; }
            bits = s | (e << 23) | ((m & 1023) << 13);
        }
    } else if (e == 31) {
        bits = s | (255u << 23) | (m << 13);
    } else {
        bits = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    union { uint32_t u; float f; } v = { .u = bits };
    return v.f;
}

// ---------------------------------------------------------------------------
// weights and activations

static float arm_w[TG_NW];                       // ARM copy (cached RAM)
static piton_buf_t gw;                           // GPU copy, same layout
static piton_buf_t g_ids, g_x, g_a, g_q, g_o, g_u, g_kc, g_vc, g_logits;

#define WB(off) (gw.bus + 4u * (off))
#define ROWB(buf, row, width) ((buf).bus + 4u * (uint32_t)(row) * (width))

static void setup_gpu(void) {
    gw = piton_alloc(TG_NW * 4);
    volatile float *w = gw.cpu;
    for (unsigned i = 0; i < TG_NW; i++)
        w[i] = arm_w[i] = h2f(tg_weights[i]);
    g_ids = piton_alloc(TG_CTX * 4);
    g_x = piton_alloc(TG_CTX * TG_D * 4);
    g_a = piton_alloc(TG_CTX * TG_D * 4);
    g_q = piton_alloc(TG_CTX * TG_D * 4);
    g_o = piton_alloc(TG_CTX * TG_D * 4);
    g_u = piton_alloc(TG_CTX * TG_F * 4);
    g_kc = piton_alloc(TG_L * TG_CTX * TG_D * 4);
    g_vc = piton_alloc(TG_L * TG_CTX * TG_D * 4);
    g_logits = piton_alloc(((TG_VOCAB + 15) & ~15) * 4);
}

// ---------------------------------------------------------------------------
// QPU path

static void q_ln(int prefill, uint32_t x, uint32_t g, uint32_t b, uint32_t y, unsigned rows) {
    tg_ln_p_args_t a = { .x_ptr = x, .g_ptr = WB(g), .b_ptr = WB(b), .y_ptr = y, .M = rows, .eps = TG_EPS };
    if (prefill)
        tg_ln_p_launch(&a, CDIV(rows, TG_LN_P_BM), 1, 1);
    else
        tg_ln_d_launch((const tg_ln_d_args_t *)&a, rows, 1, 1);
}

#define LINEAR(name, xb, w, b, res, y, rows, n) do {                                   \
        tg_##name##_args_t la = { .x_ptr = (xb), .w_ptr = WB(w), .bias_ptr = WB(b),     \
                                  .res_ptr = (res), .y_ptr = (y), .M = (rows) };         \
        tg_##name##_launch(&la, CDIV(rows, TG_##name##_BM) * CDIV(n, TG_##name##_BN)  \
                           , 1, 1);                                                      \
    } while (0)
// tile macros are upper case; map the lower-case names used above
#define TG_lin_qkv_p_BM TG_LIN_QKV_P_BM
#define TG_lin_qkv_p_BN TG_LIN_QKV_P_BN
#define TG_lin_qkv_d_BM TG_LIN_QKV_D_BM
#define TG_lin_qkv_d_BN TG_LIN_QKV_D_BN
#define TG_lin_o_p_BM TG_LIN_O_P_BM
#define TG_lin_o_p_BN TG_LIN_O_P_BN
#define TG_lin_o_d_BM TG_LIN_O_D_BM
#define TG_lin_o_d_BN TG_LIN_O_D_BN
#define TG_lin_fc1_p_BM TG_LIN_FC1_P_BM
#define TG_lin_fc1_p_BN TG_LIN_FC1_P_BN
#define TG_lin_fc1_d_BM TG_LIN_FC1_D_BM
#define TG_lin_fc1_d_BN TG_LIN_FC1_D_BN
#define TG_lin_fc2_p_BM TG_LIN_FC2_P_BM
#define TG_lin_fc2_p_BN TG_LIN_FC2_P_BN
#define TG_lin_fc2_d_BM TG_LIN_FC2_D_BM
#define TG_lin_fc2_d_BN TG_LIN_FC2_D_BN
#define TG_lin_head_d_BM TG_LIN_HEAD_D_BM
#define TG_lin_head_d_BN TG_LIN_HEAD_D_BN

static void q_block(int prefill, unsigned layer, unsigned t, unsigned rows) {
    uint32_t kc = g_kc.bus + 4u * layer * TG_CTX * TG_D, vc = g_vc.bus + 4u * layer * TG_CTX * TG_D;
    uint32_t kt = kc + 4u * t * TG_D, vt = vc + 4u * t * TG_D;
    q_ln(prefill, g_x.bus, tg_off_ln1_g[layer], tg_off_ln1_b[layer], g_a.bus, rows);
    if (prefill) {
        LINEAR(lin_qkv_p, g_a.bus, tg_off_wq[layer], tg_off_bq[layer], g_q.bus, g_q.bus, rows, TG_D);
        LINEAR(lin_qkv_p, g_a.bus, tg_off_wk[layer], tg_off_bk[layer], kt, kt, rows, TG_D);
        LINEAR(lin_qkv_p, g_a.bus, tg_off_wv[layer], tg_off_bv[layer], vt, vt, rows, TG_D);
        tg_attn_p_args_t at = { .q_ptr = g_q.bus, .k_ptr = kc, .v_ptr = vc, .o_ptr = g_o.bus,
                                .seq = t + rows, .sm_scale = TG_SM_SCALE };
        tg_attn_p_launch(&at, CDIV(rows, TG_ATTN_P_BM), TG_H, 1);
        LINEAR(lin_o_p, g_o.bus, tg_off_wo[layer], tg_off_bo[layer], g_x.bus, g_x.bus, rows, TG_D);
        q_ln(1, g_x.bus, tg_off_ln2_g[layer], tg_off_ln2_b[layer], g_a.bus, rows);
        LINEAR(lin_fc1_p, g_a.bus, tg_off_w1[layer], tg_off_b1[layer], g_u.bus, g_u.bus, rows, TG_F);
        LINEAR(lin_fc2_p, g_u.bus, tg_off_w2[layer], tg_off_b2[layer], g_x.bus, g_x.bus, rows, TG_D);
    } else {
        LINEAR(lin_qkv_d, g_a.bus, tg_off_wq[layer], tg_off_bq[layer], g_q.bus, g_q.bus, 1, TG_D);
        LINEAR(lin_qkv_d, g_a.bus, tg_off_wk[layer], tg_off_bk[layer], kt, kt, 1, TG_D);
        LINEAR(lin_qkv_d, g_a.bus, tg_off_wv[layer], tg_off_bv[layer], vt, vt, 1, TG_D);
        tg_attn_d_args_t at = { .q_ptr = g_q.bus, .k_ptr = kc, .v_ptr = vc, .o_ptr = g_o.bus,
                                .seq = t + 1, .sm_scale = TG_SM_SCALE };
        tg_attn_d_launch(&at, 1, TG_H, 1);
        LINEAR(lin_o_d, g_o.bus, tg_off_wo[layer], tg_off_bo[layer], g_x.bus, g_x.bus, 1, TG_D);
        q_ln(0, g_x.bus, tg_off_ln2_g[layer], tg_off_ln2_b[layer], g_a.bus, 1);
        LINEAR(lin_fc1_d, g_a.bus, tg_off_w1[layer], tg_off_b1[layer], g_u.bus, g_u.bus, 1, TG_F);
        LINEAR(lin_fc2_d, g_u.bus, tg_off_w2[layer], tg_off_b2[layer], g_x.bus, g_x.bus, 1, TG_D);
    }
}

static void q_head(unsigned row, float *logits) {
    q_ln(0, ROWB(g_x, row, TG_D), TG_OFF_LNF_G, TG_OFF_LNF_B, g_a.bus, 1);
    LINEAR(lin_head_d, g_a.bus, TG_OFF_W_HEAD, TG_OFF_B_HEAD, g_logits.bus, g_logits.bus, 1, TG_VOCAB);
    volatile float *l = g_logits.cpu;
    for (unsigned i = 0; i < TG_VOCAB; i++)
        logits[i] = l[i];
}

static void q_prefill(const int32_t *ids, unsigned t, float *logits) {
    volatile int32_t *gi = g_ids.cpu;
    for (unsigned i = 0; i < t; i++)
        gi[i] = ids[i];
    tg_embed_p_args_t e = { .ids_ptr = g_ids.bus, .tok_ptr = WB(TG_OFF_TOK_EMB),
                            .pos_ptr = WB(TG_OFF_POS_EMB), .x_ptr = g_x.bus, .T = t };
    tg_embed_p_launch(&e, CDIV(t, TG_EMBED_P_BM), 1, 1);
    for (unsigned layer = 0; layer < TG_L; layer++)
        q_block(1, layer, 0, t);
    q_head(t - 1, logits);
}

static void q_decode(int32_t token, unsigned t, float *logits) {
    ((volatile int32_t *)g_ids.cpu)[t] = token;
    tg_embed_d_args_t e = { .ids_ptr = g_ids.bus + 4u * t, .tok_ptr = WB(TG_OFF_TOK_EMB),
                            .pos_ptr = WB(TG_OFF_POS_EMB + t * TG_D), .x_ptr = g_x.bus, .T = 1 };
    tg_embed_d_launch(&e, 1, 1, 1);
    for (unsigned layer = 0; layer < TG_L; layer++)
        q_block(0, layer, t, 1);
    q_head(0, logits);
}

// ---------------------------------------------------------------------------
// ARM path: the same algorithm (KV cache, sliding re-prefill) in C

static float ax[TG_CTX * TG_D], aa[TG_CTX * TG_D], aq[TG_CTX * TG_D], ao[TG_CTX * TG_D];
static float au[TG_CTX * TG_F], akc[TG_L * TG_CTX * TG_D], avc[TG_L * TG_CTX * TG_D];
static float ascore[TG_CTX];

static void a_ln(const float *x, unsigned g, unsigned b, float *y, unsigned rows) {
    const float *gg = arm_w + g, *bb = arm_w + b;
    for (unsigned r = 0; r < rows; r++, x += TG_D, y += TG_D) {
        float mean = 0.0f, var = 0.0f;
        for (unsigned i = 0; i < TG_D; i++) mean += x[i];
        mean *= 1.0f / TG_D;
        for (unsigned i = 0; i < TG_D; i++) { float c = x[i] - mean; var += c * c; }
        float rstd = 1.0f / tg_sqrtf(var * (1.0f / TG_D) + TG_EPS);
        for (unsigned i = 0; i < TG_D; i++) y[i] = (x[i] - mean) * rstd * gg[i] + bb[i];
    }
}

// y[rows, n] = [y +] act(x[rows, k] @ w[k, n] + b); y may be the residual (accumulate)
static void a_linear(const float *x, unsigned w, unsigned b, float *y, unsigned rows,
                     unsigned n, unsigned k, int gelu, int residual) {
    static float acc[TG_F > TG_VOCAB ? (TG_F > TG_D ? TG_F : TG_D) : (TG_VOCAB > TG_D ? TG_VOCAB : TG_D)];
    const float *ww = arm_w + w, *bb = arm_w + b;
    for (unsigned r = 0; r < rows; r++, x += k, y += n) {
        for (unsigned j = 0; j < n; j++) acc[j] = bb[j];
        for (unsigned i = 0; i < k; i++) {
            float xi = x[i];
            const float *wr = ww + i * n;
            for (unsigned j = 0; j < n; j++) acc[j] += xi * wr[j];
        }
        for (unsigned j = 0; j < n; j++) {
            float v = acc[j];
            if (gelu) v = v / (1.0f + tg_expf(-1.702f * v));
            y[j] = residual ? y[j] + v : v;
        }
    }
}

// rows t0 .. t0+rows-1 attend causally to cache rows 0 .. their own position
static void a_attention(const float *q, const float *kc, const float *vc, float *o, unsigned t0, unsigned rows) {
    for (unsigned r = 0; r < rows; r++) {
        unsigned pos = t0 + r;
        for (unsigned h = 0; h < TG_H; h++) {
            const float *qr = q + r * TG_D + h * TG_HD;
            float m = -1e30f, l = 0.0f;
            for (unsigned j = 0; j <= pos; j++) {
                const float *kr = kc + j * TG_D + h * TG_HD;
                float s = 0.0f;
                for (unsigned i = 0; i < TG_HD; i++) s += qr[i] * kr[i];
                s *= TG_SM_SCALE;
                ascore[j] = s;
                if (s > m) m = s;
            }
            float *orow = o + r * TG_D + h * TG_HD;
            for (unsigned i = 0; i < TG_HD; i++) orow[i] = 0.0f;
            for (unsigned j = 0; j <= pos; j++) {
                float p = tg_expf(ascore[j] - m);
                l += p;
                const float *vr = vc + j * TG_D + h * TG_HD;
                for (unsigned i = 0; i < TG_HD; i++) orow[i] += p * vr[i];
            }
            float inv = 1.0f / l;
            for (unsigned i = 0; i < TG_HD; i++) orow[i] *= inv;
        }
    }
}

static void a_forward(const int32_t *ids, unsigned t0, unsigned rows, float *logits) {
    // embed rows t0 .. t0+rows-1 into ax[0 .. rows)
    for (unsigned r = 0; r < rows; r++)
        for (unsigned i = 0; i < TG_D; i++)
            ax[r * TG_D + i] = arm_w[TG_OFF_TOK_EMB + ids[r] * TG_D + i] +
                               arm_w[TG_OFF_POS_EMB + (t0 + r) * TG_D + i];
    for (unsigned layer = 0; layer < TG_L; layer++) {
        float *kc = akc + layer * TG_CTX * TG_D, *vc = avc + layer * TG_CTX * TG_D;
        a_ln(ax, tg_off_ln1_g[layer], tg_off_ln1_b[layer], aa, rows);
        a_linear(aa, tg_off_wq[layer], tg_off_bq[layer], aq, rows, TG_D, TG_D, 0, 0);
        a_linear(aa, tg_off_wk[layer], tg_off_bk[layer], kc + t0 * TG_D, rows, TG_D, TG_D, 0, 0);
        a_linear(aa, tg_off_wv[layer], tg_off_bv[layer], vc + t0 * TG_D, rows, TG_D, TG_D, 0, 0);
        a_attention(aq, kc, vc, ao, t0, rows);
        a_linear(ao, tg_off_wo[layer], tg_off_bo[layer], ax, rows, TG_D, TG_D, 0, 1);
        a_ln(ax, tg_off_ln2_g[layer], tg_off_ln2_b[layer], aa, rows);
        a_linear(aa, tg_off_w1[layer], tg_off_b1[layer], au, rows, TG_F, TG_D, 1, 0);
        a_linear(au, tg_off_w2[layer], tg_off_b2[layer], ax, rows, TG_D, TG_F, 0, 1);
    }
    a_ln(ax + (rows - 1) * TG_D, TG_OFF_LNF_G, TG_OFF_LNF_B, aa, 1);
    a_linear(aa, TG_OFF_W_HEAD, TG_OFF_B_HEAD, logits, 1, TG_VOCAB, TG_D, 0, 0);
}

// ---------------------------------------------------------------------------
// generation (shared by both paths)

typedef struct {
    int qpu;
    int32_t ids[TG_CTX];
    unsigned n;            // tokens in the window (all in the cache)
    float logits[TG_VOCAB];
} gen_t;

static void g_prefill(gen_t *g) {
    if (g->qpu) q_prefill(g->ids, g->n, g->logits);
    else a_forward(g->ids, 0, g->n, g->logits);
}

// append a token; slide the window (re-prefill the last half) when it is full
static void g_push(gen_t *g, int32_t token) {
    if (g->n == TG_CTX) {
        unsigned keep = TG_CTX / 2;
        for (unsigned i = 0; i < keep - 1; i++)
            g->ids[i] = g->ids[TG_CTX - keep + 1 + i];
        g->ids[keep - 1] = token;
        g->n = keep;
        g_prefill(g);
        return;
    }
    g->ids[g->n] = token;
    if (g->qpu) q_decode(token, g->n, g->logits);
    else a_forward(&g->ids[g->n], g->n, 1, g->logits);
    g->n++;
}

static uint32_t rng_state = TG_SEED;

static int32_t sample(const float *logits, float temperature) {
    int32_t best = 0;
    for (int32_t i = 1; i < TG_VOCAB; i++)
        if (logits[i] > logits[best]) best = i;
    if (temperature <= 0.0f)
        return best;
    static float p[TG_VOCAB];
    float sum = 0.0f, inv_t = 1.0f / temperature;
    for (int32_t i = 0; i < TG_VOCAB; i++) sum += p[i] = tg_expf((logits[i] - logits[best]) * inv_t);
    float r = (float)(piton_lcg(&rng_state) >> 8) * (1.0f / 16777216.0f) * sum;
    for (int32_t i = 0; i < TG_VOCAB; i++)
        if ((r -= p[i]) < 0.0f) return i;
    return best;
}

static void put_char(int32_t id) {
    uart_put8((uint8_t)tg_chars[id]);   // pi-install turns '\r' into a space: send bare '\n'
}

// ---------------------------------------------------------------------------
// checks and benchmark

static int check(int qpu) {
    gen_t g = { .qpu = qpu, .n = TG_PROMPT_LEN };
    for (unsigned i = 0; i < TG_PROMPT_LEN; i++) g.ids[i] = tg_ref_ids[i];
    float max_err = 0.0f, max_ref = 0.0f;
    unsigned failures = 0;
    g_prefill(&g);
    for (unsigned step = 0; step < TG_NREF; step++) {
        if (step) g_push(&g, tg_ref_ids[TG_PROMPT_LEN + step - 1]);   // teacher forcing
        for (unsigned i = 0; i < TG_VOCAB; i++) {
            float ref = tg_ref_logits[step * TG_VOCAB + i], err = absf(g.logits[i] - ref);
            if (!(err <= 2e-3f + 1e-3f * absf(ref))) failures++;
            if (err > max_err || err != err) max_err = err;
            if (absf(ref) > max_ref) max_ref = absf(ref);
        }
    }
    printk("TG_CHECK impl=%s steps=%d max_abs_error=", qpu ? "qpu" : "arm", TG_NREF);
    print_float(max_err);
    printk(" failures=%d max_abs_logit=", failures);
    print_float(max_ref);
    printk("\n");
    return failures == 0;
}

static int32_t bench_ids[2][TG_NGEN];

static void bench(int qpu) {
    gen_t g = { .qpu = qpu, .n = TG_PROMPT_LEN };
    for (unsigned i = 0; i < TG_PROMPT_LEN; i++) g.ids[i] = tg_ref_ids[i];
    uint32_t t0 = timer_get_usec();
    g_prefill(&g);
    uint32_t t1 = timer_get_usec();
    for (unsigned i = 0; i < TG_NGEN; i++) {
        int32_t next = sample(g.logits, 0.0f);
        bench_ids[qpu][i] = next;
        if (i + 1 < TG_NGEN) g_push(&g, next);
    }
    uint32_t t2 = timer_get_usec();
    printk("TG_BENCH impl=%s tokens=%d prefill_us=%d decode_us=%d total_us=%d\n",
           qpu ? "qpu" : "arm", TG_NGEN, t1 - t0, t2 - t1, t2 - t0);
    printk("TG_TEXT impl=%s ", qpu ? "qpu" : "arm");
    for (unsigned i = 0; i < TG_PROMPT_LEN; i++) printk("%c", tg_chars[tg_ref_ids[i]] == '\n' ? '|' : tg_chars[tg_ref_ids[i]]);
    for (unsigned i = 0; i < TG_NGEN; i++) printk("%c", tg_chars[bench_ids[qpu][i]] == '\n' ? '|' : tg_chars[bench_ids[qpu][i]]);
    printk("\n");
}

// ---------------------------------------------------------------------------
// interactive loop (TG_CHAT): prompts arrive line by line through pi-install

#if TG_CHAT
static int32_t char_id(char c) {
    for (int32_t i = 0; tg_chars[i]; i++)
        if (tg_chars[i] == c) return i;
    return -1;
}

static unsigned read_line(char *buf, unsigned cap) {
    unsigned n = 0;
    for (;;) {
        int c = uart_get8();
        if (c == '\r') continue;
        if (c == '\n') break;
        if (n + 1 < cap) buf[n++] = (char)c;
    }
    buf[n] = 0;
    return n;
}

static int starts(const char *s, const char *p) {
    while (*p) if (*s++ != *p++) return 0;
    return 1;
}

static unsigned parse_uint(const char *s) {
    unsigned v = 0;
    while (*s == ' ') s++;
    while (*s >= '0' && *s <= '9') v = v * 10 + (unsigned)(*s++ - '0');
    return v;
}

static void chat(void) {
    static char line[512];
    gen_t g = { .qpu = 1, .n = 0 };
    float temperature = 0.8f;
    unsigned len = TG_CHAT_LEN;
    int use_qpu = 1;
    printk("\nTinyGPT-%s ready (%d params, %d chars of context). Type a prompt and press Enter.\n"
           "An empty line continues the text; /help lists commands.\n", TG_SIZE_NAME, TG_NW, TG_CTX);
    for (;;) {
        printk("\n> ");
        read_line(line, sizeof line);
        if (starts(line, "/quit")) break;
        if (starts(line, "/help")) {
            printk("/temp N   temperature N/100 (0 = greedy; now %d)\n"
                   "/len N    characters per reply (now %d)\n"
                   "/arm      run on the ARM CPU instead   /qpu  back to the QPUs (now %s)\n"
                   "/seed N   reseed sampling\n/quit     exit (the Pi reboots)\n"
                   "Any key while it is writing stops the reply.\n",
                   (int)(temperature * 100.0f + 0.5f), len, use_qpu ? "qpu" : "arm");
            continue;
        }
        if (starts(line, "/temp")) {
            unsigned t = parse_uint(line + 5);
            temperature = (float)t * 0.01f;
            printk("temperature %d/100%s\n", t, t ? "" : " (greedy)");
            continue;
        }
        if (starts(line, "/len")) { len = parse_uint(line + 4); printk("%d chars per reply\n", len); continue; }
        if (starts(line, "/seed")) { rng_state = parse_uint(line + 5); printk("reseeded\n"); continue; }
        if (starts(line, "/arm") || starts(line, "/qpu")) {
            use_qpu = starts(line, "/qpu");
            g.n = 0;
            printk("now on the %s\n", use_qpu ? "QPUs" : "ARM CPU");
            continue;
        }
        uint32_t t0 = timer_get_usec();
        if (line[0]) {                         // new prompt: start a fresh context
            g.qpu = use_qpu;
            g.n = 0;
            for (unsigned i = 0; line[i]; i++) {
                int32_t id = char_id(line[i]);
                if (id < 0) continue;
                if (g.n == TG_CTX) {           // keep the last TG_CTX characters
                    for (unsigned j = 1; j < TG_CTX; j++) g.ids[j - 1] = g.ids[j];
                    g.n--;
                }
                g.ids[g.n++] = id;
            }
            if (!g.n) { printk("(no characters from the model's vocabulary)\n"); continue; }
            g_prefill(&g);
            for (unsigned i = 0; i < g.n; i++) put_char(g.ids[i]);
        } else if (!g.n) {
            continue;
        }
        unsigned made = 0;
        for (; made < len; made++) {
            if (uart_has_data()) { while (uart_has_data()) uart_get8(); break; }
            int32_t next = sample(g.logits, temperature);
            put_char(next);
            g_push(&g, next);
        }
        uint32_t us = timer_get_usec() - t0;
        printk("\n[%d chars in %d ms on the %s: %d chars/s]", made, us / 1000, g.qpu ? "QPUs" : "ARM",
               us ? (int)((uint64_t)made * 1000000u / us) : 0);
    }
}
#endif

static void vfp_enable(void) {
    uint32_t r;
    asm volatile("mrc p15, 0, %0, c1, c0, 2" : "=r"(r));
    r |= 0xf << 20;                                    // cp10, cp11: full access
    asm volatile("mcr p15, 0, %0, c1, c0, 2" :: "r"(r));
    asm volatile("mcr p15, 0, %0, c7, c5, 4" :: "r"(0));
    asm volatile("vmsr fpexc, %0" :: "r"(1u << 30));  // EN
}

void notmain(void) {
    vfp_enable();
    printk("PITON_START\n");
    piton_init();
    printk("TinyGPT-%s: %d weights, d=%d, %d layers x %d heads, ctx %d, vocab %d, %d QPUs\n",
           TG_SIZE_NAME, TG_NW, TG_D, TG_L, TG_H, TG_CTX, TG_VOCAB, piton_qpu_count());
    printk("TG_CLOCK arm=%d core=%d v3d=%d sdram=%d MHz\n",
           piton_clock_hz(PITON_CLOCK_ARM, 0) / 1000000, piton_clock_hz(PITON_CLOCK_CORE, 0) / 1000000,
           piton_clock_hz(PITON_CLOCK_V3D, 0) / 1000000, piton_clock_hz(PITON_CLOCK_SDRAM, 0) / 1000000);
    setup_gpu();
    // warm-up at the largest grids: every launcher allocates its uniforms now
    {
        static float logits[TG_VOCAB];
        static int32_t ids[TG_CTX];
        for (unsigned i = 0; i < TG_CTX; i++) ids[i] = tg_ref_ids[i];
        q_prefill(ids, TG_CTX, logits);
        q_decode(ids[0], 0, logits);
    }
    arm_caches_on();
    int ok = check(1) & check(0);
    printk(ok ? "TG_CHECK_OK\n" : "TG_CHECK_FAILED\n");
    bench(1);
    bench(0);
    unsigned same = 0;
    while (same < TG_NGEN && bench_ids[0][same] == bench_ids[1][same]) same++;
    printk("TG_BENCH_MATCH tokens=%d of=%d\n", same, TG_NGEN);
#if TG_CHAT
    if (ok)
        chat();
#endif
    arm_caches_off();
    printk("PITON_DONE\n");
}
