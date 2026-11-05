// Scaled dot-product attention (one head) on the VideoCore IV QPUs and the ARM1176.
//
// examples/attention_pi.py generates the kernels and the case table:
//   attn_kernels[]   PIton-compiled attn_fwd_kernel builds (head dim, tile, causal, K layout)
//   attn_problems[]  N_CTX x HEAD_DIM problems: LCG inputs generated here, or the
//                    embedded position/token pattern with V = I (the output is the
//                    attention matrix itself)
//   attn_cases[]     a kernel run on a problem
// The ARM phase (data cache on) computes every problem's reference with the
// straightforward C baseline and times it; then, with the caches off again, every
// case runs on the QPUs, is checked against the ARM reference over all outputs,
// timed, checked again, and a sample of its outputs is printed for the host to
// compare with float64 numpy.
#include "rpi.h"
#include "piton_rt.h"
#include "arm_sgemm.h"
#include "arm_attention.h"
#include "attn_cases.h"
#include "attn_data.h"

#if ATTN_REPEATS < 1 || ATTN_ARM_REPEATS < 1
#error "repeats must be positive"
#endif

enum { MAX_N = ATTN_MAX_N, MAX_D = 64, PAT_N = 64 };

static float arm_q[2][MAX_N * MAX_D], arm_k[2][MAX_N * MAX_D], arm_v[2][MAX_N * MAX_D];
static float pat_v[PAT_N * PAT_N];
static float arm_ref[ATTN_REF_FLOATS];
static float arm_tmp[MAX_N * MAX_D];
static float scores[MAX_N];
static unsigned ref_offset[ATTN_PROBLEM_COUNT];

static piton_buf_t g_q[2], g_k[2], g_kt[2], g_v[2], g_pat, g_pat_v, g_out;

typedef union { float f; uint32_t u; } fbits;

static float absf(float x) { return x < 0.0f ? -x : x; }

static int finitef(float x) {
    fbits b = { .f = x };
    return (b.u & 0x7f800000u) != 0x7f800000u;
}

static unsigned set_of(unsigned head_dim) { return head_dim == 64; }

static piton_buf_t upload(const float *src, unsigned count) {
    piton_buf_t b = piton_alloc(count * sizeof(float));
    volatile float *dst = b.cpu;
    for (unsigned i = 0; i < count; i++)
        dst[i] = src[i];
    return b;
}

static void generate_inputs(void) {
    static const unsigned dims[2] = {32, 64};
    for (unsigned s = 0; s < 2; s++) {
        unsigned count = MAX_N * dims[s];
        uint32_t sq = ATTN_SEED_Q + dims[s], sk = ATTN_SEED_K + dims[s], sv = ATTN_SEED_V + dims[s];
        for (unsigned i = 0; i < count; i++) {
            arm_q[s][i] = ATTN_Q_GAIN * piton_lcg_float(&sq);
            arm_k[s][i] = piton_lcg_float(&sk);
            arm_v[s][i] = piton_lcg_float(&sv);
        }
        g_q[s] = upload(arm_q[s], count);
        g_k[s] = upload(arm_k[s], count);
        g_v[s] = upload(arm_v[s], count);
        // K pre-transposed, [HEAD_DIM, MAX_N] (the "dn" kernels are built for N_CTX = MAX_N)
        g_kt[s] = piton_alloc(count * sizeof(float));
        volatile float *kt = g_kt[s].cpu;
        for (unsigned j = 0; j < MAX_N; j++)
            for (unsigned c = 0; c < dims[s]; c++)
                kt[c * MAX_N + j] = arm_k[s][j * dims[s] + c];
    }
    for (unsigned i = 0; i < PAT_N * PAT_N; i++)
        pat_v[i] = 0.0f;
    for (unsigned i = 0; i < PAT_N; i++)
        pat_v[i * PAT_N + i] = 1.0f;
    g_pat = upload(attn_pattern_qk, PAT_N * PAT_N);
    g_pat_v = upload(pat_v, PAT_N * PAT_N);
    g_out = piton_alloc(MAX_N * MAX_D * sizeof(float));
}

static void problem_inputs(const attn_problem_t *p, const float **q, const float **k, const float **v) {
    if (p->inputs == ATTN_INPUTS_PATTERN) {
        *q = *k = attn_pattern_qk;
        *v = pat_v;
    } else {
        unsigned s = set_of(p->head_dim);
        *q = arm_q[s];
        *k = arm_k[s];
        *v = arm_v[s];
    }
}

// ARM phase: references and timings (data cache on, no mailbox or GPU work).
static void arm_phase(void) {
    unsigned offset = 0;
    for (unsigned i = 0; i < ATTN_PROBLEM_COUNT; i++) {
        ref_offset[i] = offset;
        offset += attn_problems[i].n_ctx * attn_problems[i].head_dim;
    }
    arm_caches_on();
    for (unsigned i = 0; i < ATTN_PROBLEM_COUNT; i++) {
        const attn_problem_t *p = &attn_problems[i];
        const float *q, *k, *v;
        problem_inputs(p, &q, &k, &v);
        arm_attention(q, k, v, arm_ref + ref_offset[i], p->n_ctx, p->head_dim, p->causal, p->sm_scale, scores);
        for (unsigned r = 0; r < ATTN_ARM_REPEATS; r++) {
            uint32_t t0 = timer_get_usec();
            arm_attention(q, k, v, arm_tmp, p->n_ctx, p->head_dim, p->causal, p->sm_scale, scores);
            uint32_t us = timer_get_usec() - t0;
            printk("ATTN_ARM problem=%d us=%d\n", i, us);
        }
        unsigned differ = 0;
        for (unsigned j = 0; j < p->n_ctx * p->head_dim; j++) {
            fbits a = { .f = arm_tmp[j] }, b = { .f = arm_ref[ref_offset[i] + j] };
            differ += a.u != b.u;
        }
        printk("ATTN_ARM_REPEAT problem=%d differing=%d\n", i, differ);
    }
    arm_caches_off();
}

static void fill_nan(volatile float *x, unsigned n) {
    fbits nan = { .u = 0x7fc00000u };
    for (unsigned i = 0; i < n; i++)
        x[i] = nan.f;
}

// Every output against the ARM reference: |gpu - arm| <= ATOL + RTOL |arm|.
static int check(unsigned c, const char *pass) {
    const attn_case_t *cs = &attn_cases[c];
    const attn_problem_t *p = &attn_problems[cs->problem];
    volatile const float *got = g_out.cpu;
    const float *ref = arm_ref + ref_offset[cs->problem];
    unsigned n = p->n_ctx * p->head_dim, failures = 0, nonfinite = 0;
    float max_err = 0.0f;
    for (unsigned i = 0; i < n; i++) {
        float g = got[i], a = ref[i], e = absf(g - a);
        if (!finitef(g)) {
            nonfinite++;
            failures++;
            continue;
        }
        if (!(e <= ATTN_ATOL + ATTN_RTOL * absf(a)))
            failures++;
        if (e > max_err)
            max_err = e;
    }
    fbits b = { .f = max_err };
    printk("ATTN_CHECK case=%d pass=%s checked=%d failures=%d nonfinite=%d max_abs_err_bits=%x\n",
           c, pass, n, failures, nonfinite, b.u);
    return failures == 0;
}

static void print_samples(unsigned c) {
    const attn_case_t *cs = &attn_cases[c];
    const attn_problem_t *p = &attn_problems[cs->problem];
    volatile const uint32_t *w = g_out.cpu;
    unsigned n = p->n_ctx * p->head_dim, count = 0;
    for (unsigned i = 0; i < n; i += cs->sample_stride)
        count++;
    printk("ATTN_SAMPLE case=%d stride=%d count=%d\n", c, cs->sample_stride, count);
    unsigned col = 0;
    for (unsigned i = 0; i < n; i += cs->sample_stride) {
        printk("%x ", w[i]);
        if (++col == 8) {
            printk("\n");
            col = 0;
        }
    }
    if (col)
        printk("\n");
    printk("ATTN_SAMPLE_END case=%d\n", c);
}

static int run_case(unsigned c) {
    const attn_case_t *cs = &attn_cases[c];
    const attn_kernel_t *kn = &attn_kernels[cs->kernel];
    const attn_problem_t *p = &attn_problems[cs->problem];
    attn_call_t call = { .out = g_out.bus, .sm_scale = p->sm_scale, .n_ctx = p->n_ctx,
                         .grid_x = cs->grid_x };
    if (p->inputs == ATTN_INPUTS_PATTERN) {
        call.q = call.k = g_pat.bus;
        call.v = g_pat_v.bus;
    } else {
        unsigned s = set_of(p->head_dim);
        call.q = g_q[s].bus;
        call.k = kn->k_transposed ? g_kt[s].bus : g_k[s].bus;
        call.v = g_v[s].bus;
    }
    fill_nan(g_out.cpu, MAX_N * MAX_D);
    uint32_t t0 = timer_get_usec();
    kn->launch(&call);           // the first call also allocates the code and uniforms
    printk("ATTN_FIRST case=%d us=%d\n", c, timer_get_usec() - t0);
    int ok = check(c, "first");
    for (unsigned r = 0; r < ATTN_REPEATS; r++) {
        t0 = timer_get_usec();
        kn->launch(&call);
        uint32_t us = timer_get_usec() - t0;
        printk("ATTN_TIME case=%d us=%d\n", c, us);
    }
    ok &= check(c, "timed");
    print_samples(c);
    return ok;
}

void notmain(void) {
    piton_init();
    printk("PIton attention: softmax(Q K^T / sqrt(d)) V, %d kernels, %d problems, %d cases, %d QPUs\n",
           ATTN_KERNEL_COUNT, ATTN_PROBLEM_COUNT, ATTN_CASE_COUNT, piton_qpu_count());
    static const unsigned clocks[] = {PITON_CLOCK_ARM, PITON_CLOCK_CORE, PITON_CLOCK_V3D, PITON_CLOCK_SDRAM};
    static const char *names[] = {"arm", "core", "v3d", "sdram"};
    for (unsigned i = 0; i < 4; i++)
        printk("ATTN_CLOCK name=%s mhz=%d\n", names[i], piton_clock_hz(clocks[i], 0) / 1000000);
    printk("ATTN_CONFIG repeats=%d arm_repeats=%d\n", ATTN_REPEATS, ATTN_ARM_REPEATS);

    generate_inputs();
    arm_phase();
    int ok = 1;
    for (unsigned c = 0; c < ATTN_CASE_COUNT; c++)
        ok &= run_case(c);
    printk(ok ? "ATTN_CHECK_OK\n" : "ATTN_CHECK_FAILED\n");
    printk("PITON_DONE\n");
}
