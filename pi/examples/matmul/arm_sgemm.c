#include "rpi.h"
#include "arm_sgemm.h"

// ---------------------------------------------------------------------------
// MMU + caches (ARMv6 section descriptors, ARM1176)
// ---------------------------------------------------------------------------

static uint32_t page_table[4096] __attribute__((aligned(16384)));
static uint32_t saved_fpscr;

#define CACHED_LIMIT 0x08000000u        // program, bss, stack (STACK_ADDR = 0x8000000)

static inline void sync_barrier(void) { asm volatile("mcr p15, 0, %0, c7, c10, 4" :: "r"(0) : "memory"); }
static inline void prefetch_flush_(void) { asm volatile("mcr p15, 0, %0, c7, c5, 4" :: "r"(0) : "memory"); }

void arm_caches_on(void) {
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

void arm_caches_off(void) {
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
// cycle counter
// ---------------------------------------------------------------------------

uint32_t arm_cycles(void) {
    uint32_t c;
    asm volatile("mrc p15, 0, %0, c15, c12, 1" : "=r"(c));
    return c;
}

uint32_t arm_clock_mhz(void) {
    asm volatile("mcr p15, 0, %0, c15, c12, 0" :: "r"(1u | 4u));        // enable, reset CCNT
    uint32_t t0 = timer_get_usec(), c0 = arm_cycles();
    while (timer_get_usec() - t0 < 100000)
        ;
    uint32_t c1 = arm_cycles(), t1 = timer_get_usec();
    return (uint32_t)((float)(c1 - c0) / (float)(t1 - t0) + 0.5f);   // (no integer divide)
}

// ---------------------------------------------------------------------------
// matmuls
// ---------------------------------------------------------------------------

void arm_matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        float *c = C + i * N;
        for (int j = 0; j < N; j++)
            c[j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            const float *b = B + k * N;
            for (int j = 0; j < N; j++)
                c[j] += a * b[j];
        }
    }
}

// c[4][4] (row stride ldc) += ap[kc][4]^T bp[kc][4]: 16 accumulators in VFP registers
static void kernel4x4(const float *restrict ap, const float *restrict bp, float *restrict c, int ldc, int kc) {
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;
    for (int k = 0; k < kc; k++) {
        if ((k & 1) == 0)
            __builtin_prefetch(ap + 32);          // the A panel streams from DRAM
        float a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
        float b0 = bp[0], b1 = bp[1], b2 = bp[2], b3 = bp[3];
        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        ap += 4;
        bp += 4;
    }
    c[0] += c00; c[1] += c01; c[2] += c02; c[3] += c03;
    c += ldc;
    c[0] += c10; c[1] += c11; c[2] += c12; c[3] += c13;
    c += ldc;
    c[0] += c20; c[1] += c21; c[2] += c22; c[3] += c23;
    c += ldc;
    c[0] += c30; c[1] += c31; c[2] += c32; c[3] += c33;
}

void arm_kernel4x4_vec(const float *ap, const float *bp, float *c, int ldc_bytes, int kc);  // .S

void arm_matmul_blocked(const float *A, const float *B, float *C, int M, int N, int K,
                        float *pa, float *pb, int vector_kernel) {
    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += ARM_KC) {
        int kc = K - k0 < ARM_KC ? K - k0 : ARM_KC;
        // A rows i0..i0+3, columns k0..k0+kc as pa[i0/4][k][r]
        for (int i0 = 0; i0 < M; i0 += 4) {
            float *p = pa + i0 * kc;
            for (int k = 0; k < kc; k++)
                for (int r = 0; r < 4; r++)
                    *p++ = A[(i0 + r) * K + k0 + k];
        }
        for (int j0 = 0; j0 < N; j0 += ARM_NC) {
            int nc = N - j0 < ARM_NC ? N - j0 : ARM_NC;
            // B rows k0..k0+kc, columns j0..j0+nc as pb[jj/4][k][c] (stays in L1)
            for (int jj = 0; jj < nc; jj += 4) {
                float *p = pb + jj * kc;
                for (int k = 0; k < kc; k++)
                    for (int c = 0; c < 4; c++)
                        *p++ = B[(k0 + k) * N + j0 + jj + c];
            }
            for (int i0 = 0; i0 < M; i0 += 4)
                for (int jj = 0; jj < nc; jj += 4) {
                    if (vector_kernel && kc % 2 == 0)
                        arm_kernel4x4_vec(pa + i0 * kc, pb + jj * kc, C + i0 * N + j0 + jj, 4 * N, kc);
                    else
                        kernel4x4(pa + i0 * kc, pb + jj * kc, C + i0 * N + j0 + jj, N, kc);
                }
        }
    }
}
